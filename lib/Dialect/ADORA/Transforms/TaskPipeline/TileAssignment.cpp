//===- TileAssignment.cpp - per-kernel tile assignment (impl) -------------===//
//
// See TileAssignment.h.  Pipeline:
//   collect kernels -> count compute/io nodes (lightweight walk) ->
//   minTiles = ceil(computeNodes / pePerTile) -> build request JSON ->
//   callRanker (reused comms) -> parse per_kernel_tile -> validate / fallback
//   -> KernelOp.setAttr("adora.tile_set", DenseI64ArrayAttr).
//
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Transforms/TaskPipeline/TileAssignment.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Analysis/LLMRankerClient.h"
#include "ADORA/Dialect/ADORA/Analysis/DepSummaryView.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/Debug.h"

#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#define DEBUG_TYPE "llm-tile-assign"

namespace mlir {
namespace ADORA {

namespace {

// Attribute name written onto KernelOp.
static constexpr llvm::StringLiteral kTileSetAttr = "adora.tile_set";

// One kernel's resource estimate + decision scratch.
struct KernelTileDesc {
  ADORA::KernelOp op;
  std::string     name;
  int             computeNodes = 0;
  int             ioNodes      = 0;
  int             minTiles     = 1;
  std::vector<int> tiles;  // final decision
};

/// Count compute (GPE) and io (IOB) nodes a kernel would produce, WITHOUT
/// building a full DFG.  Mirrors the mapper's metric (mapper.cpp:465) and the
/// op->type mapping in lib/DFG/Documents/GeneralOpName.txt.
static void countKernelResource(ADORA::KernelOp kernel,
                                int &computeNodes, int &ioNodes) {
  computeNodes = 0;
  ioNodes = 0;
  kernel.walk([&](mlir::Operation *op) {
    llvm::StringRef dialect = op->getDialect()
                                  ? op->getDialect()->getNamespace()
                                  : llvm::StringRef();
    if (dialect == "arith") {
      // arith.constant is a CONST node, not a GPE -> exclude.
      if (!mlir::isa<mlir::arith::ConstantOp>(op))
        computeNodes++;
    } else if (mlir::isa<mlir::affine::AffineLoadOp,
                         mlir::affine::AffineStoreOp>(op)) {
      ioNodes++;
    }
    // affine.for / yield / apply are loop scaffolding -> not counted.
  });
}

static std::string jsonStr(llvm::StringRef s) { return "\"" + s.str() + "\""; }

/// Build the tile-assign request JSON.
static std::string buildTileRequestJson(const std::vector<KernelTileDesc> &ks,
                                        int numTiles, int pePerTile) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"phase\": \"tile_assign\",\n";
  os << "  \"num_tiles\": " << numTiles << ",\n";
  os << "  \"pe_per_tile\": " << pePerTile << ",\n";
  os << "  \"kernels\": [\n";
  for (size_t i = 0; i < ks.size(); ++i) {
    const auto &k = ks[i];
    os << "    {\"name\": " << jsonStr(k.name)
       << ", \"compute_nodes\": " << k.computeNodes
       << ", \"io_nodes\": " << k.ioNodes
       << ", \"min_tiles\": " << k.minTiles
       << ", \"deps_on\": []}";  // P1: cross-kernel deps left empty for now
    if (i + 1 < ks.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";
  os << "  \"response_schema\": "
        "\"{per_kernel_tile:[{kernel:<name>,tiles:[<int>]}]}\"\n";
  os << "}\n";
  return os.str();
}

/// Parse {"per_kernel_tile":[{"kernel":"k0","tiles":[0,1]},...]} from the raw
/// LLM response into name -> tile-id list.  String-scan, no JSON dep (mirrors
/// LLMPipelineSchedule.cpp's parseDepTypeResponse style).
static std::map<std::string, std::vector<int>>
parseTileResponse(const std::string &resp) {
  std::map<std::string, std::vector<int>> out;
  size_t pos = resp.find("\"per_kernel_tile\"");
  size_t scan = pos;
  while (scan != std::string::npos) {
    size_t kn = resp.find("\"kernel\"", scan);
    if (kn == std::string::npos) break;
    size_t q1 = resp.find('"', resp.find(':', kn) + 1);
    size_t q2 = (q1 == std::string::npos) ? std::string::npos
                                          : resp.find('"', q1 + 1);
    if (q1 == std::string::npos || q2 == std::string::npos) break;
    std::string name = resp.substr(q1 + 1, q2 - q1 - 1);

    size_t ts = resp.find("\"tiles\"", q2);
    if (ts == std::string::npos) break;
    size_t lb = resp.find('[', ts);
    size_t rb = (lb == std::string::npos) ? std::string::npos
                                          : resp.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) break;
    std::vector<int> tiles;
    std::string inner = resp.substr(lb + 1, rb - lb - 1);
    std::istringstream is(inner);
    std::string tok;
    while (std::getline(is, tok, ',')) {
      // trim leading spaces
      size_t a = tok.find_first_not_of(" \t\n");
      if (a == std::string::npos) continue;
      char *end = nullptr;
      long v = std::strtol(tok.c_str() + a, &end, 10);
      if (end != tok.c_str() + a) tiles.push_back(static_cast<int>(v));
    }
    out[name] = tiles;
    scan = rb + 1;
  }
  return out;
}

} // namespace

void assignTiles(mlir::func::FuncOp func, llvm::StringRef rankerCmd,
                 int timeoutMs, int numTiles, int pePerTile,
                 bool dryRun, llvm::StringRef logPath) {
  if (numTiles < 1) numTiles = 1;
  if (pePerTile < 1) pePerTile = 1;

  // ---- collect kernels + count resources + minTiles ----
  std::vector<KernelTileDesc> kernels;
  func.walk([&](ADORA::KernelOp kernel) {
    KernelTileDesc d;
    d.op   = kernel;
    d.name = kernel.getKernelName();
    countKernelResource(kernel, d.computeNodes, d.ioNodes);
    int mt = (d.computeNodes + pePerTile - 1) / pePerTile;  // ceil
    if (mt < 1) mt = 1;
    if (mt > numTiles) mt = numTiles;
    d.minTiles = mt;
    kernels.push_back(d);
  });

  if (kernels.empty()) return;

  // ---- default decision = [0 .. minTiles-1] for each kernel ----
  for (auto &k : kernels) {
    k.tiles.clear();
    for (int i = 0; i < k.minTiles; ++i) k.tiles.push_back(i);
  }

  // ---- query LLM (unless dry-run / no cmd) ----
  bool useLLM = !dryRun && !rankerCmd.empty();
  if (useLLM) {
    std::string req = buildTileRequestJson(kernels, numTiles, pePerTile);
    RankerRawResult raw = callRanker(rankerCmd, req, timeoutMs);
    if (!raw.ok) {
      func.emitWarning("TileAssignment: ranker failed (" + raw.error +
                       "), using default tile_set");
    } else {
      auto choices = parseTileResponse(raw.response);
      for (auto &k : kernels) {
        auto it = choices.find(k.name);
        if (it == choices.end()) continue;  // keep default
        const std::vector<int> &t = it->second;
        // validate: all ids in [0,numTiles) AND size >= minTiles
        bool legal = ((int)t.size() >= k.minTiles);
        for (int id : t)
          if (id < 0 || id >= numTiles) { legal = false; break; }
        if (legal)
          k.tiles = t;  // accept; else keep default [0..minTiles-1]
      }
      if (!logPath.empty()) {
        std::ofstream log(logPath.str(), std::ios::app);
        log << "{\"func\":\"" << func.getName().str()
            << "\",\"per_kernel_tile\":[";
        for (size_t i = 0; i < kernels.size(); ++i) {
          if (i) log << ",";
          log << "{\"kernel\":\"" << kernels[i].name << "\",\"tiles\":[";
          for (size_t j = 0; j < kernels[i].tiles.size(); ++j) {
            if (j) log << ",";
            log << kernels[i].tiles[j];
          }
          log << "]}";
        }
        log << "],\"scratchpad\":\"" << raw.scratchpad << "\"}\n";
      }
    }
  }

  // ---- write adora.tile_set on each kernel ----
  mlir::Builder builder(func.getContext());
  for (auto &k : kernels) {
    llvm::SmallVector<int64_t> t64(k.tiles.begin(), k.tiles.end());
    k.op->setAttr(kTileSetAttr, builder.getDenseI64ArrayAttr(t64));
  }
}

} // namespace ADORA
} // namespace mlir

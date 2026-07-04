//===- LLMPipelineSchedule.cpp - LLM-guided task pipeline scheduling ------===//
//
// Reads the adora.dep_summary written by ScheduleAdoraTasks and lets an
// external LLM ranker (task_schedule_ranker.py) select a dep_type assignment
// for each task pair.  The selected dep_types are written as "hw_dep_type"
// StringAttr on the relevant DataBlockLoadOp / DataBlockStoreOp / KernelOp
// so that downstream emitters (EmitCGRACall, EmitPytest) pick them up.
//
// Pass options (all optional - defaults give safe no-op behaviour):
//   adora-llm-pipeline-schedule-ranker-cmd      argv for task_schedule_ranker.py
//   adora-llm-pipeline-schedule-ranker-timeout  per-call timeout in ms (default 10000)
//   adora-llm-pipeline-schedule-ranker-log      NDJSON decision log path
//   adora-llm-pipeline-schedule-dry-run         always use plan idx 0, skip LLM call
//
// When ranker-cmd is empty (or dry-run is true) the pass runs in dry-run mode:
// it still walks the IR and computes the legal candidate set but always selects
// plan index 0 (conservative serial schedule).  The ranker subprocess is
// invoked via fork+exec with a bidirectional pipe and a poll-based timeout
// (mirrors mapper/src/mapper/online_ranker.cpp::rankOnce) -- the request JSON
// is written to the child's stdin and the one-line JSON response is read back
// from its stdout.  API key / endpoint / model are passed to the child via the
// inherited environment (OPENAI_API_KEY, PTL_BASE_URL, PTL_MODEL), never baked
// into the binary or the IR.
//
//===----------------------------------------------------------------------===//

// Dialect headers MUST precede PassDetail.h (which includes Passes.h.inc).
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Analysis/LLMRankerClient.h"
#include "ADORA/Dialect/ADORA/Transforms/TaskPipeline/TileAssignment.h"
#include "../PassDetail.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#define DEBUG_TYPE "adora-llm-pipeline-schedule"

using namespace mlir;
using namespace mlir::ADORA;

// ---------------------------------------------------------------------------
// Global pass options (command-line flags)
// ---------------------------------------------------------------------------

static llvm::cl::opt<std::string> clRankerCmd(
    "adora-llm-pipeline-schedule-ranker-cmd",
    llvm::cl::desc("argv for task_schedule_ranker.py (empty = dry-run)"),
    llvm::cl::init(""));

static llvm::cl::opt<int> clRankerTimeoutMs(
    "adora-llm-pipeline-schedule-ranker-timeout",
    llvm::cl::desc("LLM ranker call timeout in ms"),
    llvm::cl::init(10000));

static llvm::cl::opt<std::string> clRankerLog(
    "adora-llm-pipeline-schedule-ranker-log",
    llvm::cl::desc("NDJSON log path for ranker decisions"),
    llvm::cl::init(""));

static llvm::cl::opt<bool> clDryRun(
    "adora-llm-pipeline-schedule-dry-run",
    llvm::cl::desc("Always use default plan (idx 0), skip LLM call"),
    llvm::cl::init(false));

// Tile assignment needs to know the CGRA geometry, which lives in the ADG
// (mapper module) and is NOT visible to this dialect-library pass.  cgra-mapper
// injects these by assigning to the cl::opts (extern) after reading the ADG.
llvm::cl::opt<int> clNumTiles(
    "adora-llm-pipeline-schedule-num-tiles",
    llvm::cl::desc("total CGRA tiles (= adg->tileNum())"),
    llvm::cl::init(1));

llvm::cl::opt<int> clPePerTile(
    "adora-llm-pipeline-schedule-pe-per-tile",
    llvm::cl::desc("GPEs per tile (= adg->numGpeNodes()/adg->tileNum())"),
    llvm::cl::init(16));

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Dep-type string constants that mirror the hardware ISA.
static constexpr llvm::StringLiteral kDepNone   = "LD_DEP_NONE";
static constexpr llvm::StringLiteral kDepExLast = "LD_DEP_EX_LAST_TASK";
static constexpr llvm::StringLiteral kDepStLast = "LD_DEP_ST_LAST_TASK";

/// Attribute name written onto load/store/kernel ops.
static constexpr llvm::StringLiteral kHwDepTypeAttr = "hw_dep_type";

// ---------------------------------------------------------------------------
// Task descriptor collected during IR walk
// ---------------------------------------------------------------------------

struct TaskDesc {
  int taskIdx = -1;
  ADORA::DataBlockLoadOp  loadOp;
  ADORA::KernelOp         kernelOp;
  ADORA::DataBlockStoreOp storeOp;

  // Data filled during analysis
  bool rawDepOnPrev = false;  // true if loadOp reads the store of ANY earlier
                              // task (not only the immediately previous one) --
                              // covers fork-join / diamond deps, not just chains
  bool bankConflict = false;  // true if active banks overlap (from ADG)
  // Task indices this task RAW-depends on (its load reads their store). For a
  // straight-line chain this is {i-1}; for fork-join (e.g. gesummv merge reads
  // Stage1 AND Stage2) it can be multiple / non-adjacent predecessors.
  std::vector<int> depPreds;
  // True if this task is a single kernel inside an affine.for that carries
  // !ADORA.token iter_args (iteration-to-iteration kernel reuse / software
  // pipelining).  For such a task "prev" means the PREVIOUS LOOP ITERATION,
  // not the previous textual task; rawDepOnPrev is then a cross-iteration RAW.
  bool isLoopCarried = false;

  // Legal dep_type options for the LOAD of this task (vs previous task).
  // [0] is always the conservative default (ST_LAST_TASK).
  std::vector<std::string> legalOptions;
};

// ---------------------------------------------------------------------------
// Simple JSON helpers (no external deps)
// ---------------------------------------------------------------------------

static std::string jsonStr(llvm::StringRef s) { return "\"" + s.str() + "\""; }

static std::string buildRequestJson(llvm::StringRef kernelName,
                                    const std::vector<TaskDesc> &tasks) {
  std::ostringstream os;
  os << "{\n";
  os << "  \"schema_version\": \"schedule-online-v0\",\n";
  os << "  \"phase\": \"schedule_select\",\n";
  os << "  \"kernel\": " << jsonStr(kernelName) << ",\n";

  os << "  \"task_graph\": [\n";
  for (size_t i = 0; i < tasks.size(); ++i) {
    const auto &t = tasks[i];
    os << "    {\"task_id\": " << t.taskIdx
       << ", \"raw_dep_on_prev\": " << (t.rawDepOnPrev ? "true" : "false")
       << ", \"dep_preds\": [";
    for (size_t k = 0; k < t.depPreds.size(); ++k) {
      os << t.depPreds[k];
      if (k + 1 < t.depPreds.size()) os << ", ";
    }
    os << "]"
       << ", \"is_loop_carried\": " << (t.isLoopCarried ? "true" : "false")
       << ", \"legal_options\": [";
    for (size_t j = 0; j < t.legalOptions.size(); ++j) {
      os << jsonStr(t.legalOptions[j]);
      if (j + 1 < t.legalOptions.size()) os << ", ";
    }
    os << "]}";
    if (i + 1 < tasks.size()) os << ",";
    os << "\n";
  }
  os << "  ],\n";

  // P1: per-task selection (Gamma dimension A).
  // The LLM picks a dep_type for EACH task from its legal_options above,
  // instead of choosing one of two coarse whole-kernel plans. This lets it
  // express mixed schedules (e.g. task1 overlap, task2 serial) and makes
  // EX_LAST_TASK actually selectable.
  // Response schema (per task):
  //   {"per_task_choices": [{"task_id": <int>, "dep_type": "<LD_DEP_*>"}, ...]}
  // The conservative default a_0 = all ST_LAST_TASK (used on any miss).
  os << "  \"decision\": \"per_task_dep_type\",\n";
  os << "  \"response_schema\": \"{per_task_choices:[{task_id,dep_type}]}\",\n";
  os << "  \"default_dep_type\": \"" << kDepStLast.str() << "\"\n";
  os << "}\n";
  return os.str();
}

// ---------------------------------------------------------------------------
// External ranker call: fork + exec + bidirectional pipe + poll timeout.
// Mirrors mapper/src/mapper/online_ranker.cpp::rankOnce.  We deliberately
// inline this (rather than link CGRAMapperLIB) to keep MLIRADORATaskPipeline
// free of any dependency on the mapper module.
// ---------------------------------------------------------------------------

// dep_type 决策专用的响应结构 + 解析。通信由 Analysis/LLMRankerClient.h 的
// callRanker 负责（返回原始字符串），这里只负责把原始响应解析成 dep_type 选择。
struct RankerResponse {
  bool        ok          = false;
  int         selectedIdx = -1;  // legacy whole-plan idx (compat); -1 = unset
  // P1: per-task choices, task_id -> dep_type string. Empty => fall back to
  // legacy selectedIdx interpretation, then to a_0.
  std::map<int, std::string> perTaskChoices;
  std::string scratchpad;
  std::string error;
};

/// 解析 callRanker 返回的原始响应字符串，填出 dep_type 决策。
/// raw 为通用通信层的结果（含 response / scratchpad / error / ok）。
static RankerResponse parseDepTypeResponse(const RankerRawResult &raw) {
  RankerResponse resp;
  resp.scratchpad = raw.scratchpad;  // 通用推理日志，由通信层解析好

  if (!raw.ok) {
    resp.error = raw.error;
    return resp;
  }
  const std::string &response = raw.response;

  // P1: prefer per_task_choices: [{"task_id":i,"dep_type":"LD_DEP_*"}, ...]
  // Scan all {"task_id": <int> ... "dep_type": "<str>"} pairs.
  {
    size_t cur = response.find("\"per_task_choices\"");
    size_t scanFrom = (cur == std::string::npos) ? std::string::npos : cur;
    while (scanFrom != std::string::npos) {
      size_t tk = response.find("\"task_id\"", scanFrom);
      if (tk == std::string::npos) break;
      size_t tkColon = response.find(':', tk);
      if (tkColon == std::string::npos) break;
      int taskId = std::stoi(response.substr(tkColon + 1));
      size_t dt = response.find("\"dep_type\"", tkColon);
      if (dt == std::string::npos) break;
      size_t q1 = response.find('"', response.find(':', dt) + 1);
      size_t q2 = (q1 == std::string::npos) ? std::string::npos
                                            : response.find('"', q1 + 1);
      if (q1 == std::string::npos || q2 == std::string::npos) break;
      resp.perTaskChoices[taskId] = response.substr(q1 + 1, q2 - q1 - 1);
      scanFrom = q2 + 1;
    }
  }

  // Legacy / fallback: {"selected_idx": N}
  if (resp.perTaskChoices.empty()) {
    auto pos = response.find("\"selected_idx\"");
    if (pos == std::string::npos) {
      resp.error = "no per_task_choices and no selected_idx in response: " +
                   response;
      return resp;
    }
    auto colon = response.find(':', pos);
    if (colon == std::string::npos) {
      resp.error = "parse error (no colon after selected_idx)";
      return resp;
    }
    resp.selectedIdx = std::stoi(response.substr(colon + 1));
  }
  resp.ok = true;
  return resp;
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct LLMPipelineSchedulePass
    : public LLMPipelineScheduleBase<LLMPipelineSchedulePass> {

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    mlir::Builder builder(func.getContext());

    // ---- Step 0: per-kernel tile assignment (writes adora.tile_set) ----
    // Independent kernels can be placed on different tiles to overlap.  Done
    // first so downstream stages (and dep_type below) see the tile decision.
    ADORA::assignTiles(func, clRankerCmd.getValue(),
                       clRankerTimeoutMs.getValue(), clNumTiles.getValue(),
                       clPePerTile.getValue(), clDryRun.getValue(),
                       clRankerLog.getValue());

    // ---- Step 1: collect tasks in program order ----
    std::vector<TaskDesc> tasks;
    {
      int idx = 0;
      func.walk([&](Block *block) {
        ADORA::DataBlockLoadOp pendingLoad;
        ADORA::KernelOp        pendingKernel;

        for (auto &op : *block) {
          if (auto load = dyn_cast<ADORA::DataBlockLoadOp>(&op)) {
            pendingLoad = load;
          } else if (auto kern = dyn_cast<ADORA::KernelOp>(&op)) {
            pendingKernel = kern;
          } else if (auto store = dyn_cast<ADORA::DataBlockStoreOp>(&op)) {
            if (pendingLoad || pendingKernel) {
              TaskDesc td;
              td.taskIdx  = idx++;
              td.loadOp   = pendingLoad;
              td.kernelOp = pendingKernel;
              td.storeOp  = store;
              // Mark loop-carried tasks in place: a triple whose enclosing
              // block is an affine.for carrying !ADORA.token iter_args is a
              // single kernel reused across iterations (software pipelining).
              // "prev" for such a task = the previous LOOP ITERATION.
              if (auto forOp =
                      dyn_cast<mlir::affine::AffineForOp>(block->getParentOp())) {
                for (mlir::BlockArgument a : forOp.getRegionIterArgs()) {
                  if (mlir::isa<ADORA::TokenType>(a.getType())) {
                    td.isLoopCarried = true;
                    break;
                  }
                }
              }
              tasks.push_back(td);
              pendingLoad   = {};
              pendingKernel = {};
            }
          }
        }
      });
    }

    // A single loop-carried task still forms a pipeline pair with its own
    // previous iteration, so it is decidable on its own.  Straight-line tasks
    // need at least 2 to have anything to pipeline.
    bool hasLoopCarried = false;
    for (const auto &t : tasks)
      if (t.isLoopCarried) { hasLoopCarried = true; break; }
    if (tasks.empty() || (!hasLoopCarried && tasks.size() < 2))
      return;  // nothing to pipeline

    // ---- Step 2: compute RAW deps ----
    // (2a) Loop-carried tasks: the cross-iteration RAW shows up as a LOAD whose
    // async dep comes from a !ADORA.token iter_arg (a BlockArgument), NOT from a
    // sibling store op.  A loop body can have several BlockLoads (e.g. GEMM
    // loads C/A/B but only the C-tile carries the loop-carried token), and the
    // task's single loadOp is just the LAST one collected — so we must scan ALL
    // BlockLoads in the loop body, not only t.loadOp.  Per-task granularity:
    // if ANY load has a cross-iteration RAW, the whole task is serial.
    for (auto &t : tasks) {
      if (!t.isLoopCarried || !t.loadOp) continue;
      mlir::Block *body = t.loadOp->getBlock();
      for (auto &op : *body) {
        auto ld = dyn_cast<ADORA::DataBlockLoadOp>(&op);
        if (!ld) continue;
        for (mlir::Value dep : ADORA::getAsyncDeps(ld.getOperation())) {
          if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(dep)) {
            if (mlir::isa<ADORA::TokenType>(ba.getType())) {
              t.rawDepOnPrev = true;  // depends on previous iteration's store
              break;
            }
          }
        }
        if (t.rawDepOnPrev) break;
      }
    }

    // (2b) Straight-line tasks: RAW iff this task's load(s) read the store of
    // ANY earlier task -- not only the immediately previous one. This covers
    // fork-join / diamond dependencies (e.g. gesummv's merge reads Stage1 AND
    // Stage2, the latter being non-adjacent). We collect ALL such predecessors
    // into depPreds; rawDepOnPrev stays as the coarse "has any RAW predecessor"
    // flag used for legal-option generation.
    for (size_t i = 1; i < tasks.size(); ++i) {
      if (tasks[i].isLoopCarried) continue;  // handled in 2a
      if (!tasks[i].loadOp) continue;
      // async deps of this task's load -> which earlier task's store produced them
      llvm::SmallVector<mlir::Value> deps(
          ADORA::getAsyncDeps(tasks[i].loadOp.getOperation()));
      for (size_t j = 0; j < i; ++j) {
        if (!tasks[j].storeOp) continue;
        for (mlir::Value dep : deps) {
          if (dep.getDefiningOp() == tasks[j].storeOp.getOperation()) {
            tasks[i].depPreds.push_back(tasks[j].taskIdx);
            tasks[i].rawDepOnPrev = true;
            break;
          }
        }
      }
    }

    // ---- Step 3: build legal options per task (Gamma generation) ----
    // A loop-carried task is decidable even at index 0 (its "prev" is the
    // previous iteration), so it must get a legal set too.  Straight-line
    // tasks only from i>=1 (the first has no textual predecessor).
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i == 0 && !tasks[i].isLoopCarried)
        continue;  // first straight-line task: no predecessor, leave empty
      if (tasks[i].rawDepOnPrev) {
        tasks[i].legalOptions = {kDepStLast.str()};
      } else if (tasks[i].bankConflict) {
        tasks[i].legalOptions = {kDepStLast.str(), kDepExLast.str()};
      } else {
        tasks[i].legalOptions = {kDepStLast.str(), kDepExLast.str(),
                                 kDepNone.str()};
      }
    }

    // ---- Step 4: query LLM (or dry-run) ----
    std::vector<std::string> chosen(tasks.size(), kDepStLast.str());

    bool useLLM = !clDryRun && !clRankerCmd.getValue().empty();
    if (useLLM) {
      std::string kernelName = func.getName().str();
      std::string req = buildRequestJson(kernelName, tasks);
      RankerRawResult raw = callRanker(clRankerCmd.getValue(), req,
                                       clRankerTimeoutMs.getValue());
      RankerResponse resp = parseDepTypeResponse(raw);
      if (!resp.ok) {
        func.emitWarning("LLMPipelineSchedulePass: ranker failed (" +
                         resp.error + "), using default plan");
      } else if (!resp.perTaskChoices.empty()) {
        // P1: per-task choices. For each task, accept the LLM's dep_type only
        // if it is in that task's legal_options; otherwise keep a_0 (ST_LAST).
        for (size_t i = 0; i < tasks.size(); ++i) {
          if (i == 0 && !tasks[i].isLoopCarried) continue;  // no predecessor
          auto it = resp.perTaskChoices.find(tasks[i].taskIdx);
          if (it == resp.perTaskChoices.end()) continue;  // keep a_0
          const auto &opts = tasks[i].legalOptions;
          if (std::find(opts.begin(), opts.end(), it->second) != opts.end())
            chosen[i] = it->second;  // legal -> accept
          // else: illegal choice, keep a_0 (already ST_LAST)
        }
        if (!clRankerLog.getValue().empty()) {
          std::ofstream log(clRankerLog.getValue(), std::ios::app);
          log << "{\"kernel\":\"" << kernelName << "\",\"per_task_choices\":[";
          bool first = true;
          for (size_t i = 0; i < tasks.size(); ++i) {
            if (i == 0 && !tasks[i].isLoopCarried) continue;
            if (!first) log << ",";
            first = false;
            log << "{\"task_id\":" << tasks[i].taskIdx
                << ",\"dep_type\":\"" << chosen[i] << "\"}";
          }
          log << "],\"scratchpad\":\"" << resp.scratchpad << "\"}\n";
        }
      } else {
        // Legacy whole-plan fallback: idx 1 = all NONE where legal.
        int sel = resp.selectedIdx;
        if (sel == 1) {
          for (size_t i = 0; i < tasks.size(); ++i) {
            if (i == 0 && !tasks[i].isLoopCarried) continue;
            const auto &opts = tasks[i].legalOptions;
            if (std::find(opts.begin(), opts.end(), kDepNone.str()) !=
                opts.end())
              chosen[i] = kDepNone.str();
            else
              chosen[i] = opts[0];
          }
        }
        if (!clRankerLog.getValue().empty() && !resp.scratchpad.empty()) {
          std::ofstream log(clRankerLog.getValue(), std::ios::app);
          log << "{\"kernel\":\"" << kernelName
              << "\",\"selected_idx\":" << sel
              << ",\"scratchpad\":\"" << resp.scratchpad << "\"}\n";
        }
      }
    }

    // ---- Step 5: write hw_dep_type attrs ----
    // i=0 is written only when it is a loop-carried task (decidable vs its own
    // previous iteration); a straight-line first task has no predecessor.
    for (size_t i = 0; i < tasks.size(); ++i) {
      if (i == 0 && !tasks[i].isLoopCarried) continue;
      auto attr = builder.getStringAttr(chosen[i]);
      if (tasks[i].loadOp)
        tasks[i].loadOp->setAttr(kHwDepTypeAttr, attr);
      if (tasks[i].storeOp)
        tasks[i].storeOp->setAttr(kHwDepTypeAttr, attr);
    }
  }
};

}  // namespace

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ADORA::createLLMPipelineSchedulePass() {
  return std::make_unique<LLMPipelineSchedulePass>();
}

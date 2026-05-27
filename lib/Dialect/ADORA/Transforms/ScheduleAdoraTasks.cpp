//===--------------------------------------------------------------------------------------------------===//
//===- ScheduleADORATasks.cpp - Schedule ADORA CGRA tasks -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
// #include <fstream>
// #include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Analysis/LoopCarriedDep.h"
#include "ADORA/Dialect/ADORA/Transforms/ThreadLoopCarriedTokens.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DependencyAnalysis.h"
#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h"
#include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "./PassDetail.h"

using namespace llvm; // for llvm.errs()
using namespace llvm::detail;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
//===----------------------------------------------------------------------===//
// AdjustKernelMemoryFootprint to meet cachesize
//===----------------------------------------------------------------------===//

#define PASS_NAME   "adora-schedule-cgra-tasks"
#define DEBUG_TYPE  "adora-schedule-cgra-tasks"

namespace
{
struct ScheduleADORATasksPass : 
  public ScheduleADORATasksBase<ScheduleADORATasksPass>
{
public:
  bool BlockContainsKernelOp(mlir::Block* b);

  void ScheduleADORATasksInFunction(func::FuncOp func);

  void runOnOperation() override;
};
/// @brief 
/// @param b block to check whether contains a kernelop. Kernel op nested in for op is skipped.
/// @return 
bool ScheduleADORATasksPass::BlockContainsKernelOp(mlir::Block* b){
  for(auto op : b->getOps<ADORA::KernelOp>()){
    if(isa<ADORA::KernelOp>(op)){
      return true;
    }
  }

  return false;
}

void generateTaskGraphFromBlock(TaskGraph* graph, mlir::Block* block){

  /// validloads : data block which has already been loaded to on-chip memory
  std::map<ADORA::LocalMemAllocOp, LocalAllocNode*> validallocs; 

  /// validloads : data block which has already been loaded to on-chip memory
  std::map<ADORA::DataBlockLoadOp, BlockLoadNode*> validloads; 

  /// dirtystores : data block which has not been written back to main memory
  std::map<ADORA::DataBlockStoreOp, BlockStoreNode*> dirtystores;
  
  for(auto _it = block->begin(); _it != block->end(); _it++){
    mlir::Operation* op = &(*_it);
    // op->dump();
    if(isa<ADORA::KernelOp>(op)){
      ADORA::KernelOp kernelop = dyn_cast<ADORA::KernelOp>(op);
      KernelNode* kernelnode = new KernelNode(kernelop);
      graph->AddNodeAndAnalyzeDefaultDependency(kernelnode);
    }
    else if(isa<ADORA::DataBlockLoadOp>(op)){
      ADORA::DataBlockLoadOp blockloadop = dyn_cast<ADORA::DataBlockLoadOp>(op);
      BlockLoadNode* blockloadnode = new BlockLoadNode(blockloadop);
      graph->AddNodeAndAnalyzeDefaultDependency(blockloadnode);
      validloads[blockloadop] = blockloadnode;

      /// handle load-after-store dependency here
      for(auto& pair : dirtystores){
        ADORA::DataBlockStoreOp blockstoreop = pair.first;
        BlockStoreNode* blockstorenode = pair.second;
        if(checkDependencyBetweenBlockStoreAndBlockLoad(blockstoreop, blockloadop)){
          addConnectionBetweenTwoNode(blockstorenode, blockloadnode, /*dep=*/depType::Depend);
        }
      }

      // /// handle load-after-load dependency here
      // for(auto& pair : validloads){
      //   ADORA::DataBlockLoadOp visitedblockloadop = pair.first;
      //   BlockLoadNode* blockloadnode = pair.second;
      //   if(SameBlockLoad(visitedblockloadop, blockloadop)){
      //     addConnectionBetweenTwoNode(blockstorenode, blockloadnode, /*dep=*/depType::Depend);
      //   }
      // }

    }
    else if(isa<ADORA::LocalMemAllocOp>(op)){
      ADORA::LocalMemAllocOp allocop = dyn_cast<ADORA::LocalMemAllocOp>(op);
      LocalAllocNode* allocnode = new LocalAllocNode(allocop);
      graph->AddNodeAndAnalyzeDefaultDependency(allocnode);
      validallocs[allocop] = allocnode;
    }
    else if(isa<ADORA::DataBlockStoreOp>(op)){
      ADORA::DataBlockStoreOp blockstoreop = dyn_cast<ADORA::DataBlockStoreOp>(op);
      BlockStoreNode* blockstorenode = new BlockStoreNode(blockstoreop);
      graph->AddNodeAndAnalyzeDefaultDependency(blockstorenode);
      dirtystores[blockstoreop] = blockstorenode;
    }
    else{
      ///// if some stored data is used, then this store node must be written back.
    }
  }

  // analyze other dependencies
  // /// first, load-after-store
  // for(auto _it = block->begin(); _it != block->end(); _it++){
  //   mlir::Operation* op = &(*_it);
  //   op->dump();
  //   if(isa<ADORA::DataBlockLoadOp>(op)){
  //     ADORA::DataBlockLoadOp blockloadop = dyn_cast<ADORA::DataBlockLoadOp>(op);
  //     BlockLoadNode* blockloadnode = visitedloads[blockloadop];
      
  //   }
  //   else if(isa<ADORA::DataBlockStoreOp>(op)){
  //     ADORA::DataBlockStoreOp blockstoreop = dyn_cast<ADORA::DataBlockStoreOp>(op);
  //     BlockStoreNode* blockstorenode = new BlockStoreNode(blockstoreop);
  //     graph->AddNodeAndAnalyzeDefaultDependency(blockstorenode);
  //     visitedstores[blockstoreop] = blockstorenode;
  //   }
  // }    

}

/// @brief Fill P1.0 DataBlock-level dependencies into the TaskGraph.
///
/// RAW (store -> later load) is already wired in generateTaskGraphFromBlock
/// via ops' SSA use-def chain, so we do NOT re-emit it here to avoid double
/// edges. This pass additionally surfaces:
///
///   - RAR (load  -> later load ) : same backing memref + (exact | overlap)
///   - WAW (store -> later store) : same backing memref + (exact | overlap)
///   - WAR (load  -> later store) : same backing memref + (exact | overlap)
///
/// IR lexical order is taken from mlir::Operation::isBeforeInBlock when the
/// two ops share a block (always the case inside this pass: both live under
/// the same kernel-containing block). For each hit we:
///
///   1. record a DataBlockDepEdge on the graph (consumed by P4.0 serializer),
///   2. add a depType::Depend connection between the two TaskNodes (consumed
///      by the downstream scheduler).
///
/// Complexity: O(N^2) on node count; acceptable for typical kernel graphs.
/// @param graph  task graph produced by generateTaskGraphFromBlock
void analyzeDependencyInGraph(TaskGraph* graph){
  auto nodes = graph->getAllNodes();
  const size_t N = nodes.size();

  auto lexicallyBefore = [](mlir::Operation* a, mlir::Operation* b) -> bool {
    if (!a || !b) return false;
    if (a->getBlock() != b->getBlock()) return false;
    return a->isBeforeInBlock(b);
  };

  auto emit = [&](TaskNode* src, TaskNode* dst,
                  DataBlockDepKind kind, bool exact) {
    graph->addDepEdge({src, dst, kind, exact});
    addConnectionBetweenTwoNode(src, dst, /*dep=*/depType::Depend);
  };

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      if (i == j) continue;
      TaskNode* ni = nodes[i];
      TaskNode* nj = nodes[j];

      // Require a concrete IR ordering ni -> nj.
      if (!lexicallyBefore(ni->getOperation(), nj->getOperation()))
        continue;

      // --- RAR: load_i -> load_j (same original memref) ---
      if (isa<BlockLoadNode>(ni) && isa<BlockLoadNode>(nj)) {
        auto la = cast<BlockLoadNode>(ni)->getDataBlockLoadOp();
        auto lb = cast<BlockLoadNode>(nj)->getDataBlockLoadOp();
        if (checkDependencyBetweenBlockLoadAndBlockLoad(la, lb)) {
          bool exact = AccessSameDataBlock(la, lb);
          emit(ni, nj, DataBlockDepKind::RAR, exact);
        }
        continue;
      }

      // --- WAW: store_i -> store_j (same target memref) ---
      if (isa<BlockStoreNode>(ni) && isa<BlockStoreNode>(nj)) {
        auto sa = cast<BlockStoreNode>(ni)->getDataBlockStoreOp();
        auto sb = cast<BlockStoreNode>(nj)->getDataBlockStoreOp();
        if (checkDependencyBetweenBlockStoreAndBlockStore(sa, sb)) {
          bool exact = AccessSameDataBlock(sa, sb);
          emit(ni, nj, DataBlockDepKind::WAW, exact);
        }
        continue;
      }

      // --- WAR: load_i -> store_j (load's original = store's target) ---
      if (isa<BlockLoadNode>(ni) && isa<BlockStoreNode>(nj)) {
        auto la = cast<BlockLoadNode>(ni)->getDataBlockLoadOp();
        auto sb = cast<BlockStoreNode>(nj)->getDataBlockStoreOp();
        if (checkDependencyBetweenBlockLoadAndBlockStore(la, sb)) {
          bool exact = AccessSameDataBlock(sb, la); // reuse store-load exact
          emit(ni, nj, DataBlockDepKind::WAR, exact);
        }
        continue;
      }

      // --- RAW: store_i -> load_j (store's target = load's original memref) ---
      if (isa<BlockStoreNode>(ni) && isa<BlockLoadNode>(nj)) {
        auto sa = cast<BlockStoreNode>(ni)->getDataBlockStoreOp();
        auto lb = cast<BlockLoadNode>(nj)->getDataBlockLoadOp();
        if (checkDependencyBetweenBlockStoreAndBlockLoad(sa, lb)) {
          bool exact = AccessSameDataBlock(sa, lb);
          emit(ni, nj, DataBlockDepKind::RAW, exact);
        }
        continue;
      }
    }
  }
}

/// @brief P4.0 — Serialize a TaskGraph's datablock edges into ONE grouped
/// DictionaryAttr of the form:
///   { block_idx: i64, edges: [ { src, dst, kind, overlap } ... ] }
/// This is the single row appended to the top-level `adora.dep_summary`
/// ArrayAttr. `block_idx` is stored once per block instead of once per edge.
static void appendDepEdgesToAttrList(TaskGraph* graph, int blockIdx,
                                     mlir::MLIRContext* ctx,
                                     SmallVectorImpl<mlir::Attribute>& out) {
  mlir::Builder b(ctx);
  SmallVector<mlir::Attribute> edgeAttrs;
  edgeAttrs.reserve(graph->depEdges().size());
  for (const auto& e : graph->depEdges()) {
    SmallVector<mlir::NamedAttribute, 4> fields;
    fields.push_back(b.getNamedAttr("src",     b.getI64IntegerAttr(graph->getNodeId(e.src))));
    fields.push_back(b.getNamedAttr("dst",     b.getI64IntegerAttr(graph->getNodeId(e.dst))));
    fields.push_back(b.getNamedAttr("kind",    b.getStringAttr(toString(e.kind))));
    fields.push_back(b.getNamedAttr("overlap", b.getBoolAttr(e.mustOverlap)));
    edgeAttrs.push_back(b.getDictionaryAttr(fields));
  }
  SmallVector<mlir::NamedAttribute, 2> blockFields;
  blockFields.push_back(b.getNamedAttr("block_idx", b.getI64IntegerAttr(blockIdx)));
  blockFields.push_back(b.getNamedAttr("edges", b.getArrayAttr(edgeAttrs)));
  out.push_back(b.getDictionaryAttr(blockFields));
}

//===----------------------------------------------------------------------===//
// PR2 commit B — Thread SSA !ADORA.token values along dep edges.
//
// Walks graph->depEdges() and rewrites participating DataBlockLoad/Store and
// Kernel ops into their async form so that dependency edges become explicit
// MLIR happens-before relations in the use-def graph. This is the SSA-level
// counterpart of the PR1 adora.dep_summary string channel; when the pass
// option emit-token is true, the summary becomes a debug artifact only.
//
// Without tokens, downstream runtime defaults to serial execution, so the
// synchronous form remains semantically safe at all times.
//===----------------------------------------------------------------------===//

/// Attribute names already written by each op's async builder; skipped during
/// user-attribute migration to avoid double-setting.
static const llvm::StringSet<> &loadBuiltInAttrs() {
  static const llvm::StringSet<> S = {
      "map", "strides", "kernel_name", "operandSegmentSizes"};
  return S;
}
static const llvm::StringSet<> &storeBuiltInAttrs() {
  static const llvm::StringSet<> S = {
      "map", "strides", "kernel_name", "operandSegmentSizes"};
  return S;
}
static const llvm::StringSet<> &kernelBuiltInAttrs() {
  static const llvm::StringSet<> S = {"kernel_name", "operandSegmentSizes"};
  return S;
}

/// Copy every user-set attribute (pingpong, tile_id, schedule_hint, ...) from
/// `oldOp` to `newOp`, skipping attributes that the builder already wrote.
static void migrateAttrs(mlir::Operation *oldOp, mlir::Operation *newOp,
                         const llvm::StringSet<> &builtIns) {
  for (mlir::NamedAttribute a : oldOp->getAttrs())
    if (!builtIns.contains(a.getName().strref()))
      newOp->setAttr(a.getName(), a.getValue());
}

/// Rebuild `old` in its async form.
///
/// Constructs a new DataBlockLoadOp with the given async deps and optional
/// token result, migrates user attributes, RAUWs the old memref result onto
/// the new one, and erases `old`. Returns { newAsyncToken (null if
/// !produceTok), newOp* }. Side-effects: `old` is erased.
static std::pair<mlir::Value, mlir::Operation *>
rebuildAsyncLoad(ADORA::DataBlockLoadOp old, mlir::ValueRange deps,
                 bool produceTok) {
  mlir::OpBuilder b(old);
  // NOTE: pipeline-generated BlockLoad ops may carry `map` but not `strides`
  // (strides was introduced with the PR2 experimental commit). Tolerate null
  // and fall back to an identity map / empty strides array so that rebuild
  // never dereferences a null attribute pointer.
  auto mapAttr =
      old->getAttrOfType<mlir::AffineMapAttr>(
          ADORA::DataBlockLoadOp::getMapAttrStr());
  mlir::AffineMap mapVal =
      mapAttr ? mapAttr.getValue()
              : mlir::AffineMap::getMultiDimIdentityMap(
                    old.getResult().getType().cast<mlir::MemRefType>().getRank(),
                    b.getContext());
  auto strides = old->getAttrOfType<mlir::DenseI64ArrayAttr>("strides");
  if (!strides) strides = mlir::DenseI64ArrayAttr::get(b.getContext(), {});
  std::string kern = old.getKernelName().str();
  mlir::SmallVector<mlir::Value> mapOps(old.getIndices().begin(),
                                        old.getIndices().end());

  auto newOp = b.create<ADORA::DataBlockLoadOp>(
      old.getLoc(), old.getOriginalMemref(),
      mapVal, mapOps,
      old.getResult().getType().cast<mlir::MemRefType>(),
      strides, kern, deps, produceTok);

  migrateAttrs(old, newOp, loadBuiltInAttrs());
  old.getResult().replaceAllUsesWith(newOp.getResult());
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : mlir::Value(),
          newOp.getOperation()};
}

/// Rebuild `old` in its async form. Store has no memref result, so there is
/// nothing to RAUW; only the async token is potentially produced.
static std::pair<mlir::Value, mlir::Operation *>
rebuildAsyncStore(ADORA::DataBlockStoreOp old, mlir::ValueRange deps,
                  bool produceTok) {
  mlir::OpBuilder b(old);
  // WHY: same null-tolerance as rebuildAsyncLoad — strides may be absent on
  // pipeline-generated ops that predate the PR2 experimental commit.
  auto mapAttr =
      old->getAttrOfType<mlir::AffineMapAttr>(
          ADORA::DataBlockStoreOp::getMapAttrStr());
  mlir::AffineMap mapVal =
      mapAttr ? mapAttr.getValue()
              : mlir::AffineMap::getMultiDimIdentityMap(
                    old.getSourceMemref().getType().cast<mlir::MemRefType>().getRank(),
                    b.getContext());
  auto strides = old->getAttrOfType<mlir::DenseI64ArrayAttr>("strides");
  if (!strides) strides = mlir::DenseI64ArrayAttr::get(b.getContext(), {});
  std::string kern = old.getKernelName().str();
  mlir::SmallVector<mlir::Value> mapOps(old.getIndices().begin(),
                                        old.getIndices().end());

  auto newOp = b.create<ADORA::DataBlockStoreOp>(
      old.getLoc(), old.getSourceMemref(), old.getTargetMemref(),
      mapVal, mapOps, strides, kern, deps, produceTok);

  migrateAttrs(old, newOp, storeBuiltInAttrs());
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : mlir::Value(),
          newOp.getOperation()};
}

/// Rebuild `old` KernelOp in its async form. Uses the takeBody-enabled
/// builder so the kernel body region transfers verbatim.
static std::pair<mlir::Value, mlir::Operation *>
rebuildAsyncKernel(ADORA::KernelOp old, mlir::ValueRange deps,
                   bool produceTok) {
  mlir::OpBuilder b(old);
  // NOTE: pipeline-generated KernelOps may lack a `kernel_name` attribute
  // (the no-arg builder doesn't set one). Guard against null before
  // forwarding to the rebuild builder.
  std::string name;
  if (auto kn = old->getAttrOfType<mlir::StringAttr>(
          ADORA::KernelOp::getKernelNameAttrStr()))
    name = kn.getValue().str();

  auto newOp = b.create<ADORA::KernelOp>(
      old.getLoc(), name, deps, produceTok, &old.getBody());

  migrateAttrs(old, newOp, kernelBuiltInAttrs());
  // KernelOp currently carries no data results; if it ever does, map them
  // positionally here, keeping the token result last.
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : mlir::Value(),
          newOp.getOperation()};
}

/// Thread !ADORA.token SSA values along the dep edges recorded on `graph`.
///
/// Algorithm:
///   1. Build preds[dstOp] = [srcOp...] and the fan-out set `hasOut` from
///      graph->depEdges(); self-edges are dropped.
///   2. Sort participating ops by lexical order (PR1 guarantees src->dst
///      satisfies isBeforeInBlock, so the edges form a DAG).
///   3. For each op in order, rebuild it as async: wire the deduplicated
/// @brief Dump the async-token dep chain as a Graphviz DOT file.
///
/// Must be called AFTER threadTokensOnDMAs — reads the SSA !ADORA.token
/// def-use chain directly from the IR rather than from the TaskGraph.
/// Each node is a BlockLoad/BlockStore/Kernel op; each edge is one token arc.
/// Node labels include the op type and the `Id` attribute when present.
///
/// @param func     the function to inspect
/// @param path     output file path (caller ensures non-empty)
static void dumpTokenGraphAsDot(func::FuncOp func, llvm::StringRef path) {
  std::error_code ec;
  llvm::raw_fd_ostream ofs(path, ec);
  if (ec) {
    llvm::errs() << "[dump-token-graph] cannot open " << path
                 << ": " << ec.message() << "\n";
    return;
  }

  // Helper: readable label for an op.
  auto label = [](mlir::Operation *op) -> std::string {
    std::string s = op->getName().getStringRef().str();
    if (auto id = op->getAttrOfType<mlir::StringAttr>("Id"))
      s += "\\nId=" + id.getValue().str();
    if (auto kn = op->getAttrOfType<mlir::StringAttr>("KernelName"))
      s += "\\n" + kn.getValue().str();
    // stream id if assigned
    if (auto st = op->getAttrOfType<mlir::IntegerAttr>("stream"))
      s += "\\nstream=" + std::to_string(st.getInt());
    return s;
  };

  // Collect all token-producing ops and their users.
  llvm::DenseSet<mlir::Operation *> seen;
  llvm::SmallVector<std::pair<mlir::Operation *, mlir::Operation *>> edges;

  func.walk([&](mlir::Operation *op) {
    mlir::Value tok = ADORA::getAsyncTokenOrNull(op);
    if (!tok) return;
    seen.insert(op);
    for (mlir::Operation *user : tok.getUsers()) {
      seen.insert(user);
      edges.push_back({op, user});
    }
  });

  ofs << "digraph token_graph {\n";
  ofs << "  rankdir=LR;\n";
  ofs << "  node [shape=box, fontsize=10];\n";

  // Nodes
  for (mlir::Operation *op : seen) {
    ofs << "  \"" << (void *)op << "\" [label=\"" << label(op) << "\"";
    if (isa<ADORA::KernelOp>(op))
      ofs << ", style=filled, fillcolor=lightyellow";
    else if (isa<ADORA::DataBlockLoadOp>(op))
      ofs << ", style=filled, fillcolor=lightblue";
    else if (isa<ADORA::DataBlockStoreOp>(op))
      ofs << ", style=filled, fillcolor=lightcoral";
    ofs << "];\n";
  }

  // Edges
  for (auto &[src, dst] : edges)
    ofs << "  \"" << (void *)src << "\" -> \"" << (void *)dst
        << "\" [label=\"token\", color=darkgreen];\n";

  ofs << "}\n";
  llvm::errs() << "[dump-token-graph] written to " << path << "\n";
}

///      tokens of its predecessors, produce a new token iff the op has any
///      outgoing edge.
///   4. Patch TaskNode back-pointers via oldToNew so graph metadata stays
///      valid after the old ops are erased.
static void threadTokensOnDMAs(
    TaskGraph *graph,
    const llvm::DenseSet<mlir::Operation *> &extraProducers = {}) {
  using mlir::Operation;
  using mlir::Value;

  llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 2>> preds;
  llvm::DenseSet<Operation *> hasOut;
  for (const auto &e : graph->depEdges()) {
    Operation *s = e.src ? e.src->getOperation() : nullptr;
    Operation *d = e.dst ? e.dst->getOperation() : nullptr;
    // NOTE: RemoveRedundant* may erase ops while leaving stale dep edges.
    // An erased op's getBlock() returns null; skip such dangling references.
    if (!s || !s->getBlock()) continue;
    if (!d || !d->getBlock()) continue;
    if (s == d) continue;
    preds[d].push_back(s);
    hasOut.insert(s);
  }
  // PR6.2: any op that the LC analyzer flagged as a loop-carried producer
  // also needs its token result materialised, even if it has no intra-block
  // successor.  Adding to hasOut + all triggers rebuild-with-token below.
  for (auto *p : extraProducers) {
    if (!p || !p->getBlock()) continue;
    hasOut.insert(p);
  }
  if (preds.empty() && hasOut.empty()) return;

  llvm::SetVector<Operation *> all;
  for (auto &kv : preds) {
    all.insert(kv.first);    for (auto *s : kv.second) all.insert(s);
  }
  for (auto *s : hasOut) all.insert(s);

  llvm::SmallVector<Operation *> ordered(all.begin(), all.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock()) return a->isBeforeInBlock(b);
    return a < b;  // different blocks: stable but arbitrary (not hit in practice)
  });

  llvm::DenseMap<Operation *, Value>       tokens;    // old op -> new token
  llvm::DenseMap<Operation *, Operation *> oldToNew; // for TaskNode patch-up

  for (Operation *op : ordered) {
    // 3a. Gather dedup'd tokens from already-rebuilt predecessors.
    llvm::SmallSetVector<Value, 4> depSet;
    auto itP = preds.find(op);
    if (itP != preds.end()) {
      for (Operation *p : itP->second) {
        if (p == op) continue;
        auto tIt = tokens.find(p);
        if (tIt != tokens.end() && tIt->second) depSet.insert(tIt->second);
      }
    }
    llvm::SmallVector<Value> deps(depSet.begin(), depSet.end());
    bool produce = hasOut.contains(op);

    std::pair<Value, Operation *> rebuilt;
    if (auto l = mlir::dyn_cast<ADORA::DataBlockLoadOp>(op))
      rebuilt = rebuildAsyncLoad(l, deps, produce);
    else if (auto s = mlir::dyn_cast<ADORA::DataBlockStoreOp>(op))
      rebuilt = rebuildAsyncStore(s, deps, produce);
    else if (auto k = mlir::dyn_cast<ADORA::KernelOp>(op))
      rebuilt = rebuildAsyncKernel(k, deps, produce);
    else
      continue;  // non-async-capable nodes (e.g. LocalMemAlloc) are skipped.

    if (rebuilt.first)  tokens[op]   = rebuilt.first;
    if (rebuilt.second) oldToNew[op] = rebuilt.second;
  }

  // 4. Patch TaskNode::_operation so graph remains usable post-rebuild.
  for (TaskNode *n : graph->getAllNodes()) {
    auto it = oldToNew.find(n->getOperation());
    if (it != oldToNew.end()) n->setOperation(it->second);
  }
}

/// Cross-check that SSA token edges match dep_summary edges. Used in CI with
/// the `cross-check-summary-vs-token` option.
///
/// Returns failure() on mismatch; caller decides whether to emitWarning or
/// signalPassFailure.
static mlir::LogicalResult
verifyTokensMatchSummary(TaskGraph *graph) {
  // Expected edges from the graph (ground truth).
  llvm::DenseSet<std::pair<mlir::Operation *, mlir::Operation *>> expected;
  for (const auto &e : graph->depEdges()) {
    auto *s = e.src ? e.src->getOperation() : nullptr;
    auto *d = e.dst ? e.dst->getOperation() : nullptr;
    if (!s || !d || s == d) continue;
    expected.insert({s, d});
  }

  // Actual edges derived from SSA token def-use after rebuild.
  llvm::DenseSet<std::pair<mlir::Operation *, mlir::Operation *>> actual;
  for (TaskNode *n : graph->getAllNodes()) {
    mlir::Operation *d = n->getOperation();
    if (!d || !ADORA::isAsyncCapable(d)) continue;
    for (mlir::Value tok : ADORA::getAsyncDeps(d)) {
      mlir::Operation *s = tok.getDefiningOp();
      if (s) actual.insert({s, d});
    }
  }

  return (expected == actual) ? mlir::success() : mlir::failure();
}

////////////////////////////////////////////////////
//// rewrite task graph through dependency analysis
////////////////////////////////////////////////////
/// @brief Removes redundant pairs of BlockStoreNode and BlockLoadNode in the task graph.
/// 
/// This function iterates through the nodes in the task graph and identifies pairs
/// of BlockLoadNode and BlockStoreNode that access the same memory block.
/// If such pairs are found, it connects the store node's kernel to the load node's
/// kernel, replaces the load node with the source node of the store, and schedules
/// the load node for deletion to optimize memory access and reduce redundancy.
/// When a BlockStore writes to a global memref and a later BlockLoad reads
/// the same block, the data can stay on-chip.  Wire the producing kernel
/// directly to the consuming kernel so threadTokensOnDMAs emits a token
/// on the BlockStore and threads it into the BlockLoad's asyncDependencies.
/// The BlockLoad MLIR op is intentionally left in place; threadTokensOnDMAs
/// will rebuild it as an async op (with the token dep) and erase the original.
void RemoveRedundantBlockStoreLoadPair(TaskGraph* graph){
  std::vector<TaskNode*> nodes = graph->getAllNodes();

  // Collect redundant loads to erase in a second pass (avoids iterator
  // invalidation and ensures replaceAllUsesWith happens before erase).
  llvm::SmallVector<std::pair<BlockLoadNode*, mlir::Value>> toReplace;

  // Guard against the same loadnode being queued more than once.
  // This can happen when multiple storenodes write to the same data block
  // (e.g. a store chain A→buf, B→buf, load←buf): generateTaskGraphFromBlock
  // adds RAW edges from *all* prior dirty stores to the load, so the inner
  // loop below would push the loadnode once per matching store.  Erasing the
  // same Op twice causes a null-TypeStorage crash in the second pass.
  llvm::SmallPtrSet<BlockLoadNode*, 8> processed;

  for (TaskNode* node : nodes) {
    if (!isa<BlockLoadNode>(node)) continue;
    BlockLoadNode* loadnode = dyn_cast<BlockLoadNode>(node);
    if (processed.contains(loadnode)) continue;

    ADORA::DataBlockLoadOp load = loadnode->getDataBlockLoadOp();

    // Among all incoming store-nodes that access the same data block, pick
    // the one that is LATEST in program order (i.e. the most recent write).
    // Using an earlier store's source buffer would be semantically incorrect.
    BlockStoreNode* bestStoreNode = nullptr;
    for (auto innode : node->getInNodes()) {
      if (!isa<BlockStoreNode>(innode)) continue;
      BlockStoreNode* storenode = dyn_cast<BlockStoreNode>(innode);
      ADORA::DataBlockStoreOp store = storenode->getDataBlockStoreOp();
      if (!AccessSameDataBlock(store, load)) continue;

      if (!bestStoreNode) {
        bestStoreNode = storenode;
      } else {
        // Keep the store that appears LATER in the block (closer to the load).
        mlir::Operation* bestOp  = bestStoreNode->getDataBlockStoreOp().getOperation();
        mlir::Operation* candOp  = storenode->getDataBlockStoreOp().getOperation();
        if (bestOp->getBlock() == candOp->getBlock() &&
            bestOp->isBeforeInBlock(candOp)) {
          bestStoreNode = storenode;
        }
      }
    }

    if (!bestStoreNode) continue;
    processed.insert(loadnode);

    ADORA::DataBlockStoreOp bestStore = bestStoreNode->getDataBlockStoreOp();

    // Wire producing kernel → consuming kernel so the scheduler sees
    // the inter-kernel dependency even after the load is removed.
    KernelNode* sourcekernel = bestStoreNode->getKernelNode();
    for (auto sinkkernel : loadnode->getKernelNodes()) {
      addConnectionBetweenTwoNode(sourcekernel, sinkkernel, depType::Depend);
      // Also add a depEdge so threadTokensOnDMAs sees store → sinkkernel
      // and threads a token. Without this the erased BlockLoad leaves a
      // gap: preds/hasOut in threadTokensOnDMAs skip dead ops (block==null),
      // so kernel_3mm_2 would get async{} with no deps and emit no gather.
      graph->addDepEdge({bestStoreNode, sinkkernel, DataBlockDepKind::RAW, true});
    }

    // The data already lives in the on-chip buffer (bestStore.SourceMemref).
    // Schedule this BlockLoad for removal: replace its result with the
    // on-chip buffer BEFORE erasing, so downstream users stay valid.
    toReplace.push_back({loadnode, bestStore.getSourceMemref()});
  }

  // Second pass: replaceAllUsesWith then erase (order matters!).
  for (auto &[loadnode, onChipBuf] : toReplace) {
    ADORA::DataBlockLoadOp load = loadnode->getDataBlockLoadOp();
    // Replace all downstream uses of the load result with the on-chip buffer.
    // MUST happen before erase — otherwise downstream ops hold dangling SSA refs.
    load.getResult().replaceAllUsesWith(onChipBuf);
    // Erase the redundant BlockLoad from the MLIR IR.
    // (TaskGraph is per-block and short-lived; no need to remove from _nodes.)
    load.erase();
  }
}

void RemoveRedundantBlockLoads(TaskGraph* graph){
  // TODO: implement load-after-load elimination (WAR/RAR same block).
  (void)graph;
}

// ============================================================================
// PR6.1 — loop-carried dep analysis glue.
//
// The real work lives in lib/Dialect/ADORA/Analysis/LoopCarriedDep.cpp; this
// file only collects the enclosing scf.for / affine.for ops containing a
// KernelOp body, runs the analyzer, and serialises the result into
// `adora.lc_dep_summary` on the FuncOp.
//
// The earlier prototype (`findLoopCarriedStoreLoadPair` + `wireLoopCarriedToken`)
// was removed — precision and scope upgraded, and loop-carried token threading
// is deferred to PR6.3 where it will consume the structured result here.
// ============================================================================

/// Collect every scf.for / affine.for in `func` whose body directly contains
/// an ADORA::KernelOp. Nested loops are each returned individually (caller
/// can decide per-level policy later).
static SmallVector<Operation*>
collectEnclosingLoopsWithKernel(func::FuncOp func) {
  SmallVector<Operation*> out;
  func.walk([&](Operation *op) {
    if (!isa<scf::ForOp, affine::AffineForOp>(op)) return;
    Region &region = op->getRegion(0);
    if (region.empty()) return;
    Block &body = region.front();
    for (auto k : body.getOps<ADORA::KernelOp>()) {
      (void)k;
      out.push_back(op);
      break;
    }
  });
  return out;
}




/// @brief A wrapper
/// @param func 
void ScheduleADORATasksPass::ScheduleADORATasksInFunction(func::FuncOp func){
  // PR6.2 (v4): keep affine.for. Loop-carried tokens are threaded via
  // affine.for iter_args / affine.yield in the LC pass below — no need to
  // promote to scf.for.  (Empirically verified affine.for supports custom
  // token iter_args; see docs/affine_for_yield_token_verification.md.)
  // (void)affineForOuterToSCF(func, 1);

  //////////////
  /// 1st step: get all block that needs to be scanned
  //////////////
  SmallVector<mlir::Block*> blocks;
  for(auto _it = func.getBody().begin(); _it != func.getBody().end(); _it++){
    mlir::Block* block = &*(_it); 
    if (BlockContainsKernelOp(block)) {
      blocks.push_back(block);
    }
  }
  func.walk([&](AffineForOp forop){
    mlir::Block* _b =  forop.getBody();
    if(BlockContainsKernelOp(_b)){
      blocks.push_back(_b);
    }
  });


  //////////////
  /// 2nd step: build task graph, analyze deps, simplify, thread tokens
  //////////////
  int idx = 0;
  SmallVector<mlir::Attribute> allEdgeAttrs; // P4.0 — accumulated across blocks

  // PR6.2: pre-scan loop-carried producers so threadTokensOnDMAs can force
  // a token result on them even if they have no intra-block successor
  // (typical case: a BlockStore at iter k feeding a BlockLoad at iter k+1).
  llvm::DenseSet<mlir::Operation *> lcProducers;
  if (emitTokens && threadLCTokens) {
    for (Operation *loopOp :
         collectEnclosingLoopsWithKernel(func)) {
      auto r = mlir::ADORA::analysis::analyzeLoopCarriedDeps(loopOp);
      for (const auto &c :
           mlir::ADORA::analysis::groupEdgesIntoChains(r, /*includeRAR=*/false))
        if (c.producer) lcProducers.insert(c.producer);
    }
  }

  for(auto block : blocks){
    TaskGraph* graph = new TaskGraph;

    // Step A: build graph nodes from block ops
    generateTaskGraphFromBlock(graph, block);

    // Step B: O(N²) dep analysis — RAW/WAR/WAW between BlockLoad/BlockStore
    analyzeDependencyInGraph(graph);
    appendDepEdgesToAttrList(graph, idx, func.getContext(), allEdgeAttrs);

    // Step C: simplify redundant transfers (must run BEFORE token threading:
    // RemoveRedundant* may erase ops; live asyncToken results would crash MLIR)
    RemoveRedundantBlockStoreLoadPair(graph);
    RemoveRedundantBlockLoads(graph);

    // Step D: thread SSA !ADORA.token along dep edges
    if (emitTokens)
      threadTokensOnDMAs(graph, lcProducers);

    // Step D2 (PR6.1) — loop-carried dep analysis is now done at the FuncOp
    // level after the block-pass loop completes (see lc_dep_summary below),
    // because LC edges live per-enclosing-loop, not per-block.

    // Step E: cross-check tokens vs dep_summary (CI only)
    if (emitTokens && crossCheck) {
      if (failed(verifyTokensMatchSummary(graph))) {
        func.emitError("adora async-token edges disagree with dep_summary "
                       "(cross-check-summary-vs-token)");
        signalPassFailure();
      }
    }

    idx++;
  }

  // P4.0 — attach `adora.dep_summary` to the host function for mapper-side
  // consumption via DepSummaryView. Empty list still attached (zero edges)
  // so downstream consumers can unambiguously detect that the pass ran.
  // PR2 commit B — gated by `emit-summary` (default true) so that once the
  // ecosystem fully migrates to SSA tokens we can retire this attribute
  // without pass-API changes.
  if (emitSummary)
    func->setAttr("adora.dep_summary",
                  ArrayAttr::get(func.getContext(), allEdgeAttrs));

  // PR6.1 — run loop-carried dep analysis at the FuncOp level: every
  // enclosing scf.for / affine.for whose body contains a KernelOp gets
  // analysed; non-empty results are serialised into `adora.lc_dep_summary`.
  //
  // Output is consumed by:
  //   - PR6.2: decides which inner affine.for must be promoted to scf.for
  //   - PR6.3: threadLoopCarriedTokens (real iter_args insertion)
  //   - tests / diagnostics
  if (emitSummary || threadLCTokens) {
    SmallVector<mlir::Attribute> lcAttrs;
    int loopIdx = 0;
    for (Operation *loopOp : collectEnclosingLoopsWithKernel(func)) {
      auto r = mlir::ADORA::analysis::analyzeLoopCarriedDeps(loopOp);
      if (r.empty()) { loopIdx++; continue; }
      if (emitSummary) {
        lcAttrs.push_back(
            mlir::ADORA::analysis::serializeLoopCarriedDeps(
                r, loopIdx, func.getContext()));
      }
      // PR6.2: thread loop-carried tokens through affine.for iter_args.
      if (threadLCTokens && emitTokens) {
        if (auto fo = dyn_cast<affine::AffineForOp>(loopOp)) {
          if (failed(mlir::ADORA::threadLoopCarriedTokensOnAffineFor(fo, r))) {
            signalPassFailure();
            return;
          }
        }
      }
      loopIdx++;
    }
    if (emitSummary && !lcAttrs.empty())
      func->setAttr("adora.lc_dep_summary",
                    ArrayAttr::get(func.getContext(), lcAttrs));
  }

  // PR1 — mark the enclosing module as post-schedule so downstream passes and
  // the mapper can assert scheduling has run. `adora.dep_summary` remains the
  // authoritative data channel in PR1; PR2 will make async tokens on
  // BlockLoad/BlockStore carry the ordering and retire this attribute.
  if (auto module = func->getParentOfType<ModuleOp>())
    module->setAttr("adora.scheduled", UnitAttr::get(func.getContext()));

  // Optional: dump token dep graph as DOT (--dump-token-graph=<path>).
  if (emitTokens && !tokenGraphPath.empty())
    dumpTokenGraphAsDot(func, tokenGraphPath);
}

void ScheduleADORATasksPass::runOnOperation()
{
  ScheduleADORATasksInFunction(getOperation());

  return;
}

} // namespace


std::unique_ptr<OperationPass<func::FuncOp>> 
  mlir::ADORA::createScheduleADORATasksPass()
{
  return std::make_unique<ScheduleADORATasksPass>();
}
//===- AssignStreams.cpp - Assign stream IDs to async-capable ops -----===//
//
// PR4 commit A.
//
// Topological-sort + greedy stream assignment for the !ADORA.token dep graph:
//   - Root ops (no asyncDependencies) → fresh stream (round-robin up to
//     max-streams).
//   - Dependent ops → inherit min(predecessor stream IDs).
//
// Writes `stream : i32` as a generic attribute on each async-capable op.
// Consumed by adora-lower-async-tokens (PR4 commit B) which passes the stored
// value to adora.signal/wait instead of the current hardcoded stream=0.
//
// Must run before adora-lower-async-tokens (which erases asyncToken results
// and asyncDependencies operands, destroying the dep-graph structure).
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "./PassDetail.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-assign-streams"

namespace {

struct AssignStreamsPass : public AssignStreamsBase<AssignStreamsPass> {
  void runOnOperation() override;
};

} // namespace

void AssignStreamsPass::runOnOperation() {
  func::FuncOp func = getOperation();

  // Collect all async-capable ops that participate in the token dep graph
  // (either produce a token result or carry asyncDependencies operands).
  SmallVector<Operation *> asyncOps;
  func.walk([&](Operation *op) {
    if (isAsyncCapable(op) &&
        (getAsyncTokenOrNull(op) || !getAsyncDeps(op).empty()))
      asyncOps.push_back(op);
  });

  if (asyncOps.empty())
    return;

  // Build token → producer map.
  llvm::DenseMap<Value, Operation *> tokenToProducer;
  for (Operation *op : asyncOps)
    if (Value tok = getAsyncTokenOrNull(op))
      tokenToProducer[tok] = op;

  // Build forward edges (successor lists) and in-degree for Kahn's algorithm.
  llvm::DenseMap<Operation *, unsigned> inDegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> succs;
  for (Operation *op : asyncOps) {
    inDegree.try_emplace(op, 0u);
    succs.try_emplace(op);
  }
  for (Operation *op : asyncOps) {
    for (Value dep : getAsyncDeps(op)) {
      auto it = tokenToProducer.find(dep);
      if (it == tokenToProducer.end())
        continue;
      Operation *pred = it->second;
      succs[pred].push_back(op);
      inDegree[op]++;
    }
  }

  // Kahn's BFS topological sort.  Using a vector-as-queue (head index) so
  // that ops collected in program order stay in program order — this gives
  // deterministic stream numbers in tests.
  SmallVector<Operation *> queue;
  queue.reserve(asyncOps.size());
  for (Operation *op : asyncOps)
    if (inDegree[op] == 0)
      queue.push_back(op);

  size_t head = 0;
  while (head < queue.size()) {
    Operation *op = queue[head++];
    for (Operation *s : succs[op])
      if (--inDegree[s] == 0)
        queue.push_back(s);
  }
  // `queue` is now the topological order (it consumed all reachable ops).

  // Greedy stream assignment in topological order.
  llvm::DenseMap<Operation *, unsigned> streamOf;
  unsigned nextStream = 0;

  for (Operation *op : queue) {
    OperandRange deps = getAsyncDeps(op);
    unsigned assigned;

    if (deps.empty()) {
      // Root op: take next fresh stream, wrap at maxStreams.
      assigned = nextStream++ % maxStreams;
    } else {
      // Dependent op: inherit minimum stream ID among predecessors.
      assigned = maxStreams; // sentinel — start above any valid ID
      for (Value dep : deps) {
        auto it = tokenToProducer.find(dep);
        if (it == tokenToProducer.end())
          continue;
        auto si = streamOf.find(it->second);
        if (si != streamOf.end())
          assigned = std::min(assigned, si->second);
      }
      // Fallback: predecessor outside the collected graph (should not happen
      // for well-formed IR, but handle gracefully).
      if (assigned == maxStreams)
        assigned = nextStream++ % maxStreams;
    }

    streamOf[op] = assigned;
    LLVM_DEBUG(llvm::dbgs() << "[assign-streams] " << op->getName()
                            << " → stream " << assigned << "\n");
    op->setAttr("stream",
                IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                 static_cast<int64_t>(assigned)));
  }
}

std::unique_ptr<OperationPass<func::FuncOp>>
mlir::ADORA::createAssignStreamsPass() {
  return std::make_unique<AssignStreamsPass>();
}

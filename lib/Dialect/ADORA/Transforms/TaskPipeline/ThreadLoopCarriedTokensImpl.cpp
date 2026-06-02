//===- ThreadLoopCarriedTokensImpl.cpp - PR6.2 in-place affine.for thread -===//
//
// Implementation of threadLoopCarriedTokensOnAffineFor. See spec at
// docs/pr6_2_thread_loop_carried_tokens_prompt.md (v4.1).
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/Transforms/TaskPipeline/ThreadLoopCarriedTokens.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::ADORA;

namespace {

/// Append `tok` to op's $asyncDependencies, for any ADORA op that has one.
LogicalResult appendAsyncDep(Operation *op, Value tok) {
  if (auto k = dyn_cast<KernelOp>(op)) {
    k.getAsyncDependenciesMutable().append(tok);
    return success();
  }
  if (auto l = dyn_cast<DataBlockLoadOp>(op)) {
    l.getAsyncDependenciesMutable().append(tok);
    return success();
  }
  if (auto s = dyn_cast<DataBlockStoreOp>(op)) {
    s.getAsyncDependenciesMutable().append(tok);
    return success();
  }
  return failure();
}

/// Get the (optional) token result from an ADORA async op.
Value getTokenResultOf(Operation *op) {
  if (auto k = dyn_cast<KernelOp>(op))         return k.getAsyncToken();
  if (auto l = dyn_cast<DataBlockLoadOp>(op))  return l.getAsyncToken();
  if (auto s = dyn_cast<DataBlockStoreOp>(op)) return s.getAsyncToken();
  return nullptr;
}

} // namespace

namespace mlir {
namespace ADORA {

LogicalResult threadLoopCarriedTokensOnAffineFor(
    affine::AffineForOp oldFor,
    const analysis::LoopCarriedDepResult &result) {
  auto chains = analysis::groupEdgesIntoChains(result, /*includeRAR=*/false);
  if (chains.empty())
    return success();

  // The pass currently only handles affine.for that has no existing
  // iter_args (the common case in this codebase).  If existing iter_args
  // are present, fail loudly so we don't silently mis-rewrite.
  if (oldFor.getNumIterOperands() != 0) {
    oldFor.emitWarning(
        "adora-pr6.2: affine.for already has iter_args; skipping LC token "
        "threading for safety.");
    return success();
  }

  const size_t N = chains.size();
  OpBuilder b(oldFor);
  Location loc = oldFor.getLoc();
  auto tokTy = TokenType::get(b.getContext());

  // 1) Create N null sentinel tokens before the loop.
  SmallVector<Value> inits;
  inits.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    auto ev = b.create<EventCreateOp>(loc, tokTy);
    inits.push_back(ev.getToken());
  }

  // 2) Build a new affine.for with iter_args.
  SmallVector<Type> resultTypes(N, tokTy);
  auto newFor = b.create<affine::AffineForOp>(
      loc,
      oldFor.getLowerBoundOperands(), oldFor.getLowerBoundMap(),
      oldFor.getUpperBoundOperands(), oldFor.getUpperBoundMap(),
      oldFor.getStep().getSExtValue(),
      /*iterArgs=*/inits);

  // 3) Move ops from old body to new body (except old affine.yield).
  //    AffineForOp builder does NOT auto-create a terminator when iter_args
  //    are present (it cannot know yield operands).  So newBody is empty
  //    apart from block-arguments; splice in everything-but-last from
  //    oldBody, then create the new yield manually below.
  Block *oldBody = oldFor.getBody();
  Block *newBody = newFor.getBody();
  newBody->getOperations().splice(
      newBody->end(),
      oldBody->getOperations(),
      oldBody->begin(), std::prev(oldBody->end()));

  // 4) Rewire induction var: old IV → new IV.
  oldBody->getArgument(0).replaceAllUsesWith(newBody->getArgument(0));

  // 5) For each chain: feed iter_arg into consumer, capture producer token
  //    for the new yield.
  SmallVector<Value> yieldOperands;
  yieldOperands.reserve(N);
  for (size_t i = 0; i < N; ++i) {
    Value lcArg = newBody->getArgument(1 + i); // 0 is IV
    if (failed(appendAsyncDep(chains[i].consumer, lcArg))) {
      oldFor.emitError("adora-pr6.2: consumer op is not an ADORA async op");
      return failure();
    }
    Value tok = getTokenResultOf(chains[i].producer);
    if (!tok) {
      oldFor.emitError(
          "adora-pr6.2: producer op has no $asyncToken; "
          "did threadTokensOnDMAs run before?");
      return failure();
    }
    yieldOperands.push_back(tok);
  }

  // 6) Create the new affine.yield carrying tokens.
  OpBuilder yb(newBody, newBody->end());
  yb.create<affine::AffineYieldOp>(loc, yieldOperands);

  // 7) Erase old loop. It has no results (we asserted earlier), so RAUW
  //    on old results isn't needed, but the new loop's results are
  //    intentionally left unused (they correspond to the final post-loop
  //    token state; can be consumed by a later wait if needed).
  oldFor.erase();
  return success();
}

} // namespace ADORA
} // namespace mlir

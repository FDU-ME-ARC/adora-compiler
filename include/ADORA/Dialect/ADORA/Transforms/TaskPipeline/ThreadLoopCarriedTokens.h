//===- ThreadLoopCarriedTokens.h - PR6.2 loop-carried token threading ---*- C++ -*-===//
//
// Helper used by ScheduleADORATasksPass to materialise loop-carried
// !ADORA.token dependencies as iter_args / affine.yield on affine.for.
// See docs/pr6_2_thread_loop_carried_tokens_prompt.md.
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_TRANSFORMS_THREADLOOPCARRIEDTOKENS_H_
#define ADORA_DIALECT_ADORA_TRANSFORMS_THREADLOOPCARRIEDTOKENS_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Support/LogicalResult.h"

#include "ADORA/Dialect/ADORA/Analysis/LoopCarriedDep.h"

namespace mlir {
namespace ADORA {

/// Rewrite `forOp` in-place with iter_args(!ADORA.token x N) /
/// affine.yield, threading loop-carried tokens based on `result`.
/// Returns success() on no-op (no chains) or successful rewrite.
LogicalResult threadLoopCarriedTokensOnAffineFor(
    affine::AffineForOp forOp,
    const analysis::LoopCarriedDepResult &result);

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_TRANSFORMS_THREADLOOPCARRIEDTOKENS_H_

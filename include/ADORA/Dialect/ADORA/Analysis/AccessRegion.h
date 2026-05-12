//===- AccessRegion.h - DRAM region accessed by a DMA op --------*- C++ -*-===//
//
// Copyright 2023-2024 The ADORA Authors.
//
// AccessRegion abstracts the DRAM region accessed by a BlockLoad / BlockStore
// op. It is the core primitive used by:
//   - intra-iteration overlap test    (future migration of checkDependencyBetween*)
//   - inter-iteration overlap test    (loop-carried dep analysis, PR6.1)
//   - load-after-load elimination     (PR6.6)
//
// Pure data + pure functions: no MLIR mutation. Safe to unit-test.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_DIALECT_ADORA_ANALYSIS_ACCESSREGION_H_
#define ADORA_DIALECT_ADORA_ANALYSIS_ACCESSREGION_H_

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace ADORA {
namespace analysis {

/// A rectangular tile-shaped region accessed by a single BlockLoad / BlockStore.
///
/// region[dim] = [ startExprs[dim](operands),
///                 startExprs[dim](operands) + sizes[dim] )
///
/// startExprs share a single dim-list (the op's AffineMap dims) bound to
/// `operands`. `sizes[dim]` is the static tile size on that dimension.
struct AccessRegion {
  /// Op that produced this region (BlockLoad or BlockStore).
  Operation *op = nullptr;
  /// The DRAM memref being accessed (memref SSA value).
  Value memref;
  /// Per-dim start expressions in the affine-map dim space.
  SmallVector<AffineExpr, 4> startExprs;
  /// Tile size per dim (static, taken from the result/source memref shape).
  SmallVector<int64_t, 4> sizes;
  /// Source memref shape (used to clip overlap probes; informational).
  SmallVector<int64_t, 4> sourceShape;
  /// SSA values bound to the affine map's dims (in dim-order).
  SmallVector<Value, 4> operands;

  /// Build from an ADORA::DataBlockLoadOp or ADORA::DataBlockStoreOp.
  /// Returns failure for unsupported ops.
  static FailureOr<AccessRegion> fromOp(Operation *op);

  /// Returns whether `op` is a BlockLoad.
  static bool isLoad(Operation *op);
  /// Returns whether `op` is a BlockStore.
  static bool isStore(Operation *op);

  /// Produce a new region where every occurrence of `iv` in startExprs'
  /// operand list is shifted by `+delta` (i.e. modelling the *next* iteration).
  ///
  /// If `iv` is not among `operands`, returns *this unchanged.
  AccessRegion shiftedByIV(Value iv, int64_t delta) const;

  /// Conservative rectangle overlap test on the same memref.
  ///
  /// Per-dim rule:
  ///   - If both start exprs reduce to *the same* affine form modulo a
  ///     constant delta on a single shared symbol, compute closed-form
  ///     interval overlap. Disjoint → return false.
  ///   - Otherwise mark this dim inconclusive (conservatively overlap).
  ///
  /// Different memrefs always return false.
  bool overlapsWith(const AccessRegion &other) const;

  /// Stronger predicate: both regions are byte-identical (same memref, same
  /// start expr, same size on every dim).
  bool sameAs(const AccessRegion &other) const;
};

} // namespace analysis
} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_ANALYSIS_ACCESSREGION_H_

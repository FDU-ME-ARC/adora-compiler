//===- AccessRegion.cpp - DRAM region accessed by a DMA op ----------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/Analysis/AccessRegion.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::analysis;

// ---------------------------------------------------------------------------
// op-type helpers
// ---------------------------------------------------------------------------
bool AccessRegion::isLoad(Operation *op) {
  return isa_and_nonnull<ADORA::DataBlockLoadOp>(op);
}
bool AccessRegion::isStore(Operation *op) {
  return isa_and_nonnull<ADORA::DataBlockStoreOp>(op);
}

// ---------------------------------------------------------------------------
// fromOp — extract memref / startExprs / sizes from a BlockLoad or BlockStore
// ---------------------------------------------------------------------------
FailureOr<AccessRegion> AccessRegion::fromOp(Operation *op) {
  AccessRegion r;
  r.op = op;
  AffineMap map;
  ArrayRef<int64_t> tileShape;
  ArrayRef<int64_t> srcShape;
  ValueRange operands;

  if (auto load = dyn_cast<ADORA::DataBlockLoadOp>(op)) {
    r.memref = load.getOriginalMemref();
    map = load.getAffineMap();
    operands = load.getMapOperands();
    tileShape = load.getResultType().getShape();
    srcShape = load.getOriginalMemrefType().getShape();
  } else if (auto store = dyn_cast<ADORA::DataBlockStoreOp>(op)) {
    r.memref = store.getTargetMemref();
    map = store.getAffineMap();
    operands = store.getMapOperands();
    tileShape = store.getSourceMemrefType().getShape();
    srcShape = store.getTargetMemrefType().getShape();
  } else {
    return failure();
  }

  // Ranks must match the number of map results for a regular BlockLoad/Store.
  // (Special case: empty map for memref<...> -> memref<...> 0-d source — handle
  // conservatively by leaving startExprs empty.)
  for (auto e : map.getResults())
    r.startExprs.push_back(e);
  for (auto s : tileShape)
    r.sizes.push_back(s);
  for (auto s : srcShape)
    r.sourceShape.push_back(s);
  for (auto v : operands)
    r.operands.push_back(v);
  return r;
}

// ---------------------------------------------------------------------------
// shiftedByIV — Substitute iv → iv + delta in affine expressions
// ---------------------------------------------------------------------------
AccessRegion AccessRegion::shiftedByIV(Value iv, int64_t delta) const {
  if (operands.empty()) return *this; // No operands → no IV refs
  
  // Find the position of `iv` in operands.
  int ivPos = -1;
  for (size_t i = 0; i < operands.size(); ++i) {
    if (operands[i] == iv) { ivPos = i; break; }
  }
  if (ivPos < 0) return *this; // `iv` not in operands
  
  AccessRegion shifted = *this;
  Builder b(iv.getContext());
  AffineExpr dimIV = b.getAffineDimExpr(ivPos);
  // Build a replacement vector for replaceDims(ArrayRef<AffineExpr>) form:
  // dims[i] -> AffineDimExpr(i), except dims[ivPos] -> dims[ivPos] + delta.
  SmallVector<AffineExpr, 4> dimReplacements;
  dimReplacements.reserve(operands.size());
  for (unsigned i = 0; i < operands.size(); ++i) {
    if ((int)i == ivPos)
      dimReplacements.push_back(b.getAffineDimExpr(i) + delta);
    else
      dimReplacements.push_back(b.getAffineDimExpr(i));
  }
  SmallVector<AffineExpr, 4> newExprs;
  for (auto e : startExprs) {
    AffineExpr shifted_e = e.replaceDims(dimReplacements);
    newExprs.push_back(shifted_e);
  }
  shifted.startExprs = newExprs;
  return shifted;
}

// ---------------------------------------------------------------------------
// Helpers for overlapsWith
// ---------------------------------------------------------------------------
struct Interval {
  int64_t lb = 0;
  int64_t ub = -1; // inclusive
};

static bool intervalsOverlap(const Interval &a, const Interval &b) {
  return (a.lb <= b.ub) && (b.lb <= a.ub);
}

static bool tryGetConstantBox(ArrayRef<AffineExpr> startExprs,
                               ArrayRef<int64_t> sizes,
                               SmallVector<Interval> &outBox) {
  outBox.clear();
  outBox.reserve(startExprs.size());
  
  for (size_t d = 0; d < startExprs.size(); ++d) {
    auto cst = startExprs[d].dyn_cast<AffineConstantExpr>();
    if (!cst) return false;
    if (d >= sizes.size()) return false;
    if (sizes[d] == ShapedType::kDynamic) return false;
    
    int64_t lb = cst.getValue();
    int64_t ub = lb + sizes[d] - 1;
    outBox.push_back(Interval{lb, ub});
  }
  return true;
}

// ---------------------------------------------------------------------------
// overlapsWith — Conservative box overlap on same memref
// ---------------------------------------------------------------------------
bool AccessRegion::overlapsWith(const AccessRegion &other) const {
  if (memref != other.memref) return false;
  if (startExprs.size() != other.startExprs.size()) return true; // rank mismatch → assume overlap
  
  SmallVector<Interval> boxA, boxB;
  bool okA = tryGetConstantBox(startExprs, sizes, boxA);
  bool okB = tryGetConstantBox(other.startExprs, other.sizes, boxB);
  
  // If either side can't fold to constant boxes, assume overlap (conservative).
  if (!okA || !okB) return true;
  
  // Both are constant boxes — check per-dimension overlap.
  for (size_t d = 0; d < boxA.size(); ++d) {
    if (!intervalsOverlap(boxA[d], boxB[d]))
      return false; // Disjoint in this dimension → no overlap
  }
  return true; // Overlaps in all dimensions
}

bool AccessRegion::sameAs(const AccessRegion &other) const {
  if (memref != other.memref) return false;
  if (startExprs.size() != other.startExprs.size()) return false;
  if (sizes != other.sizes) return false;
  for (size_t i = 0; i < startExprs.size(); ++i)
    if (startExprs[i] != other.startExprs[i]) return false;
  return true;
}

//===- AccessRegion.cpp - DRAM region accessed by a DMA op ----------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/Analysis/AccessRegion.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/Analysis/FlatLinearValueConstraints.h"
#include "mlir/IR/AffineMap.h"
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
// overlapsWith — Exact overlap check via Presburger arithmetic
// ---------------------------------------------------------------------------
bool AccessRegion::overlapsWith(const AccessRegion &other) const {
  if (memref != other.memref) return false;
  if (startExprs.size() != other.startExprs.size()) return true;

  SmallVector<Interval> boxA, boxB;
  bool okA = tryGetConstantBox(startExprs, sizes, boxA);
  bool okB = tryGetConstantBox(other.startExprs, other.sizes, boxB);

  if (okA && okB) {
    // Both sides are constant boxes — exact per-dimension check.
    for (size_t d = 0; d < boxA.size(); ++d)
      if (!intervalsOverlap(boxA[d], boxB[d]))
        return false;
    return true;
  }

  // Symbolic case: build a Presburger constraint system to check whether any
  // integer element index can simultaneously lie in region A and region B.
  //
  // Both regions must share the same operands (guaranteed by shiftedByIV).
  if (operands != other.operands)
    return true; // mismatched operand sets → conservative

  unsigned N = operands.size(); // number of IV/symbol dim variables
  unsigned R = startExprs.size(); // memref rank
  if (R == 0) return true;

  // Variable layout: [op_0 .. op_{N-1},  i_0 .. i_{R-1}]
  // (N dims for operands, R dims for element indices, 0 symbols)
  FlatLinearConstraints cs(/*numDims=*/N + R, /*numSymbols=*/0);

  MLIRContext *ctx = startExprs[0].getContext();

  for (unsigned d = 0; d < R; ++d) {
    if (d >= sizes.size() || sizes[d] == ShapedType::kDynamic ||
        d >= other.sizes.size() || other.sizes[d] == ShapedType::kDynamic)
      return true; // unknown tile size → conservative

    int64_t sA = sizes[d], sB = other.sizes[d];
    AffineExpr aS = startExprs[d], bS = other.startExprs[d];

    // Region A:  aS <= i[d] <= aS + sA - 1
    // Region B:  bS <= i[d] <= bS + sB - 1
    // Use addBound(LB/UB, pos=N+d, AffineMap with N+R dims).
    if (failed(cs.addBound(presburger::BoundType::LB, N + d,
                           AffineMap::get(N + R, 0, {aS}, ctx), true)) ||
        failed(cs.addBound(presburger::BoundType::UB, N + d,
                           AffineMap::get(N + R, 0, {aS + (sA - 1)}, ctx), true)) ||
        failed(cs.addBound(presburger::BoundType::LB, N + d,
                           AffineMap::get(N + R, 0, {bS}, ctx), true)) ||
        failed(cs.addBound(presburger::BoundType::UB, N + d,
                           AffineMap::get(N + R, 0, {bS + (sB - 1)}, ctx), true)))
      return true; // addBound failed → conservative

  }

  // If the integer polyhedron is empty, no element can be in both regions.
  return !cs.isIntegerEmpty();
}

bool AccessRegion::sameAs(const AccessRegion &other) const {
  if (memref != other.memref) return false;
  if (startExprs.size() != other.startExprs.size()) return false;
  if (sizes != other.sizes) return false;
  for (size_t i = 0; i < startExprs.size(); ++i)
    if (startExprs[i] != other.startExprs[i]) return false;
  return true;
}

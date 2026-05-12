//===- LoopCarriedDep.cpp - Loop-carried DMA dep analysis -----------------===//
//
// Copyright 2023-2024 The ADORA Authors.
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/Analysis/LoopCarriedDep.h"
#include "ADORA/Dialect/ADORA/Analysis/AccessRegion.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::ADORA::analysis;

llvm::StringRef mlir::ADORA::analysis::getLCKindString(LCKind k) {
  switch (k) {
    case LCKind::RAW: return "LC-RAW";
    case LCKind::WAR: return "LC-WAR";
    case LCKind::WAW: return "LC-WAW";
    case LCKind::RAR: return "LC-RAR";
  }
  return "LC-???";
}

// ---------------------------------------------------------------------------
// Helpers: extract IV / step / body from loopOp.
// ---------------------------------------------------------------------------
static Block *getLoopBody(Operation *loopOp) {
  if (auto f = dyn_cast<scf::ForOp>(loopOp)) return f.getBody();
  if (auto f = dyn_cast<affine::AffineForOp>(loopOp)) return f.getBody();
  return nullptr;
}
static Value getLoopIV(Operation *loopOp) {
  if (auto f = dyn_cast<scf::ForOp>(loopOp))           return f.getInductionVar();
  if (auto f = dyn_cast<affine::AffineForOp>(loopOp))  return f.getInductionVar();
  return nullptr;
}
static int64_t getLoopStep(Operation *loopOp) {
  if (auto f = dyn_cast<affine::AffineForOp>(loopOp))
    return f.getStepAsInt();
  if (auto f = dyn_cast<scf::ForOp>(loopOp)) {
    // scf.for step is a Value — try matchConstant
    if (auto cst = f.getStep().getDefiningOp<arith::ConstantIndexOp>())
      return cst.value();
    return 1; // unknown — treat as 1
  }
  return 1;
}

// Decide LCKind given access nature of src/dst.
static LCKind classify(bool srcIsStore, bool dstIsStore) {
  if (srcIsStore && !dstIsStore)  return LCKind::RAW; // store(prev) -> load(curr)
  if (!srcIsStore && dstIsStore)  return LCKind::WAR; // load(prev)  -> store(curr)
  if (srcIsStore && dstIsStore)   return LCKind::WAW;
  return LCKind::RAR;
}

// ---------------------------------------------------------------------------
LoopCarriedDepResult mlir::ADORA::analysis::analyzeLoopCarriedDeps(
    Operation *loopOp) {
  LoopCarriedDepResult r;
  r.loopOp = loopOp;
  Block *body = getLoopBody(loopOp);
  if (!body) return r;
  Value iv = getLoopIV(loopOp);
  int64_t step = getLoopStep(loopOp);

  // Collect all DMA ops in body (depth-1; nested loops aren't this loop's job).
  SmallVector<Operation*> dmaOps;
  for (Operation &op : body->getOperations()) {
    if (AccessRegion::isLoad(&op) || AccessRegion::isStore(&op))
      dmaOps.push_back(&op);
  }
  if (dmaOps.size() < 1) return r;

  // For every (src in iter k, dst in iter k+1) pair on the *same memref*:
  //   - compute src.shiftedByIV(iv, 0)  vs  dst.shiftedByIV(iv, +step)
  //   - if regions overlap, emit a LC edge classified by Load/Store kinds.
  // RAR edges are emitted too (cheap; downstream may filter — useful for
  // load-after-load elimination in PR6.6).
  for (Operation *src : dmaOps) {
    auto srcR = AccessRegion::fromOp(src);
    if (failed(srcR)) continue;
    for (Operation *dst : dmaOps) {
      auto dstR = AccessRegion::fromOp(dst);
      if (failed(dstR)) continue;
      if (srcR->memref != dstR->memref) continue;

      AccessRegion nextDst = dstR->shiftedByIV(iv, step);
      if (!srcR->overlapsWith(nextDst)) continue;

      LoopCarriedDepEdge e;
      e.src = src;
      e.dst = dst;
      e.kind = classify(AccessRegion::isStore(src), AccessRegion::isStore(dst));
      e.enclosingLoop = loopOp;
      e.loopIV = iv;
      e.step = step;
      e.exact = srcR->sameAs(nextDst);
      r.edges.push_back(e);
    }
  }
  return r;
}

// ---------------------------------------------------------------------------
Attribute mlir::ADORA::analysis::serializeLoopCarriedDeps(
    const LoopCarriedDepResult &r, int loopIdx, MLIRContext *ctx) {
  Builder b(ctx);
  SmallVector<Attribute> edgeAttrs;
  for (const auto &e : r.edges) {
    NamedAttribute fields[] = {
      b.getNamedAttr("kind", b.getStringAttr(getLCKindString(e.kind))),
      b.getNamedAttr("step", b.getI64IntegerAttr(e.step)),
      b.getNamedAttr("exact", b.getBoolAttr(e.exact)),
    };
    edgeAttrs.push_back(b.getDictionaryAttr(fields));
  }
  NamedAttribute outer[] = {
    b.getNamedAttr("loop_idx", b.getI64IntegerAttr(loopIdx)),
    b.getNamedAttr("loop_op", b.getStringAttr(
        r.loopOp ? r.loopOp->getName().getStringRef() : "<null>")),
    b.getNamedAttr("edges", b.getArrayAttr(edgeAttrs)),
  };
  return b.getDictionaryAttr(outer);
}

// ---------------------------------------------------------------------------
// PR6.2: chain aggregation.
// Merge edges sharing the same (src, dst) pair into one chain. RAR edges
// are skipped unless the caller asks for them (they correspond to load-
// after-load elimination, not a real hazard).
// ---------------------------------------------------------------------------
SmallVector<LCChain> mlir::ADORA::analysis::groupEdgesIntoChains(
    const LoopCarriedDepResult &r, bool includeRAR) {
  SmallVector<LCChain> chains;
  // O(N^2) dedup against already-collected chains; chain count is tiny in
  // practice (single-digit), so this is fine and avoids hashing Operation*.
  for (const auto &e : r.edges) {
    if (!includeRAR && e.kind == LCKind::RAR) continue;
    bool dup = false;
    for (auto &c : chains) {
      if (c.producer == e.src && c.consumer == e.dst) {
        dup = true;
        break;
      }
    }
    if (dup) continue;
    chains.push_back({e.src, e.dst, e.kind});
  }
  return chains;
}

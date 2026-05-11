//===- LoopCarriedDep.h - Loop-carried DMA dep analysis ---------*- C++ -*-===//
//
// Copyright 2023-2024 The ADORA Authors.
//
// Detects dependencies that cross iteration boundaries of an enclosing loop
// (scf.for or affine.for) containing ADORA BlockLoad / BlockStore ops.
//
// Output is consumed by:
//   - PR6.2 — affine.for → scf.for conversion (decides which inner loops
//             need iter_args for token threading)
//   - PR6.3 — threadLoopCarriedTokens (real iter_args insertion)
//   - Diagnostics / inspection via the `adora.lc_dep_summary` attribute
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_DIALECT_ADORA_ANALYSIS_LOOPCARRIEDDEP_H_
#define ADORA_DIALECT_ADORA_ANALYSIS_LOOPCARRIEDDEP_H_

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace ADORA {
namespace analysis {

enum class LCKind { RAW, WAR, WAW, RAR };

llvm::StringRef getLCKindString(LCKind k);

/// A single loop-carried dependency edge.
struct LoopCarriedDepEdge {
  Operation *src = nullptr;      // op at iteration k    (predecessor)
  Operation *dst = nullptr;      // op at iteration k+1  (successor)
  LCKind     kind = LCKind::RAW;
  Operation *enclosingLoop = nullptr; // scf.for or affine.for
  Value      loopIV;
  int64_t    step = 1;
  bool       exact = false;      // overlap regions are byte-identical
};

struct LoopCarriedDepResult {
  Operation *loopOp = nullptr;
  SmallVector<LoopCarriedDepEdge> edges;

  bool empty() const { return edges.empty(); }
};

/// Analyze loop-carried DMA dependencies of an enclosing loop.
/// `loopOp` must be an scf::ForOp or affine::AffineForOp directly enclosing
/// a block containing BlockLoad / BlockStore ops.
LoopCarriedDepResult analyzeLoopCarriedDeps(Operation *loopOp);

/// Serialize a result to a DictionaryAttr suitable for embedding inside a
/// `adora.lc_dep_summary` ArrayAttr.
Attribute serializeLoopCarriedDeps(const LoopCarriedDepResult &r,
                                    int loopIdx, MLIRContext *ctx);

} // namespace analysis
} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_ANALYSIS_LOOPCARRIEDDEP_H_

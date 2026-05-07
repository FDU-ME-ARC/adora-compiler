//===----------------------------------------------------------------------===//
// DepSummaryView.h — read-only parser for the `adora.dep_summary` attribute.
//
// P4.0: ScheduleAdoraTasks attaches an ArrayAttr of DictionaryAttr rows to the
// enclosing FuncOp. Each row describes one DataBlock-level dependency edge:
//
//   { block_idx: i64, src: i64, dst: i64, kind: str, overlap: i1 }
//
// This header exposes a POD record + a single parse entry point so downstream
// consumers (the mapper bridge, visualizers, tests) can read the summary
// without linking any MLIR Transform/Dialect logic beyond core IR.
//
// Design intent: keep this file dependency-light — consumers outside the
// ADORA dialect (e.g. a future mapper bridge) only need BuiltinAttributes.
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_TRANSFORMS_TASKGRAPH_DEPSUMMARYVIEW_H_
#define ADORA_DIALECT_ADORA_TRANSFORMS_TASKGRAPH_DEPSUMMARYVIEW_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>
#include <string>

namespace mlir {
namespace ADORA {

/// Mirrors the ArrayAttr row produced by appendDepEdgesToAttrList.
/// Kept as a plain struct so mapper-side code (potentially built without
/// MLIR in the compile unit) can consume this after a one-shot extraction.
struct DepSummaryRecord {
  int64_t     blockIdx;     ///< index of the kernel-containing block
  int64_t     srcNodeId;    ///< graph-local id of the producer / earlier op
  int64_t     dstNodeId;    ///< graph-local id of the consumer / later op
  std::string kind;         ///< "RAW" | "WAR" | "WAW" | "RAR"
  bool        mustOverlap;  ///< true = exact-same-block; false = conservative
};

/// Parse an `adora.dep_summary` attribute off any mlir::Operation (typically
/// the host FuncOp). Returns an empty vector if the attribute is missing,
/// malformed, or not an ArrayAttr of DictionaryAttrs.
///
/// The parser is permissive: individual malformed rows are silently dropped
/// to keep downstream consumers robust against future schema additions.
SmallVector<DepSummaryRecord> parseDepSummary(Operation *op);

/// Convenience overload — parse from an explicit ArrayAttr (e.g. from tests).
SmallVector<DepSummaryRecord> parseDepSummary(ArrayAttr arr);

/// The attribute name used by ScheduleAdoraTasks.
inline StringRef getDepSummaryAttrName() { return "adora.dep_summary"; }

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_TRANSFORMS_TASKGRAPH_DEPSUMMARYVIEW_H_

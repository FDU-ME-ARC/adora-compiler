//===----------------------------------------------------------------------===//
// DepSummaryView.h — read-only parser for the `adora.dep_summary` attribute.
//
// P4.0: ScheduleAdoraTasks attaches an ArrayAttr of DictionaryAttr rows to the
// enclosing FuncOp. Each row describes one DataBlock-level dependency edge.
//
// Moved from include/ADORA/Dialect/ADORA/Transforms/TaskPipeline/TaskGraph/DepSummaryView.h
// to the Analysis layer (PR6.1). The old path is preserved as a shim so
// existing includes keep working.
//
// Design intent: keep this file dependency-light — consumers outside the
// ADORA dialect (e.g. a future mapper bridge) only need BuiltinAttributes.
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_ANALYSIS_DEPSUMMARYVIEW_H_
#define ADORA_DIALECT_ADORA_ANALYSIS_DEPSUMMARYVIEW_H_

#include "ADORA/Dialect/ADORA/Analysis/DepKind.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include <cstdint>

namespace mlir {
namespace ADORA {

/// Mirrors the ArrayAttr row produced by appendDepEdgesToAttrList.
struct DepSummaryRecord {
  int64_t          blockIdx;
  int64_t          srcNodeId;
  int64_t          dstNodeId;
  DataBlockDepKind kind;
  bool             mustOverlap;
};

SmallVector<DepSummaryRecord> parseDepSummary(Operation *op);
SmallVector<DepSummaryRecord> parseDepSummary(ArrayAttr arr);

inline StringRef getDepSummaryAttrName() { return "adora.dep_summary"; }

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_ANALYSIS_DEPSUMMARYVIEW_H_

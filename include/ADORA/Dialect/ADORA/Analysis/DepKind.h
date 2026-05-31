//===----------------------------------------------------------------------===//
// DepKind.h — shared enum + (de)serialization for DataBlock-level dep kind.
//
// Lives in its own header so both the producer side (TaskGraph / ScheduleAdora
// Tasks) and the consumer side (DepSummaryView, mapper bridges, tests) share
// exactly one type. String forms are stable identifiers and form part of the
// `adora.dep_summary` attribute schema — do not rename without bumping the
// schema version.
//
// Moved from include/ADORA/Dialect/ADORA/Transforms/TaskPipeline/TaskGraph/DepKind.h to
// the Analysis layer (PR6.1). The old path is preserved as a shim so existing
// includes keep working.
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_ADORA_ANALYSIS_DEPKIND_H_
#define ADORA_DIALECT_ADORA_ANALYSIS_DEPKIND_H_

#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringRef.h"
#include <optional>

namespace mlir {
namespace ADORA {

/// DataBlock-level dependency kind. Values are stable; do not renumber.
enum class DataBlockDepKind {
  RAW = 0,   ///< read-after-write  (store -> later load)
  WAR = 1,   ///< write-after-read  (load  -> later store)
  WAW = 2,   ///< write-after-write
  RAR = 3    ///< read-after-read  (conservative; load coalescing)
};

/// Total serialization — every enumerator has a string form. Kept inline
/// to avoid a dedicated TU.
inline llvm::StringRef toString(DataBlockDepKind k) {
  switch (k) {
    case DataBlockDepKind::RAW: return "RAW";
    case DataBlockDepKind::WAR: return "WAR";
    case DataBlockDepKind::WAW: return "WAW";
    case DataBlockDepKind::RAR: return "RAR";
  }
  return "UNK";
}

/// Inverse. Returns nullopt on unknown string so callers can log / warn.
inline std::optional<DataBlockDepKind> parseDepKind(llvm::StringRef s) {
  if (s == "RAW") return DataBlockDepKind::RAW;
  if (s == "WAR") return DataBlockDepKind::WAR;
  if (s == "WAW") return DataBlockDepKind::WAW;
  if (s == "RAR") return DataBlockDepKind::RAR;
  return std::nullopt;
}

} // namespace ADORA
} // namespace mlir

#endif // ADORA_DIALECT_ADORA_ANALYSIS_DEPKIND_H_

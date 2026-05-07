//===----------------------------------------------------------------------===//
// DepSummaryView.cpp — parser impl for P4.0's `adora.dep_summary`.
//
// Attribute schema (grouped form, produced by ScheduleAdoraTasks):
//   adora.dep_summary = [
//     { block_idx: i64, edges: [
//         { src: i64, dst: i64, kind: str, overlap: i1 },
//         ... ] },
//     ...
//   ]
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "adora-dep-summary-view"

namespace mlir {
namespace ADORA {

namespace {

/// Generic lookup + dyn_cast on a DictionaryAttr field.
/// Using the free mlir::dyn_cast form per LLVM 17+ guidance (old member-form
/// is deprecated). Returns nullptr on miss or wrong type.
template <typename AttrT>
static AttrT tryGet(DictionaryAttr dict, StringRef name) {
  if (!dict)
    return nullptr;
  auto a = dict.get(name);
  if (!a)
    return nullptr;
  return mlir::dyn_cast<AttrT>(a);
}

} // namespace

static bool decodeEdge(DictionaryAttr edgeDict, int64_t blockIdx,
                       DepSummaryRecord &out) {
  auto srcA = tryGet<IntegerAttr>(edgeDict, "src");
  auto dstA = tryGet<IntegerAttr>(edgeDict, "dst");
  auto kndA = tryGet<StringAttr >(edgeDict, "kind");
  auto ovlA = tryGet<BoolAttr   >(edgeDict, "overlap");
  if (!srcA || !dstA || !kndA || !ovlA)
    return false;
  auto parsed = parseDepKind(kndA.getValue());
  if (!parsed) {
    LLVM_DEBUG(llvm::dbgs()
               << "adora.dep_summary: unknown kind '" << kndA.getValue()
               << "' for edge in block " << blockIdx << " — dropped\n");
    return false;
  }
  out.blockIdx    = blockIdx;
  out.srcNodeId   = srcA.getInt();
  out.dstNodeId   = dstA.getInt();
  out.kind        = *parsed;
  out.mustOverlap = ovlA.getValue();
  return true;
}

SmallVector<DepSummaryRecord> parseDepSummary(ArrayAttr arr) {
  SmallVector<DepSummaryRecord> out;
  if (!arr)
    return out;
  for (Attribute groupElem : arr) {
    auto group = mlir::dyn_cast<DictionaryAttr>(groupElem);
    if (!group)
      continue;
    auto blkA = tryGet<IntegerAttr>(group, "block_idx");
    auto edgesA = tryGet<ArrayAttr>(group, "edges");
    if (!blkA || !edgesA)
      continue;
    int64_t blockIdx = blkA.getInt();
    out.reserve(out.size() + edgesA.size());
    for (Attribute edgeElem : edgesA) {
      auto edgeDict = mlir::dyn_cast<DictionaryAttr>(edgeElem);
      if (!edgeDict)
        continue;
      DepSummaryRecord rec{};
      if (decodeEdge(edgeDict, blockIdx, rec))
        out.push_back(std::move(rec));
    }
  }
  return out;
}

SmallVector<DepSummaryRecord> parseDepSummary(Operation *op) {
  if (!op)
    return {};
  auto attr = op->getAttr(getDepSummaryAttrName());
  if (!attr)
    return {};
  auto arr = mlir::dyn_cast<ArrayAttr>(attr);
  if (!arr)
    return {};
  return parseDepSummary(arr);
}

} // namespace ADORA
} // namespace mlir

//===----------------------------------------------------------------------===//
// DepSummaryView.cpp — parser impl for P4.0's `adora.dep_summary`.
//===----------------------------------------------------------------------===//
#include "ADORA/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.h"

namespace mlir {
namespace ADORA {

// Extract an integer from a NamedAttribute list of a DictionaryAttr.
// Returns true on success and writes to `out`. Accepts IntegerAttr (signed
// or unsigned) — DictionaryAttr::get("foo") returns the Attribute or nullptr.
static bool tryGetInt(DictionaryAttr dict, StringRef name, int64_t &out) {
  auto a = dict.get(name);
  if (!a) return false;
  auto intAttr = a.dyn_cast<IntegerAttr>();
  if (!intAttr) return false;
  out = intAttr.getInt();
  return true;
}

static bool tryGetBool(DictionaryAttr dict, StringRef name, bool &out) {
  auto a = dict.get(name);
  if (!a) return false;
  auto boolAttr = a.dyn_cast<BoolAttr>();
  if (!boolAttr) return false;
  out = boolAttr.getValue();
  return true;
}

static bool tryGetStr(DictionaryAttr dict, StringRef name, std::string &out) {
  auto a = dict.get(name);
  if (!a) return false;
  auto strAttr = a.dyn_cast<StringAttr>();
  if (!strAttr) return false;
  out = strAttr.getValue().str();
  return true;
}

SmallVector<DepSummaryRecord> parseDepSummary(ArrayAttr arr) {
  SmallVector<DepSummaryRecord> out;
  if (!arr) return out;
  out.reserve(arr.size());
  for (Attribute elem : arr) {
    auto dict = elem.dyn_cast<DictionaryAttr>();
    if (!dict) continue;
    DepSummaryRecord rec{};
    if (!tryGetInt (dict, "block_idx", rec.blockIdx))    continue;
    if (!tryGetInt (dict, "src",       rec.srcNodeId))   continue;
    if (!tryGetInt (dict, "dst",       rec.dstNodeId))   continue;
    if (!tryGetStr (dict, "kind",      rec.kind))        continue;
    if (!tryGetBool(dict, "overlap",   rec.mustOverlap)) continue;
    out.push_back(std::move(rec));
  }
  return out;
}

SmallVector<DepSummaryRecord> parseDepSummary(Operation *op) {
  if (!op) return {};
  auto attr = op->getAttr(getDepSummaryAttrName());
  if (!attr) return {};
  auto arr = attr.dyn_cast<ArrayAttr>();
  if (!arr) return {};
  return parseDepSummary(arr);
}

} // namespace ADORA
} // namespace mlir

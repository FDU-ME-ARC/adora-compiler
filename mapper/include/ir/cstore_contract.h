#ifndef __CSTORE_CONTRACT_H__
#define __CSTORE_CONTRACT_H__

#include <map>
#include <string>

namespace CStoreContract {

inline void recordNonMemoryLogicalPort(std::map<int, int> &portUse,
                                       int logicalPort, bool isMemory) {
  if (!isMemory)
    ++portUse[logicalPort];
}

inline std::string logicalPortViolation(const std::map<int, int> &portUse,
                                        int operandCount) {
  for (const auto &use : portUse) {
    if (use.first < 0 || use.first >= operandCount)
      return "logical operand " + std::to_string(use.first) +
             " is out of range";
  }
  for (int operand = 0; operand < operandCount; ++operand) {
    auto use = portUse.find(operand);
    if (use == portUse.end() || use->second != 1)
      return "logical operand " + std::to_string(operand) +
             " must be covered exactly once";
  }
  return "";
}

inline std::string immediateViolation(int immediateCount) {
  if (immediateCount <= 1)
    return "";
  return "has " + std::to_string(immediateCount) +
         " immediate inputs, but the DFG representation supports at most 1";
}

} // namespace CStoreContract

#endif

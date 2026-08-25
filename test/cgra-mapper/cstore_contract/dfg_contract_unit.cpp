#include "ir/dfg_ir.h"
#include "mlir_cdfg.h"
#include "op/operations.h"

#include <iostream>
#include <string>

#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

namespace {

LLVMCDFGNode *addNode(LLVMCDFG &cdfg, mlir::MLIRContext &context,
                      const std::string &type) {
  mlir::OperationState state(mlir::UnknownLoc::get(&context), "test.node");
  return cdfg.addNode(mlir::Operation::create(state), type);
}

void addEdge(LLVMCDFG &cdfg, LLVMCDFGNode *source, LLVMCDFGNode *destination,
             int logicalPort, EdgeType type = EDGE_TYPE_DATA) {
  destination->addInputNode(source, logicalPort);
  source->addOutputNode(destination);
  cdfg.addEdge(source, destination, type);
}

} // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: cstore-dfg-contract-test <case> <operations.json>\n";
    return 2;
  }

  const std::string testCase = argv[1];
  const bool isStore = testCase.rfind("store-", 0) == 0;
  const std::string shape = isStore ? testCase.substr(6) : testCase;
  if (shape != "zero-immediate" && shape != "one-immediate" &&
      shape != "two-immediate" && shape != "missing" &&
      shape != "duplicate" && shape != "out-of-range" &&
      shape != "memory-ignored") {
    std::cerr << "unknown case: " << testCase << "\n";
    return 2;
  }

  Operations::Instance(argv[2]);
  mlir::MLIRContext context;
  context.allowUnregisteredDialects();
  LLVMCDFG cdfg("contract-unit");
  LLVMCDFGNode *destination = addNode(cdfg, context, isStore ? "store" : "CSTORE");
  const int operandCount = isStore ? 2 : 3;
  for (int operand = 0; operand < operandCount; ++operand) {
    if (shape == "missing" && operand == operandCount - 1)
      continue;
    const bool immediate =
        (shape == "one-immediate" && operand == operandCount - 1) ||
        (shape == "two-immediate" && operand >= operandCount - 2);
    LLVMCDFGNode *source =
        addNode(cdfg, context, immediate ? "CONST" : "Input");
    int logicalPort = operand;
    if (shape == "duplicate" && operand == operandCount - 1)
      logicalPort = 0;
    if (shape == "out-of-range" && operand == operandCount - 1)
      logicalPort = operandCount;
    addEdge(cdfg, source, destination, logicalPort);
  }
  if (shape == "duplicate") {
    LLVMCDFGNode *last = addNode(cdfg, context, "Input");
    addEdge(cdfg, last, destination, operandCount - 1);
  }
  if (shape == "memory-ignored") {
    LLVMCDFGNode *memorySource = addNode(cdfg, context, "Input");
    addEdge(cdfg, memorySource, destination, operandCount - 1, EDGE_TYPE_MEM);
  }

  DFGIR dfg(&cdfg);
  return dfg.getDFG() ? 0 : 2;
}

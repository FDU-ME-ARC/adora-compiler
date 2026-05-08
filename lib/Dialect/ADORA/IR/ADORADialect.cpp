//===- ADORADialect.cpp - MLIR Dialect for ADORA Kernels implementation -------===//
//===----------------------------------------------------------------------===//
//
// This file implements the ADORA kernel-related dialect and its operations.
//
//===----------------------------------------------------------------------===//

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

using namespace mlir;
using namespace mlir::ADORA;


#include "ADORA/Dialect/ADORA/IR/ADORAOpsDialect.cpp.inc"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

#define GET_TYPEDEF_CLASSES
#include "ADORA/Dialect/ADORA/IR/ADORAOpsTypes.cpp.inc"

void ADORADialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ADORA/Dialect/ADORA/IR/ADORAOps.cpp.inc"
  >();

  addOperations<
#define GET_OP_LIST
#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.cpp.inc"
  >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "ADORA/Dialect/ADORA/IR/ADORAOpsTypes.cpp.inc"
  >();
}


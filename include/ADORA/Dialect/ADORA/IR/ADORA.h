//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORA_IR_H_
#define CGRAOPT_DIALECT_ADORA_IR_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// #include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

// #include "llvm/ADT/SmallSet.h" /// use std::unordered_set instead of std::list
#include<set>

#ifndef INCLUDE_DEFINE_ADORA_DIALECT
#define INCLUDE_DEFINE_ADORA_DIALECT
#include "ADORA/Dialect/ADORA/IR/ADORAOpsDialect.h.inc"
#endif

// Types are provided via KernelOp/ADORAKernelOp.h (included below) to avoid
// double-emitting the TypeDef class definitions when both headers are used.

#define GET_OP_CLASSES
#include "ADORA/Dialect/ADORA/IR/ADORAOps.h.inc"

#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.h"
//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ADORA {
//===----------------------------------------------------------------------===//
// A templated find func for smallvector
//===----------------------------------------------------------------------===//
template <typename T, unsigned N>
inline int findElement(const llvm::SmallVector<T, N>& vec, const T& elem) {
  for (unsigned i = 0; i < vec.size(); ++i) {
    if (vec[i] == elem) {
      return i;
    }
  }
  return -1;
}

//===----------------------------------------------------------------------===//
// A templated find func for value range
//===----------------------------------------------------------------------===//
inline mlir::Value findElement(const ValueRange vec, const mlir::Value& elem) {
  for (ValueRange::iterator itr = vec.begin(); itr != vec.end(); ++itr) {
    if (*itr == elem) {
      return *itr;
    }
  }
  
  return NULL;
}
} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_IR_H_

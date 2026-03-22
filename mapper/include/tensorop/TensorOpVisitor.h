//===----------------------------------------------------------------------===//
//
// Similar to Visitor.h in scalehls project
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_TENSOR_VISITOR_H
#define ADORA_TENSOR_VISITOR_H

// #include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/ADT/TypeSwitch.h"

namespace mlir {
namespace ADORA {
// namespace ADORATensor {
// using namespace hls;

/// This class is a emitor for SSACFG operation nodes.
template <typename ConcreteType, typename ResultType, typename... ExtraArgs>
class ADORATensorOpVisitorBase {
public:
  ResultType dispatchVisitor(Operation *op, ExtraArgs... args) {
    auto *thisCast = static_cast<ConcreteType *>(this);
    return TypeSwitch<Operation *, ResultType>(op)
        .template Case<
            // ADORA Dialect
            ADORA::ADORATensor::GemmOp,
            ADORA::ADORATensor::ConvOp

            >([&](auto opNode) -> ResultType {
              return thisCast->visitOp(opNode, args...);
            })
        .Default([&](auto opNode) -> ResultType {
          return thisCast->visitInvalidOp(op, args...);
        });
  }

  /// This callback is invoked on any invalid operations.
  virtual ResultType visitInvalidOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

  /// This callback is invoked on any operations that are not handled by the
  /// concrete emitor.
  ResultType visitUnhandledOp(Operation *op, ExtraArgs... args) {
    return ResultType();
  }

#define HANDLE(OPTYPE)                                                         \
  ResultType visitOp(OPTYPE op, ExtraArgs... args) {                           \
    return static_cast<ConcreteType *>(this)->visitUnhandledOp(op, args...);   \
  }

  // ADORA dialect operations.
  HANDLE(ADORA::ADORATensor::GemmOp);
  HANDLE(ADORA::ADORATensor::ConvOp);
  // HANDLE(ADORA::DataBlockStoreOp);
  // HANDLE(ADORA::LocalMemAllocOp);
  // HANDLE(ADORA::KernelOp);

#undef HANDLE
};

} // namespace ADORA
} // namespace mlir

#endif // CGRV_EMITCGRA_EMITOR_H

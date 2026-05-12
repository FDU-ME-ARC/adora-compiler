//===- ConvertKernelCallToLLVMPass.h - Pass entrypoint ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_CONVERSION_KERNELCALLTOLLVM_H_
#define ADORA_CONVERSION_KERNELCALLTOLLVM_H_

#include <memory>
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
class LowerToLLVMOptions;
class ModuleOp;
template <typename T>
class OperationPass;
class Pass;
namespace ADORA {

/// Convert outer affine.for to scf.for (inner loops stay affine).
LogicalResult affineForOuterToSCF(func::FuncOp func, unsigned outerDepth = 1);

std::unique_ptr<OperationPass<ModuleOp>> createConvertKernelCallToLLVMPass();
std::unique_ptr<OperationPass<ModuleOp>> createConvertADORAToSCFPass();
std::unique_ptr<OperationPass<ModuleOp>> createMathRewritePass();
std::unique_ptr<OperationPass<ModuleOp>> createADORAAsyncRuntimeToLLVMPass();
// std::unique_ptr<OperationPass<ModuleOp>>
// createConvertKernelCallToLLVMPass(const LowerToLLVMOptions &options);
#define GEN_PASS_REGISTRATION
#include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h.inc"
}
} // namespace mlir

#endif // ADORA_CONVERSION_KERNELCALLTOLLVM_H_

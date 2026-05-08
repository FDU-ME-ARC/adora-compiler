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

namespace mlir {
class LowerToLLVMOptions;
class ModuleOp;
template <typename T>
class OperationPass;
class Pass;
namespace ADORA {



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

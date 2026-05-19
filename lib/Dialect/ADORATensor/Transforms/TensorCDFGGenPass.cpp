//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation for ADORA Tensor
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"

#include <iostream>
#include <string>
#include <bit>

// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Misc/DFG.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOpLowerToKernel.h"
#include "PassDetail.h"
// lower tensor op
#include "../../../../mapper/include/tensorop/TensorOp.h"


// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-gen-tensor-op-cdfg"

namespace mlir{
namespace ADORA{
namespace ADORATensor{

class ADORATensorOpCdfgGenPass : public ADORATensorOpCdfgGenPassBase<ADORATensorOpCdfgGenPass>
{
  void runOnOperation() override;
};

std::unique_ptr<OperationPass<ModuleOp>> createADORATensorOpCdfgGenPass()
{
  return std::make_unique<ADORATensorOpCdfgGenPass>();
}

}
}
} // namespace

void ADORATensorOpCdfgGenPass::runOnOperation()
{
  mlir::ModuleOp m = getOperation();

  /// Get function name
  func::FuncOp Func;
  unsigned cnt = 0;
  for (auto FuncOp : getOperation().getOps<func::FuncOp>())
  {
    // cnt++;
    Func = FuncOp;
    std::string funcname = Func.getSymName().str();

    // Get loop level from scf ForOP
    /// Generating DFG
    std::string GeneralOpNameFile_str;
    if (GeneralOpNameFile == nullptr) {
      GeneralOpNameFile_str = "lib/DFG/Documents/GeneralOpName.txt";
    }
    else
      GeneralOpNameFile_str = GeneralOpNameFile;
    // LLVMCDFG *CDFG = new LLVMCDFG(funcname, GeneralOpNameFile_str);

    // ADORA::KernelOp kernel;
    // OpBuilder b(m);
    // m->walk([&](ADORA::KernelOp k){
    //   kernel = k;
    //   WalkResult::interrupt();
    // });
    int kernel_cnt = 0;

    TensorOpCDFGVisitor engine(m->getContext());
    m.walk([&](mlir::Operation* op) {
      engine.dispatchVisitor(op);
    });
    // simplifyLoopLevelsInModuleOp(m);
    // m.dump();
    m.walk([&](ADORA::KernelOp kernel) {
      std::string kernelName = kernel.getKernelName();
      if(kernelName.empty()){
        kernelName = "kernel_" + std::to_string(kernel_cnt);
      }
      LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
      generateCDFGfromKernel(CDFG, kernel, /*verbose=*/false);
      CDFG->CDFGtoDOT(funcname + "_" + kernelName + "_CDFG.dot");
      kernel_cnt++;
    });   

    // generateCDFGfromKernel(CDFG, kernel);

    // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");
  }
  // assert(cnt == 1 && "There should be only 1 topFunc in IR Module.");

}
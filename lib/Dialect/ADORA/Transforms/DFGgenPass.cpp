//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"

#include <iostream>
#include <string>
#include <bit>
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "PassDetail.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "../../../DFG/inc/mlir_cdfg.h"
#include "ADORA/Misc/DFG.h"

// For Block handle 
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-dfg-gen"

namespace
{
  class ADORALoopCdfgGenPass : public ADORALoopCdfgGenBase<ADORALoopCdfgGenPass>
  {
    void runOnOperation() override;
  };
} // namespace

void ADORALoopCdfgGenPass::runOnOperation()
{
  mlir::Operation *m = getOperation();
  LLVM_DEBUG(m->dump());

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
    bool generationFailed = false;
    m->walk([&](ADORA::KernelOp kernel) {
      std::string kernelName = kernel.getKernelName();
      if(kernelName.empty()){
        kernelName = "kernel_" + std::to_string(kernel_cnt);
      }
      LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
      if (failed(generateCDFGfromKernel(CDFG, kernel, /*verbose=*/Verbose))) {
        delete CDFG;
        generationFailed = true;
        return WalkResult::interrupt();
      }
      CDFG->CDFGtoDOT(kernelName + "_CDFG.dot");
      delete CDFG;
      kernel_cnt++;
      return WalkResult::advance();
    });
    if (generationFailed) {
      signalPassFailure();
      return;
    }

    // generateCDFGfromKernel(CDFG, kernel);

    // CDFG->CDFGtoDOT(CDFG->name_str()+"_CDFG.dot");
  }
  // assert(cnt == 1 && "There should be only 1 topFunc in IR Module.");

}

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createADORALoopCdfgGenPass()
{
  return std::make_unique<ADORALoopCdfgGenPass>();
}

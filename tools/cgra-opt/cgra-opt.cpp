#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/ToolOutputFile.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Transforms/Passes.h"
#include "ADORA/Misc/Passes.h"


#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"

// Defined in the test directory, no public header.
namespace mlir {
void registerTestLoopPermutationPass();
namespace test {

int registerTestLinalgCodegenStrategy();
} // namespace test
} // namespace mlir

// Register important affine passes
inline void registerAffinePassesForADORA() {

  mlir::affine::registerAffineDataCopyGenerationPass();
  mlir::affine::registerAffineLoopInvariantCodeMotionPass();
  mlir::affine::registerAffineLoopTilingPass();
  mlir::affine::registerAffineLoopFusionPass();
  mlir::affine::registerAffineLoopUnrollPass();
  mlir::affine::registerAffineScalarReplacementPass();

  // my add
  mlir::affine::registerAffineLoopUnrollAndJamPass();
  mlir::affine::registerAffineLoopNormalizePass();
  mlir::affine::registerSimplifyAffineStructuresPass();

  // Test passes
  mlir::registerTestLoopPermutationPass();
}

int main(int argc, char **argv) {
  // mlir::registerAllDialects();
  // mlir::registerAllPasses();
  mlir::DialectRegistry registry;

  //===--------------------------------------------------------------------===//
  // Register mlir dialects and passes
  //===--------------------------------------------------------------------===//

  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerCSEPass();
  mlir::registerLinalgPasses();

  registerAffinePassesForADORA();
  mlir::bufferization::registerPromoteBuffersToStackPass();

  mlir::registerConvertLinalgToStandardPass();
  // mlir::registerConvertLinalgToLLVMPass(); // This pass maps linalg to blas
  // mlir::registerLinalgLowerToAffineLoopsPass();
  mlir::registerConvertLinalgToAffineLoopsPass();
  mlir::registerConvertFuncToLLVMPass();

  mlir::registerFinalizeMemRefToLLVMConversionPass();
  // mlir::registerExpandStridedMetadata();
  // mlir::memref::registerExpandOpsPass();
  // mlir::memref::registerNormalizeMemRefsPass();
  mlir::memref::registerMemRefPasses();

  mlir::registerSCFToControlFlowPass();
  mlir::registerConvertAffineToStandardPass();
  mlir::registerConvertMathToLLVMPass();
  mlir::registerConvertMathToLibmPass();
  mlir::registerArithToLLVMConversionPass();
  mlir::arith::registerArithExpandOpsPass();

  mlir::registerReconcileUnrealizedCastsPass();

  // Add the following to selectively include the necessary dialects. You only
  // need to register dialects that will be *parsed* by the tool, not the one
  // generated
  // clang-format off
  registry.insert<mlir::func::FuncDialect,
                  mlir::memref::MemRefDialect,
                  mlir::LLVM::LLVMDialect,
                  mlir::linalg::LinalgDialect,
                  mlir::math::MathDialect,
                  mlir::scf::SCFDialect,
                  mlir::cf::ControlFlowDialect,
                  mlir::vector::VectorDialect,
                  mlir::arith::ArithDialect,
                  mlir::affine::AffineDialect,
                  mlir::DLTIDialect,
                  mlir::ml_program::MLProgramDialect,
                  mlir::tensor::TensorDialect,
                  mlir::bufferization::BufferizationDialect>();

  // Dialects
  registry.insert<mlir::ADORA::ADORADialect,
                  mlir::ADORA::ADORATensor::ADORATensorDialect>();

  // ----- My Dialect -----
  mlir::ADORA::registerADORALoopCdfgGenPass();
  mlir::ADORA::registerExtractAffineForToKernelPass();
  mlir::ADORA::registerAdjustKernelMemoryFootprintPass();
  mlir::ADORA::registerExtractKernelToFuncPass();
  mlir::ADORA::registerAutoDesignSpaceExplorePass();
  mlir::ADORA::registerSimplifyAffineLoopLevelsPass();
  mlir::ADORA::registerTestPrintOpNestingPass();
  mlir::ADORA::registerSimplifyLoadStoreInLoopNestPass();
  mlir::ADORA::registerADORAAffineLoopUnroll();
  mlir::ADORA::registerADORALoopUnrollAndJam();
  mlir::ADORA::registerADORAAutoUnroll();
  mlir::ADORA::registerAffineLoopReorder();
  mlir::ADORA::registerScheduleADORATasks();
  mlir::ADORA::registerLLMPipelineSchedule();
  // Async-token backend passes are disabled: dependency ordering is carried by
  // SSA !ADORA.token from schedule-tasks; the pytest emit path consumes tokens
  // directly and does not lower them. Source files are kept but not registered.
  //   registerAssignStreamsPass / registerBufferReusePass /
  //   registerLowerAsyncTokensPass / registerADORAAsyncRuntimeToLLVMPass
  // mlir::ADORA::registerAssignStreamsPass();
  // mlir::ADORA::registerBufferReusePass();
  // mlir::ADORA::registerLowerAsyncTokensPass();
  // mlir::ADORA::registerADORAAsyncRuntimeToLLVMPass();

  mlir::ADORA::registerConvertKernelCallToLLVMPass();
  mlir::ADORA::registerConvertADORAToSCFPass();
  mlir::ADORA::registerMathRewrite();

  /// ADORATensor
  mlir::ADORA::ADORATensor::registerLinalgToSystolicGEMMPass();

  mlir::registerSCFForLoopCanonicalizationPass();
  

  return failed(
      mlir::MlirOptMain(argc, argv, "Fail\n", registry));
}

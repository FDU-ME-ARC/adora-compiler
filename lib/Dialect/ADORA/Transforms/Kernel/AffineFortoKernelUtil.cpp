#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"

#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

using namespace llvm; // for llvm.errs()
// using namespace llvm::detail;
using namespace mlir;
using namespace mlir::ADORA;

//===----------------------------------------------------------------------===//
// SpecifiedAffineFortoKernel
//===----------------------------------------------------------------------===//
OpTable TramUnsupportOpTable = 
{
  /// math dialect
  // ::mlir::math::ExpOp::getOperationName(), // math.exp now we support it with math rewrite
  ::mlir::math::ErfOp::getOperationName(), // math.erf
  /// controlflow dialect
  ::mlir::cf::AssertOp::getOperationName(), // cf.erf
  /// arith dialect
  // ::mlir::arith::CmpFOp::getOperationName() // arith.cmpf
};
LogicalResult mlir::ADORA::SpecifiedAffineFortoKernel(mlir::affine::AffineForOp& kernelforOp){
  /// Walk every op in this forop to check whether unsupport op is contained
  auto WalkResult = kernelforOp.getBody()->walk([&](mlir::Operation* op){
    StringRef opname = op->getName().getStringRef();
    if(TramUnsupportOpTable.count(opname)){ /// unsupported op is contained by forop
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if(WalkResult.wasInterrupted()){
    llvm::errs() << "[Info] Containing unsupported operations.\n" ;
    return LogicalResult::failure();
  }

  OpBuilder builder(kernelforOp.getOperation());

  // Create a kernel op and move the body region of the innermost loop into it
  Location loc = kernelforOp.getLoc();
  auto KernelOp = builder.create<ADORA::KernelOp>(loc);
  builder.setInsertionPointToEnd(&KernelOp.getBody().front());
  builder.create<ADORA::TerminatorOp>(loc);
  builder.setInsertionPointToStart(&KernelOp.getBody().front());

  // Move the selected loop op into the Kernel body.
  // Note: we must splice from the loop's *parent block* operation list.
  auto &parentOps = kernelforOp->getBlock()->getOperations();
  KernelOp.getBody().front().getOperations().splice(
      KernelOp.getBody().front().begin(), parentOps,
      Block::iterator(kernelforOp));

  return LogicalResult::success();
}

LogicalResult mlir::ADORA::SpecifiedAffineFortoKernel(mlir::affine::AffineForOp& kernelforOp, std::string kernel_name){

  if(SpecifiedAffineFortoKernel(kernelforOp).succeeded()){
    if(kernel_name != "")
      dyn_cast<ADORA::KernelOp>(kernelforOp.getOperation()->getParentOp()).setKernelName(kernel_name);
    return LogicalResult::success();
  }

  return LogicalResult::failure();
}
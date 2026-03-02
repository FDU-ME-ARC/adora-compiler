//===- SCFToKernelPass.cpp - Convert a loop nest to a kernel to be accelerated -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
// #include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Parser/Parser.h"
#include <iostream>
#include <sstream>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "../PassDetail.h"


using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// AffineToKERNEL
//===----------------------------------------------------------------------===//
#define DEBUG_TYPE "adora-extract-for-to-kernel"

namespace {
  
/// Collect the "kernel loops" under `rootFor` following the rule:
/// - We want strictly nested loop chains (no sibling `affine.for` at any level).
/// - Scan from outer to inner. If any sibling loops exist under a loop, the
///   siblings themselves become kernels, and we keep applying the same rule
///   recursively for each sibling.
static void getImmediateChildForOps(AffineForOp forOp,
                                   SmallVectorImpl<AffineForOp> &out) {
  for (Operation &op : forOp.getBody()->getOperations()) {
    if (auto child = dyn_cast<AffineForOp>(op))
      out.push_back(child);
  }
}

/// If a sibling `affine.for` is found at or under `forOp`, collect the final
/// kernel loops (possibly deeper than that sibling level) into `outKernels` and
/// return true. If no sibling exists anywhere under `forOp`, return false.
static bool collectKernelsIfSiblingExists(AffineForOp forOp,
                                         SmallVectorImpl<AffineForOp> &outKernels) {
  SmallVector<AffineForOp, 4> children;
  getImmediateChildForOps(forOp, children);
  if (children.empty())
    return false;

  if (children.size() == 1)
    return collectKernelsIfSiblingExists(children.front(), outKernels);

  // Sibling loops exist at this level.
  for (AffineForOp child : children) {
    SmallVector<AffineForOp, 4> childKernels;
    if (collectKernelsIfSiblingExists(child, childKernels)) {
      outKernels.append(childKernels.begin(), childKernels.end());
    } else {
      // No further sibling under this child => the child itself is a kernel.
      outKernels.push_back(child);
    }
  }
  return true;
}

static void appendKernelLoopsFromRoot(AffineForOp rootFor,
                                     SmallVectorImpl<AffineForOp> &outKernels) {
  SmallVector<AffineForOp, 8> kernels;
  if (collectKernelsIfSiblingExists(rootFor, kernels))
    outKernels.append(kernels.begin(), kernels.end());
  else
    outKernels.push_back(rootFor);
}

static void collectKernelLoopsInOp(Operation *op,
                                  llvm::SmallPtrSetImpl<Operation *> &seen,
                                  SmallVectorImpl<AffineForOp> &outKernels) {
  for (Region &region : op->getRegions()) {
    for (Block &block : region.getBlocks()) {
      // First, handle immediate affine.for ops in this block.
      SmallVector<AffineForOp, 8> immediateForOps;
      for (Operation &nested : block.getOperations())
        if (auto forOp = dyn_cast<AffineForOp>(nested))
          immediateForOps.push_back(forOp);

      for (AffineForOp forOp : immediateForOps) {
        SmallVector<AffineForOp, 8> kernels;
        appendKernelLoopsFromRoot(forOp, kernels);
        for (AffineForOp k : kernels) {
          Operation *kOp = k.getOperation();
          if (seen.insert(kOp).second)
            outKernels.push_back(k);
        }
      }
    }
  }
}

// A pass that traverses top-level loops in the function and converts them to
// ADORA Kernel operations.  
struct AffineForKernelCaptor: public ExtractAffineForToKernelBase<AffineForKernelCaptor> {
  AffineForKernelCaptor() = default;
  void runOnOperation() override {
    if(FunctionName!="-"){
      /// function name is specified
      llvm::SmallVector<std::string> functions;
      std::string token;
      std::stringstream ss(FunctionName);
      while (std::getline(ss, token, ',')) {
        functions.push_back(token); 
      }
      std::string ThisFunction = getOperation().getSymName().str();

      if(findElement(functions, ThisFunction) == -1){
        /// this function is not specified
        return ;
      }
    }

    unsigned kernel_count = 0;
    SmallVector<AffineForOp, 16> kernelForOps;
    llvm::SmallPtrSet<Operation *, 32> seenKernelOps;
    collectKernelLoopsInOp(getOperation().getOperation(), seenKernelOps,
                           kernelForOps);

    for (AffineForOp forOp : kernelForOps) {
      LLVM_DEBUG(forOp.dump());
      auto ForWalkResult = forOp.walk([&](Operation *op){ 
        if(
            op->getName().getStringRef()== AffineLoadOp::getOperationName() ||
            op->getName().getStringRef()== AffineStoreOp::getOperationName()||
            op->getName().getStringRef()== AffineForOp::getOperationName()||
            op->getName().getStringRef()== AffineIfOp ::getOperationName()||
            op->getName().getStringRef()== AffinePrefetchOp ::getOperationName()||
            op->getName().getStringRef()== AffineVectorLoadOp ::getOperationName()||
            op->getName().getStringRef()== AffineVectorStoreOp ::getOperationName()||
            op->getName().getStringRef()== AffineYieldOp ::getOperationName() ||
            op->getName().getStringRef()== mlir::memref::LoadOp ::getOperationName()||
            /// arith
            op->getName().getStringRef()== mlir::arith::TruncFOp ::getOperationName() ||
            op->getName().getStringRef()== mlir::arith::TruncIOp ::getOperationName() ||
            op->getName().getStringRef()== mlir::arith::UIToFPOp ::getOperationName() )
        {
          return WalkResult::advance();
        }
        else
          return WalkResult::interrupt();
      });
    
      if(ForWalkResult.wasInterrupted()){
        if(mlir::succeeded(ADORA::SpecifiedAffineFortoKernel(forOp)))
          kernel_count++;
      }
    }
    

    /// rename kernels
    if(kernel_count==1){
        /** Initialize name of kernel **/
        ADORA::KernelOp kn = *(getOperation().getOps<ADORA::KernelOp>().begin());
        std::string kernelname = getOperation().getSymName().str();
        kn.setKernelName(kernelname);
    }
    else{
      kernel_count = 0;
      getOperation().walk([&](ADORA::KernelOp kn){
        std::string kernelname = getOperation().getSymName().str() + 
          "_" + std::to_string(kernel_count++);
        kn.setKernelName(kernelname); 
      });
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> mlir::ADORA::createExtractAffineForToKernelPass() {
  return std::make_unique<AffineForKernelCaptor>();
}

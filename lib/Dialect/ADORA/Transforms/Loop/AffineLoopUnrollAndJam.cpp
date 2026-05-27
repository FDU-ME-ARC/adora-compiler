//===- LoopUnrollAndJam.cpp - Code to perform loop unroll and jam ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements loop unroll and jam. Unroll and jam is a transformation
// that improves locality, in particular, register reuse, while also improving
// operation level parallelism. The example below shows what it does in nearly
// the general case. Loop unroll and jam currently works if the bounds of the
// loops inner to the loop being unroll-jammed do not depend on the latter.
//
// Before      After unroll and jam of i by factor 2:
//
//             for i, step = 2
// for i         S1(i);
//   S1;         S2(i);
//   S2;         S1(i+1);
//   for j       S2(i+1);
//     S3;       for j
//     S4;         S3(i, j);
//   S5;           S4(i, j);
//   S6;           S3(i+1, j)
//                 S4(i+1, j)
//               S5(i);
//               S6(i);
//               S5(i+1);
//               S6(i+1);
//
// Note: 'if/else' blocks are not jammed. So, if there are loops inside if
// op's, bodies of those loops will not be jammed.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/FileSystem.h"
#include <optional>
#include <filesystem>

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Misc/DFG.h"
#include "../PassDetail.h"

#define DEBUG_TYPE "adora-loop-unroll-jam"

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

// #define KernelLoop(k) k.getOps<AffineForOp>().begin();
/// @brief find all unrolling factors (which divides trip count)
/// @param affinefor
/// @return vector which contains all unrolling factors
static SmallVector<unsigned> FindUnrollingFactors(affine::AffineForOp& affinefor){
  // assert(Node.IsInnermost()&&"Only innermost loop-nest can be unrolled.");
  auto optionalTripCount = getConstantTripCount(affinefor);
  assert(optionalTripCount&&"Variable loop bound!");
  SmallVector<unsigned> validFactors;
  unsigned factor = 1;
  unsigned tripCount = optionalTripCount.value();
  while (factor <= tripCount) {
    /// Push back the current factor.
    /// unrolling factor = 1 means no unrolling applied
    validFactors.push_back(factor);

    // Find the next possible size.
    ++factor;
    while (factor <= tripCount && tripCount % factor != 0)
      ++factor;
  }
  // Node.UnrollFactors = std::move(validFactors);
  return validFactors;
}

// Gathers all maximal sub-blocks of operations that do not themselves
// include a for op (a operation could have a descendant for op though
// in its tree).  Ignore the block terminators.
struct ADORAJamBlockGatherer {
  // Store iterators to the first and last op of each sub-block found.
  std::vector<std::pair<Block::iterator, Block::iterator>> subBlocks;
  ADORAJamBlockGatherer(){}
  // This is a linear time walk.
  void walk(Operation *op) {
    for (auto &region : op->getRegions())
      for (auto &block : region)
        walk(block);
  }

  void walk(Block &block) {
    // block.dump();
    for (auto it = block.begin(), e = std::prev(block.end()); it != e;) {
      auto subBlockStart = it;
      while (it != e && !isa<AffineForOp>(&*it) && !isa<ADORA::KernelOp>(&*it))
        ++it;
      if (it != subBlockStart)
        subBlocks.emplace_back(subBlockStart, std::prev(it));
      // Process all for ops that appear next.
      while (it != e && (isa<AffineForOp>(&*it) || isa<ADORA::KernelOp>(&*it)) )
        walk(&*it++);
    }
  }
};

/// Computes the cleanup loop lower bound of the loop being unrolled with
/// the specified unroll factor; this bound will also be upper bound of the main
/// part of the unrolled loop. Computes the bound as an AffineMap with its
/// operands or a null map when the trip count can't be expressed as an affine
/// expression.
// static void
// getCleanupLoopLowerBound(AffineForOp forOp, unsigned unrollFactor,
//                          AffineMap &cleanupLbMap,
//                          SmallVectorImpl<mlir::Value> &cleanupLbOperands) {
//   AffineMap tripCountMap;
//   SmallVector<mlir::Value, 4> tripCountOperands;
//   getTripCountMapAndOperands(forOp, &tripCountMap, &tripCountOperands);
//   // Trip count can't be computed.
//   if (!tripCountMap) {
//     cleanupLbMap = AffineMap();
//     return;
//   }

//   OpBuilder b(forOp);
//   auto lbMap = forOp.getLowerBoundMap();
//   auto lb = b.create<AffineApplyOp>(forOp.getLoc(), lbMap,
//                                     forOp.getLowerBoundOperands());

//   // For each upper bound expr, get the range.
//   // Eg: affine.for %i = lb to min (ub1, ub2),
//   // where tripCountExprs yield (tr1, tr2), we create affine.apply's:
//   // lb + tr1 - tr1 % ufactor, lb + tr2 - tr2 % ufactor; the results of all
//   // these affine.apply's make up the cleanup loop lower bound.
//   // copy from LoopUtils.cpp
//   SmallVector<AffineExpr, 4> bumpExprs(tripCountMap.getNumResults());
//   SmallVector<mlir::Value, 4> bumpValues(tripCountMap.getNumResults());
//   int64_t step = forOp.getStep().getSExtValue();
//   for (unsigned i = 0, e = tripCountMap.getNumResults(); i < e; i++) {
//     auto tripCountExpr = tripCountMap.getResult(i);
//     bumpExprs[i] = (tripCountExpr - tripCountExpr % unrollFactor) * step;
//     auto bumpMap = AffineMap::get(tripCountMap.getNumDims(),
//                                   tripCountMap.getNumSymbols(), bumpExprs[i]);
//     bumpValues[i] =
//         b.create<AffineApplyOp>(forOp.getLoc(), bumpMap, tripCountOperands);
//   }

//   SmallVector<AffineExpr, 4> newUbExprs(tripCountMap.getNumResults());
//   for (unsigned i = 0, e = bumpExprs.size(); i < e; i++)
//     newUbExprs[i] = b.getAffineDimExpr(0) + b.getAffineDimExpr(i + 1);

//   cleanupLbOperands.clear();
//   cleanupLbOperands.push_back(lb);
//   cleanupLbOperands.append(bumpValues.begin(), bumpValues.end());
//   cleanupLbMap = AffineMap::get(1 + tripCountMap.getNumResults(), 0, newUbExprs,
//                                 b.getContext());
//   // Simplify the cleanupLbMap + cleanupLbOperands.
//   fullyComposeAffineMapAndOperands(&cleanupLbMap, &cleanupLbOperands);
//   cleanupLbMap = simplifyAffineMap(cleanupLbMap);
//   canonicalizeMapAndOperands(&cleanupLbMap, &cleanupLbOperands);
//   // Remove any affine.apply's that became dead from the simplification above.
//   for (auto v : bumpValues)
//     if (v.use_empty())
//       v.getDefiningOp()->erase();

//   if (lb.use_empty())
//     lb.erase();
// }

// /// Helper to generate cleanup loop for unroll or unroll-and-jam when the trip
// /// count is not a multiple of `unrollFactor`.
// /// Copy from LoopUntils.cpp
// static LogicalResult generateCleanupLoopForUnroll(AffineForOp forOp,
//                                                   uint64_t unrollFactor) {
//   // Insert the cleanup loop right after 'forOp'.
//   OpBuilder builder(forOp->getBlock(), std::next(Block::iterator(forOp)));
//   auto cleanupForOp = cast<AffineForOp>(builder.clone(*forOp));

//   // Update uses of `forOp` results. `cleanupForOp` should use `forOp` result
//   // and produce results for the original users of `forOp` results.
//   auto results = forOp.getResults();
//   auto cleanupResults = cleanupForOp.getResults();
//   auto cleanupIterOperands = cleanupForOp.getInits();

//   for (auto e : llvm::zip(results, cleanupResults, cleanupIterOperands)) {
//     std::get<0>(e).replaceAllUsesWith(std::get<1>(e));
//     cleanupForOp->replaceUsesOfWith(std::get<2>(e), std::get<0>(e));
//   }

//   AffineMap cleanupMap;
//   SmallVector<mlir::Value> cleanupOperands;
//   getCleanupLoopLowerBound(forOp, unrollFactor, cleanupMap, cleanupOperands);
//   if (!cleanupMap)
//     return failure();

//   cleanupForOp.setLowerBound(cleanupOperands, cleanupMap);
//   // Promote the loop body up if this has turned into a single iteration loop.
//   (void)promoteIfSingleIteration(cleanupForOp);

//   // Adjust upper bound of the original loop; this is the same as the lower
//   // bound of the cleanup loop.
//   forOp.setUpperBound(cleanupOperands, cleanupMap);
//   return success();
// }

LogicalResult mlir::ADORA::loopUnrollAndJamByFactor(affine::AffineForOp& forop, unsigned ur_factor){
  if(ur_factor == 1) return success();
  /************************************
    * Unroll outer loop
    ************************************/
  ADORAJamBlockGatherer jbg;
  jbg.walk(forop);
  auto &subBlocks = jbg.subBlocks;
  // kernel.dump();
  // `operandMaps[i - 1]` carries old->new operand mapping for the ith unrolled
  // iteration. There are (`unrollJamFactor` - 1) iterations.
  SmallVector<IRMapping, 4> operandMaps(ur_factor - 1);

  // Collect loops with iter_args.
  SmallVector<AffineForOp, 4> loopsWithIterArgs;
  forop.walk([&](AffineForOp aForOp) {
    if (aForOp.getNumIterOperands() > 0)
      loopsWithIterArgs.push_back(aForOp);
  });

  // For any loop with iter_args, replace it with a new loop that has
  // `unrollJamFactor` copies of its iterOperands, iter_args and yield
  // operands.
  SmallVector<AffineForOp, 4> newLoopsWithIterArgs;
  IRRewriter rewriter(forop.getContext());
  for (AffineForOp oldForOp : loopsWithIterArgs) {
    // oldForOp.dump();
    SmallVector<mlir::Value> dupIterOperands, dupYieldOperands;
    ValueRange oldIterOperands = oldForOp.getInits();
    ValueRange oldIterArgs = oldForOp.getRegionIterArgs();
    ValueRange oldYieldOperands =
      cast<AffineYieldOp>(oldForOp.getBody()->getTerminator()).getOperands();
    // Get additional iterOperands, iterArgs, and yield operands. We will
    // fix iterOperands and yield operands after cloning of sub-blocks.
    for (unsigned i = ur_factor - 1; i >= 1; --i) {
      dupIterOperands.append(oldIterOperands.begin(), oldIterOperands.end());
      dupYieldOperands.append(oldYieldOperands.begin(), oldYieldOperands.end());
    }
    // Create a new loop with additional iterOperands, iter_args and yield
    // operands. This new loop will take the loop body of the original loop.
    bool forOpReplaced = oldForOp == forop;
    AffineForOp newForOp =
      cast<AffineForOp>(*oldForOp.replaceWithAdditionalYields(
          rewriter, dupIterOperands, /*replaceInitOperandUsesInLoop=*/false,
          [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs) {
            return dupYieldOperands;
          }));
    newLoopsWithIterArgs.push_back(newForOp);
    // `forOp` has been replaced with a new loop.
    if (forOpReplaced)
      forop = newForOp;
    // Update `operandMaps` for `newForOp` iterArgs and results.
    ValueRange newIterArgs = newForOp.getRegionIterArgs();
    unsigned oldNumIterArgs = oldIterArgs.size();
    ValueRange newResults = newForOp.getResults();
    unsigned oldNumResults = newResults.size() / ur_factor;
    assert(oldNumIterArgs == oldNumResults &&
         "oldNumIterArgs must be the same as oldNumResults");
    for (unsigned i = ur_factor - 1; i >= 1; --i) {
      for (unsigned j = 0; j < oldNumIterArgs; ++j) {
        // `newForOp` has `unrollJamFactor` - 1 new sets of iterArgs and
        // results. Update `operandMaps[i - 1]` to map old iterArgs and results
        // to those in the `i`th new set.
        operandMaps[i - 1].map(newIterArgs[j],
                             newIterArgs[i * oldNumIterArgs + j]);
        operandMaps[i - 1].map(newResults[j],
                             newResults[i * oldNumResults + j]);
      }
    }
  }

  // Scale the step of loop being unroll-jammed by the unroll-jam factor.
  int64_t step = forop.getStep().getSExtValue();
  forop.setStep(step * ur_factor);

  auto forOpIV = forop.getInductionVar();
  // Unroll and jam (appends unrollJamFactor - 1 additional copies).
  for (unsigned i = ur_factor - 1; i >= 1; --i) {
    for (auto &subBlock : subBlocks) {
      LLVM_DEBUG(subBlock.first->getBlock()->dump());
      // Builder to insert unroll-jammed bodies. Insert right at the end of
      // sub-block.
      OpBuilder builder(subBlock.first->getBlock(), std::next(subBlock.second));

      // If the induction variable is used, create a remapping to the value for
      // this unrolled instance.
      if (!forOpIV.use_empty()) {
        // iv' = iv + i * step, i = 1 to unrollJamFactor-1.
        auto d0 = builder.getAffineDimExpr(0);
        auto bumpMap = AffineMap::get(1, 0, d0 + i * step);
        auto ivUnroll =
          builder.create<AffineApplyOp>(forop.getLoc(), bumpMap, forOpIV);
        operandMaps[i - 1].map(forOpIV, ivUnroll);
      }
      // Clone the sub-block being unroll-jammed.
      for (auto it = subBlock.first; it != std::next(subBlock.second); ++it){
        builder.clone(*it, operandMaps[i - 1]);
        LLVM_DEBUG(forop.dump(); llvm::errs() << "\n";);
      }
      
    }
    // Fix iterOperands and yield op operands of newly created loops.
    for (auto newForOp : newLoopsWithIterArgs) {
      unsigned oldNumIterOperands =
        newForOp.getNumIterOperands() / ur_factor;
      unsigned numControlOperands = newForOp.getNumControlOperands();
      auto yieldOp = cast<AffineYieldOp>(newForOp.getBody()->getTerminator());
      unsigned oldNumYieldOperands = yieldOp.getNumOperands() / ur_factor;
    assert(oldNumIterOperands == oldNumYieldOperands &&
             "oldNumIterOperands must be the same as oldNumYieldOperands");
      for (unsigned j = 0; j < oldNumIterOperands; ++j) {
          // The `i`th duplication of an old iterOperand or yield op operand
          // needs to be replaced with a mapped value from `operandMaps[i - 1]`
          // if such mapped value exists.
        newForOp.setOperand(numControlOperands + i * oldNumIterOperands + j,
                            operandMaps[i - 1].lookupOrDefault(
                                newForOp.getOperand(numControlOperands + j)));
        yieldOp.setOperand(
            i * oldNumYieldOperands + j,
          operandMaps[i - 1].lookupOrDefault(yieldOp.getOperand(j)));
      }
    }
    // forop.dump();
  }
    // if (forop.getNumResults() > 0) {
    //   // Create reduction ops to combine every `unrollJamFactor` related results
    //   // into one value. For example, for %0:2 = affine.for ... and addf, we add
    //   // %1 = arith.addf %0#0, %0#1, and replace the following uses of %0#0 with
    //   // %1.
    //   rewriter.setInsertionPointAfter(forop);
    //   auto loc = forop.getLoc();
    //   unsigned oldNumResults = forop.getNumResults() / ur_factor;
    //   // for (LoopReduction &reduction : reductions) {
    //   //   unsigned pos = reduction.iterArgPosition;
    //   //   mlir::Value lhs = forop.getResult(pos);
    //   //   mlir::Value rhs;
    //   //   SmallPtrSet<Operation *, 4> newOps;
    //   //   for (unsigned i = unrollJamFactor - 1; i >= 1; --i) {
    //   //     rhs = forop.getResult(i * oldNumResults + pos);
    //   //     // Create ops based on reduction type.
    //   //     lhs = arith::getReductionOp(reduction.kind, rewriter, loc, lhs, rhs);
    //   //     if (!lhs)
    //   //       return failure();
    //   //     Operation *op = lhs.getDefiningOp();
    //   //     assert(op && "Reduction op should have been created");
    //   //     newOps.insert(op);
    //   //   }
    //   //   // Replace all uses except those in newly created reduction ops.
    //   //   forop.getResult(pos).replaceAllUsesExcept(lhs, newOps);
    //   // }
    // }

  // Promote the loop body up if this has turned into a single iteration loop.
  (void)promoteIfSingleIteration(forop);
  return success();
}


namespace {
/// Loop unroll jam pass. Currently, this just unroll jams the first
/// outer loop in a Function.
struct ADORALoopUnrollAndJam
    : public ADORALoopUnrollAndJamBase<ADORALoopUnrollAndJam> {
  unsigned NumGPE = 0;
  unsigned NumIOB = 0; 
  explicit ADORALoopUnrollAndJam() {}
  LogicalResult KernelUnrollAndJamWithResourceLimits(ADORA::KernelOp kernel, mlir::ModuleOp m);
  void runOnOperation() override;
};
} // namespace

//////////
// A wrapper for KernelUnrollAndJam;
/////////
LogicalResult ADORALoopUnrollAndJam::
      KernelUnrollAndJamWithResourceLimits(ADORA::KernelOp kernel, mlir::ModuleOp m){
  ModuleOp topmodule = m;  
  affine::AffineForOp Parentfor = dyn_cast_or_null<affine::AffineForOp> 
                                              (kernel.getOperation()->getParentOp());
  MLIRContext* context = topmodule.getContext();
  // make a new dir
  // SmallVector<std::string> DesignSpaceFiles;
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::string folderName = "DesignSpace";
  std::filesystem::path DesignSpacefolderPath = currentPath / folderName;
  std::filesystem::create_directory(DesignSpacefolderPath);

  int max_ALU = 0, max_LSU = 0;
  std::string final_FilePath="";

  std::string GeneralOpNameFile_str;
  if (GeneralOpNameFile == nullptr) {
    std::cerr << "Environment variable \"GeneralOpNameFile\" is not set." << std::endl;
    GeneralOpNameFile_str = "lib/DFG/Documents/GeneralOpName.txt";
    std::cerr << "Using fallback: \"" << GeneralOpNameFile_str << "\"" << std::endl;
  }
  else
    GeneralOpNameFile_str = GeneralOpNameFile;

  /************************************
   * Get unrolljam factor
   ************************************/
  SmallVector<unsigned> ur_factors = FindUnrollingFactors(Parentfor);
  for(auto ur_factor : ur_factors){
    topmodule = cast<ModuleOp>(m.getOperation()->clone());
    // ADORA::KernelOp kernelur = getSingleKernelFromFunc(*(topmodule.getOps<func::FuncOp>().begin()));
    ADORA::KernelOp kernelur = getKernelFromCopiedModule(topmodule, kernel);

    affine::AffineForOp Parentfor = dyn_cast_or_null<affine::AffineForOp> 
                                              (kernelur.getOperation()->getParentOp());
    if(Parentfor == NULL)
      return failure(); /// kernelur is not in a for op
    
    /***********************************
    * TODO:Dependency analysis
    ************************************/


    // Parentfor.dump();
    std::optional<uint64_t> mayBeConstantTripCount = getConstantTripCount(Parentfor);
    // If the trip count is lower than the unroll jam factor, no unroll jam.
    if (!mayBeConstantTripCount /*&& *mayBeConstantTripCount < unrollJamFactor*/) {
      LLVM_DEBUG(llvm::dbgs() << "[failed] TripCount is not constant.\n");
      return failure();
    }

    /// Unroll and jam
    (void)loopUnrollAndJamByFactor(Parentfor, ur_factor);

    std::string fileName;
    if(kernelur.hasKernelName())
      fileName = kernelur.getKernelName();
    else
      fileName = "Unrolljam";
    
    std::string filePath = DesignSpacefolderPath.string() + "/" + fileName + "_uf_" +  std::to_string(ur_factor) + ".mlir";

    LLVM_DEBUG(topmodule.dump());

    std::error_code ec;    
    llvm::raw_fd_ostream outputFile(filePath, ec, sys::fs::FA_Write);
    if (ec) {
      llvm::errs() << "Error opening file: " << ec.message() << filePath << "\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    topmodule.print(outputFile);

    /// Generating DFG
    LLVMCDFG *CDFG = new LLVMCDFG(fileName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernelur, /*verbose=*/true);
    CDFG->CDFGtoDOT(DesignSpacefolderPath.string() + "/" + CDFG->name_str()+"_CDFG_unrolljam.dot");
    
    ADORA::DFGInfo dfginfo = GetDFGinfo(CDFG);       
    if(dfginfo.Num_ALU <= NumGPE && dfginfo.Num_LSU <= NumIOB) 
    {
      if(dfginfo.Num_LSU > max_LSU || 
        (dfginfo.Num_ALU > max_ALU && dfginfo.Num_LSU == max_LSU))
      {
        final_FilePath = filePath;
        max_LSU = dfginfo.Num_LSU;
        max_ALU = dfginfo.Num_ALU;
      }
    }
    else{
      /// If the resource occupied by current dfg oversizes adg,
      /// we don't have to keep unrolling to get larger dfg.
      break;
    }
  }/// End of traversing on every design point

  std::string errorMessage;
  auto file = openInputFile(final_FilePath, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    signalPassFailure();
    return LogicalResult::failure();
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> m_ = parseSourceFile<ModuleOp>(sourceMgr, context); 
  mlir::ModuleOp moduleop = m_.get();
  SymbolTable symbolTable(moduleop.getOperation());

  topmodule = m;
  func::FuncOp oldfunc = *(topmodule.getOps<func::FuncOp>().begin());
  func::FuncOp newfunc = *(moduleop.getOps<func::FuncOp>().begin());
  newfunc.getOperation()->moveBefore(oldfunc);
  oldfunc.getOperation()->erase();


  LLVM_DEBUG(topmodule.dump());

  return success();
}



std::unique_ptr<OperationPass<mlir::ModuleOp>>
    mlir::ADORA::createADORALoopUnrollAndJamPass() {
  return std::make_unique<ADORALoopUnrollAndJam>();
}



void ADORALoopUnrollAndJam::runOnOperation() {
  if(CGRAadg == "notdefined" || CGRAadg == ""){
    LLVM_DEBUG(llvm::errs() << "CGRAadg not defined.\n");
    NumGPE = 32;
    NumIOB = 16;
  }
  else{
    NumGPE = getInstanceNumFromADG(CGRAadg, "GPE");
    NumIOB = getInstanceNumFromADG(CGRAadg, "IOB");
  }
  // if (unrollJamFactor)
  //   this->unrollJamFactor = *unrollJamFactor;
  
  // if (getOperation().isExternal())
  //   return;
  auto m = getOperation();
  // func.dump();
  SmallVector<ADORA::KernelOp> kernels;
  m.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  for(ADORA::KernelOp kernel : kernels){
    LogicalResult r = KernelUnrollAndJamWithResourceLimits(kernel, m);
    m.dump();
  }

  for(auto func : m.getOps<func::FuncOp>()){
    ResetIndexOfBlockAccessOpInFunc(func);
  }
}

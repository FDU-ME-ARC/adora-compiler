//===- AffineLoopUnroll.cpp - customed loop unroll -----------===//

#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AffineExprVisitor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Parser/Parser.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/GraphWriter.h"

#include <numeric>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>
#include <stack>

#include "../../../../DFG/inc/mlir_cdfg.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Misc/DFG.h"
#include "../PassDetail.h"


using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
#define DEBUG_TYPE "adora-affine-loop-unroll"
namespace {
struct ADORAAffineLoopUnrollPass : public ADORAAffineLoopUnrollBase<ADORAAffineLoopUnrollPass> {
  // ADORAAffineLoopUnrollPass() = default;
  unsigned NumGPE = 0;
  unsigned NumIOB = 0; 
  explicit ADORAAffineLoopUnrollPass() { }
  /* Function define */

  SmallVector<DesignPoint> ConstructTilingUnrollSpace(SmallVector<ADORA::ForNode> ForNodes);
  // SmallVector<SmallVector<unsigned>> ConstructUnrollSpace(SmallVector<ADORA::ForNode> ForNodes);
  // LogicalResult loopUnrollByFactor_opt(AffineForOp forOp, uint64_t unrollFactor,
  //   // function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
  //   bool cleanUpUnroll); /// move to DSE.h
  LogicalResult KernelUnrollWithResourceLimits(ADORA::KernelOp kernel, mlir::ModuleOp& m);
  void runOnOperation() override; 

};
} // namespace

/// @brief A design point is a number sequence:
///        tilefactor(loop0),unrollfactor(loop0),tilefactor(loop1),unrollfactor(loop1).......
/// for example:
///       (1,1),(1,1),(1,3) represents a 3-level nested loop and the innermost loop is unrolled by 3
/// @param ForNodes All for loops
/// @return A SmallVector containing all design point
SmallVector<DesignPoint> ADORAAffineLoopUnrollPass::
            ConstructTilingUnrollSpace(SmallVector<ADORA::ForNode> ForNodes){
  SmallVector<DesignPoint> AllDesignSpace;
  DesignPoint point;
  unsigned point_num = 1;
  for (ADORA::ForNode Node : ForNodes) {
    AllDesignSpace = ExpandTilingAndUnrollingFactors(Node, AllDesignSpace);
    point_num *= Node.UnrollFactors.size();
  }

  /*** Print Design Space ***/
  llvm::errs() <<"//-----------  Design Space  ----------//\n";
  llvm::errs() <<"Loop-nested structure:\n";
  for (ADORA::ForNode Node : ForNodes) {
    if (!Node.HasParentFor())/// outermost loop
      Node.dumpTree();
  }

  llvm::errs() <<"Design Space:\n";
  for(DesignPoint point : AllDesignSpace){
    bool bracket_flag = 1;
    for(unsigned factor : point){
      if(bracket_flag) 
        llvm::errs() << "(";

      llvm::errs() << factor;

      if(bracket_flag) 
        llvm::errs() << ",";
      else 
        llvm::errs() << ")";

      bracket_flag = !bracket_flag;
    }
    llvm::errs() << "\n";
  }

  llvm::errs() <<"//-------------------------------------//\n";
  return AllDesignSpace;
}


template <typename LoadOrStoreT, typename OpToWalkT>
static SmallVector<LoadOrStoreT,  4> GetAllHoistOpInThisForLevel(OpToWalkT op_to_walk){
  SmallVector<LoadOrStoreT,  4> ToHoistOps;
  op_to_walk.walk([&](LoadOrStoreT accessop)
  {
    Operation* ParentOp = accessop.getOperation()->getParentOp();
    if( ParentOp == op_to_walk.getOperation() // Only consider this level
     && ParentOp->getName().getStringRef() == AffineForOp::getOperationName() )
    { 
      AffineForOp ParentForOp = dyn_cast<AffineForOp>(*ParentOp);
      // Value memref;
      // SmallVector<Value, 4> IVs;
      for(mlir::Value index : accessop.getIndices()){
        if(index == ParentForOp.getInductionVar()){
          // This accessop op can't be hoisted because it is constrained by loop of its level. 
          return WalkResult::advance();
        }
      }
      // This accessop op can be hoisted because it is not constrained by IV of its parent fopOp. 
      ToHoistOps.push_back(accessop);
    }
    return WalkResult::advance();
  });
  
  return ToHoistOps;
}

/// Remove the `idx`-th yield value, iter_arg, and result from an AffineForOp.
static AffineForOp removeIdxthIterArgOfAffineForOp(AffineForOp& forOp, unsigned idx) {
  assert(idx < forOp.getNumIterOperands() &&
         "Index out of bounds for AffineForOp iter operands!");
  // reduce yield operand
  auto yieldOp = dyn_cast<AffineYieldOp>(forOp.getBody()->getTerminator());
  SmallVector<mlir::Value, 4> newYieldOperands;
  for (unsigned i = 0, e = yieldOp.getNumOperands(); i < e; ++i) {
    if (i != idx) {
      newYieldOperands.push_back(yieldOp.getOperand(i));
    }
  }
  yieldOp.getOperation()->setOperands(newYieldOperands);

  /// erase idx-th arg
  forOp.getBody()->eraseArgument(idx + 1);
 
  // Create a new loop before the existing one, with the reduced operands.
  IRRewriter rewriter(forOp.getContext());
  rewriter.setInsertionPoint(forOp.getOperation());
  SmallVector<mlir::Value, 4> newIterInits;
  for (unsigned i = 0, e = forOp.getNumIterOperands(); i < e; ++i) {
    if (i != idx) {
      newIterInits.push_back(forOp.getInits()[i]);
    }
  }
  AffineForOp newLoop = rewriter.create<AffineForOp>(
    forOp.getLoc(), forOp.getLowerBoundOperands(), forOp.getLowerBoundMap(),
    forOp.getUpperBoundOperands(), forOp.getUpperBoundMap(), forOp.getStep().getSExtValue(), newIterInits);
  
  // llvm::errs() <<" block: \n";
  // newLoop.getOperation()->getBlock()->dump();
  /// move original block to new for block, to erase idx-th result
  mlir::Block* originBlock = forOp.getBody();
  mlir::Block* newBlock = newLoop.getBody();
  originBlock->moveBefore(newBlock);
  newBlock->erase();

  // /// print every op and their oprand
  // newLoop.walk([&](Operation* op){
  //   llvm::errs() << "-------op------ \n";
  //   op->dump();
  //   for(auto operand: op->getOperands()){
  //     llvm::errs() << "    -----oprand---- \n";
  //     operand.dump();
  //     llvm::errs() << "       ---parent--- \n";
  //     operand.getParentBlock()->getParentOp()->dump();
  //   }
  // });
  // forOp.walk([&](Operation* op){
  //   llvm::errs() << "-------op------ \n";
  //   op->dump();
  //   for(auto operand: op->getOperands()){
  //     llvm::errs() << "    -----oprand---- \n";
  //     operand.dump();
  //     llvm::errs() << "       ---parent--- \n";
  //     operand.getParentBlock()->getParentOp()->dump();
  //   }
  // });

  /// Replace iterargs
  // for(int i = 0; i < forOp.getNumIterOperands(); i++){
  //   if(i < idx)
  //     forOp.getRegionIterArgs()[i].replaceAllUsesWith(newLoop.getRegionIterArgs()[i]);
  //   else if(i > idx)
  //     forOp.getRegionIterArgs()[i].replaceAllUsesWith(newLoop.getRegionIterArgs()[i-1]);
  // }
  // forOp.getInductionVar().replaceAllUsesWith(newLoop.getInductionVar());

  /// erase original for op
  forOp.erase();

  // llvm::errs() <<" block: \n";
  // newLoop.getOperation()->getBlock()->dump();
  return newLoop;
}

/// @brief Move a pair of load store op in the original for op, eliminate loop-carried iteration variables
/// @param storeop 
/// @param loadop 
/// @param storeop 
/// @return success or not
static AffineForOp MoveLoadStorePairIn(AffineForOp forop, AffineLoadOp loadop, AffineStoreOp storeop){ 
  assert(getPositionRelationship(loadop.getOperation(), storeop.getOperation()) == PositionRelationInLoop::SameLevel);
  unsigned idx;
  for(idx = 0; idx < forop.getNumIterOperands(); idx ++){
    mlir::Value operand = forop.getInits()[idx];
    if(operand == loadop.getResult())
      break;
  }
  assert(idx < forop.getNumIterOperands());
  assert(forop.getResult(idx) == storeop.getOperand(storeop.getStoredValOperandIndex()));

  /// Move load in the front, and remove iterarg
  // loadop.getOperation()->moveBefore(forop.getBody()->begin());
  /// set the use of idx-th iterarg
  SmallVector<mlir::Operation*> IterArgConsumers = getAllUsesInBlock(forop.getRegionIterArgs()[idx], forop.getBody());
  for(mlir::Operation* Consumer : IterArgConsumers){
    unsigned i;
    for(i = 0; i < Consumer->getNumOperands(); i++){
      if(Consumer->getOperand(i) == forop.getRegionIterArgs()[idx]){
        Consumer->setOperand(i, loadop.getResult());
      }
    }
  }


  // forop.getOperation()->eraseOperand(idx);
  // forop.getBody()->eraseArgument(idx+1); /// erase the iterarg of block argument
  // forop.dump();

  /// Move store in the back
  AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
  storeop.getOperation()->remove();

  forop.getBody()->push_back(storeop);
  storeop.getOperation()->moveBefore(yieldop);
  storeop.setOperand(storeop.getStoredValOperandIndex(), yieldop.getOperand(idx));

  /// remove idx-th iterarg, yield, and result
  AffineForOp newLoop = removeIdxthIterArgOfAffineForOp(forop, idx);

  /// move load op in
  loadop.getOperation()->remove();
  newLoop.getBody()->push_front(loadop.getOperation());

  // newLoop.dump();
  return newLoop;
}


/************
 * A wrapper of official unroll function to do some pre-optimization:
 * 1. hoist load store pair.
*/
LogicalResult mlir::ADORA::loopUnrollByFactor_opt(
    AffineForOp forOp, uint64_t unrollFactor,
    // function_ref<void(unsigned, Operation *, OpBuilder)> annotateFn,
    bool cleanUpUnroll)
{
  ///TODO: Some problem exists here.
  SmallVector<AffineLoadOp,  4> ToHoistLoads;
  SmallVector<AffineStoreOp, 4> ToHoistStores;
  /////////
  /// Step 1 : Get all loads op to be hoisted
  /////////
  ToHoistLoads = GetAllHoistOpInThisForLevel<AffineLoadOp>(forOp);
  /////////
  /// Step 2 : Get all stores op to be hoisted
  /////////
  ToHoistStores = GetAllHoistOpInThisForLevel<AffineStoreOp>(forOp);

  /////////
  /// Step 3 : Do hoists for load-store pairs
  /////////
  SmallVector<AffineLoadOp,  4> ToHoistLoads_copy = ToHoistLoads;
  SmallVector<AffineStoreOp, 4> ToHoistStores_copy = ToHoistStores;
  SmallVector<std::pair<AffineLoadOp, AffineStoreOp>, 4> HoistedPairs;
  // forOp.dump();
  std::optional<AffineForOp> FixedForOp = forOp; 
  for(AffineLoadOp loadop : ToHoistLoads_copy){
    /// Check whether this load occurs with a corresponding store which have
    /// the same memref and address to access. 
    /// If so, this load-store pair 
    /// should be hoisted while construct a loop-carried variable.
    for(AffineStoreOp storeop : ToHoistStores_copy){
      llvm::errs() << "[info] loadop: " << loadop << "\n";   
      llvm::errs() << "[info] storeop: " << storeop << "\n";    
      if(LoadStoreSameMemAddr(loadop, storeop)){
        FixedForOp = MoveLoadStorePairOut(loadop, storeop);
        if(FixedForOp.has_value()){
          FixedForOp.value().dump();
          // Remove hoisted load/store ops from to-check vector 
          AffineLoadOp* it_ld = std::find(ToHoistLoads.begin(), ToHoistLoads.end(), loadop);
          assert(it_ld != ToHoistLoads.end());
          ToHoistLoads.erase(it_ld);
          AffineStoreOp* it_st = std::find(ToHoistStores.begin(), ToHoistStores.end(), storeop);
          assert(it_st != ToHoistStores.end());
          ToHoistStores.erase(it_st);
          HoistedPairs.push_back(std::pair(loadop, storeop));
          break;
        }
      }
    }
  }
  // if()
  FixedForOp.value().dump();
  
  /////////
  /// Step 4 : Do hoists for remaining load/store 
  /////////
  for(AffineLoadOp loadop : ToHoistLoads){
    /// If the loadop has no corresponding store then just hoist.
    Operation* ParentOp = loadop.getOperation()->getParentOp();
    if(isa<ADORA::KernelOp>(ParentOp))
      continue;

    assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
    loadop.getOperation()->moveBefore(ParentOp);
  }    
  for(AffineStoreOp storeop : ToHoistStores){
    Operation* ParentOp = storeop.getOperation()->getParentOp();
    if(isa<ADORA::KernelOp>(ParentOp))
      continue;

    assert(ParentOp->getName().getStringRef() == AffineForOp::getOperationName());
    storeop.getOperation()->moveAfter(ParentOp);
  }

  if(!FixedForOp.has_value()){
    FixedForOp = forOp;
  }

  if(unrollFactor < getConstantTripCount(FixedForOp.value()).value_or(0)){
    LogicalResult UnrollResult = loopUnrollByFactor(FixedForOp.value(), unrollFactor, /*annotateFn=*/nullptr, cleanUpUnroll);
    /// Put those op back for those which are move out as before.
    FixedForOp.value().dump();

    for(auto HoistedPair: HoistedPairs){
      HoistedPair.first.getOperation()->getBlock()->dump();
      HoistedPair.first.dump();
      HoistedPair.second.getOperation()->getBlock()->dump();
      HoistedPair.second.dump();
      FixedForOp = MoveLoadStorePairIn(FixedForOp.value(), HoistedPair.first, HoistedPair.second);
      assert(FixedForOp.has_value());
    }
    // FixedForOp.value().dump();
    return UnrollResult;
  }
  else {
    // FixedForOp.value().dump();
    LogicalResult UnrollResult = loopUnrollByFactor(FixedForOp.value(), unrollFactor, /*annotateFn=*/nullptr, cleanUpUnroll);
    return UnrollResult;
  }
}

//////////
// A wrapper for KernelUnroll
/////////
LogicalResult ADORAAffineLoopUnrollPass::
      KernelUnrollWithResourceLimits(ADORA::KernelOp kernel, mlir::ModuleOp& m){
  SmallVector<ADORA::ForNode> ForNodes = createAffineForTreeInsideKernel(kernel);
  SmallVector<SmallVector<unsigned>> UnrollDesignSpace;
  /** construct design space **/
  for (ADORA::ForNode& Node : ForNodes) {
    /* unrolling if innermost */
    // if (Node.IsInnermost()){
    FindUnrollingFactors(Node);
    // }
  }
  assert(ForNodes.end()-1 == ForNodes[0].getOutermostFor());
  
  // construct design space through all possible unroll factors
  UnrollDesignSpace = ConstructUnrollSpace(ForNodes);
  // ADORA::ForNode node = ForNodes[0].getOutermostFor();
  
  // make a new dir
  SmallVector<std::string> DesignSpaceFiles;
  std::filesystem::path currentPath = std::filesystem::current_path();
  std::string folderName = "DesignSpace";
  std::filesystem::path DesignSpacefolderPath = currentPath / folderName;
  std::filesystem::create_directory(DesignSpacefolderPath);
  /** Traverse the whole design space **/
  int max_ALU = 0, max_LSU = 0;
  std::string final_FilePath="";
  for(UnrollDesignPoint point : UnrollDesignSpace){
    ModuleOp topmodule = cast<ModuleOp>(m.getOperation()->clone());
    // ADORA::KernelOp kernelur = getSingleKernelFromFunc(*(topmodule.getOps<func::FuncOp>().begin()));
    ADORA::KernelOp kernelur = getKernelFromCopiedModule(topmodule, kernel);
  
    SmallVector<ADORA::ForNode> ForNodesUR = createAffineForTreeInsideKernel(kernelur);
    std::string fileName;
    if(kernelur.hasKernelName())
      fileName = kernelur.getKernelName();
    else
      fileName = "Unroll";
    for(unsigned node_cnt = 0; node_cnt < ForNodesUR.size(); node_cnt++){
    /// update filename 
      fileName += "_" + std::to_string(point[node_cnt]);
      // llvm::errs() <<"filename:" << fileName << "\n";
      /// Skip the tiling factor
      
      /// Apply unrolling by this factor
      unsigned unrollfactor = point[node_cnt];
      ADORA::ForNode NodeToUnroll = ForNodesUR[node_cnt];
      if(unrollfactor > 1){
        if(failed(loopUnrollByFactor_opt(NodeToUnroll.getForOp(), 
              unrollfactor, /*annotateFn=nullptr,*/ /*cleanUpUnroll=*/false))){
          llvm::errs() << "Unroll failed!!!\n";
          signalPassFailure();
          return LogicalResult::failure();
        }
      }
      // llvm::errs() << "[debug] new for:\n";
      // NodeToUnroll.dumpForOp();
    }
    /// create a new mlir file
    std::string filePath = DesignSpacefolderPath.string() + "/" + fileName + ".mlir";
    // llvm::errs() << "filePath: " << filePath << "\n";
    std::error_code ec;
    llvm::raw_fd_ostream outputFile(filePath, ec, sys::fs::FA_Write);
    if (ec) {
      llvm::errs() << "Error opening file: " << ec.message() << filePath << "\n";
      signalPassFailure();
      return LogicalResult::failure();
    }
    DesignSpaceFiles.push_back(fileName);
  
    // llvm::errs() << "[Info] design point " << fileName << ":\n";
    topmodule.dump();
  
    simplifyConstantAffineApplyOpsInRegion(kernelur.getBody());
    simplifyLoadAndStoreOpsInRegion(kernelur.getBody());
    
    topmodule.print(outputFile);
    topmodule.dump();      
  
    /// Generating DFG
    std::string GeneralOpNameFile_str;
    if (GeneralOpNameFile == nullptr) {
      std::cerr << "Environment variable \"GeneralOpNameFile\" is not set." << std::endl;
      GeneralOpNameFile_str = "lib/DFG/Documents/GeneralOpName.txt";
      std::cerr << "Using fallback: \"" << GeneralOpNameFile_str << "\"" << std::endl;
    }
    else
      GeneralOpNameFile_str = GeneralOpNameFile;
    LLVMCDFG *CDFG = new LLVMCDFG(fileName, GeneralOpNameFile_str);
    generateCDFGfromKernel(CDFG, kernelur, /*verbose=*/true);
    CDFG->CDFGtoDOT(DesignSpacefolderPath.string() + "/" + CDFG->name_str()+"_CDFG_unroll.dot");
  
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
    assert(0);
  }
  
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(file), SMLoc());
  mlir::OwningOpRef<mlir::ModuleOp> final_m = parseSourceFile<ModuleOp>(sourceMgr, m.getContext()); 
  mlir::ModuleOp moduleop = final_m.get();
  SymbolTable symbolTable(moduleop.getOperation());
  
  // moduleop.dump();
  // getOperation().dump();
  // m.replace
  // for(int index = 0; index < m.getOps<func::FuncOp>().size(); index++){
  //   func::FuncOp oldfunc = *(m.getOps<func::FuncOp>()[index]);
  //   func::FuncOp newfunc = *(m.getOps<func::FuncOp>()[index]);
  //   newfunc.getOperation()->moveBefore(oldfunc);
  //   oldfunc.getOperation()->erase();
  // }
  func::FuncOp oldfunc = *(m.getOps<func::FuncOp>().begin());
  func::FuncOp newfunc = *(moduleop.getOps<func::FuncOp>().begin());
  newfunc.getOperation()->moveBefore(oldfunc);
  oldfunc.getOperation()->erase();

  return LogicalResult::success();
}

/// @brief 
void ADORAAffineLoopUnrollPass::runOnOperation(){
  ////////////
  /// get hardware info
  ///////////
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

  ModuleOp topmodule = getOperation();
  // MLIRContext* context = topmodule.getContext();
  // auto originmodule = topmodule.getOperation()->clone();

  // int func_cnt = 0;
  // for (auto _ : topmodule.getOps<func::FuncOp>()) {
    // func_cnt++;
  // }
  // assert(func_cnt == 1 && "A kernel module can only contain 1 function.");

  // func::FuncOp topfunc = *(topmodule.getOps<func::FuncOp>().begin());
  SmallVector<ADORA::KernelOp> kernels;
  topmodule.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });

  for(ADORA::KernelOp kernel: kernels){
    KernelUnrollWithResourceLimits(kernel, topmodule);
  }
}


std::unique_ptr<OperationPass<ModuleOp>> 
        mlir::ADORA::createADORAAffineLoopUnrollPass() {
  return std::make_unique<ADORAAffineLoopUnrollPass>();
}

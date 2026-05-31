//===- AdjustToCachesize.cpp - Adjust(Partition) Kernel according to a customized Cachesize -----------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Parser/Parser.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Transforms/DialectConversion.h"

#include <iostream>
// #include <fstream>
// #include <filesystem>
#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/CommandLine.h"
// #include "llvm/Support/raw_ostream.h"
// #include "llvm/Support/FileSystem.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Analysis/DependencyAnalysis.h"
#include "../PassDetail.h"

using namespace llvm; // for llvm.errs()
using namespace llvm::detail;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;


#define IDEAL_DMA_REQ_LEN 100
#define GetOutermostLoopOfKernel(k) *(k.getOps<mlir::affine::AffineForOp>().begin())
//===----------------------------------------------------------------------===//
// AdjustKernelMemoryFootprint to meet cachesize
//===----------------------------------------------------------------------===//

#define PASS_NAME "adora-adjust-kernel-mem-footprint"

#define DEBUG_TYPE "adora-adjust-kernel-mem-footprint"

namespace
{
  
  // A pass that traverses Kernels in the Module and
  // Adjust or Partition Kernel according to a customized Cachesize
  struct AdjustMemoryFootprintPass : public AdjustKernelMemoryFootprintBase<AdjustMemoryFootprintPass>
  {
  public:
    void runOnOperation() override;
    uint64_t excessFactor_toCachesize(ADORA::KernelOp &Kernel);
    ADORA::KernelOp check_AllKernelMemoryFootprint(func::FuncOp topFunc, unsigned &Part_Factor);
    void outloop_partition(func::FuncOp Func, ADORA::KernelOp &Kernel, unsigned Part_Factor);
    std::optional<int64_t> getSingleMemrefFootprintBytes(AffineForOp forOp);  
    void simplifyAffileLoopLevel(func::FuncOp topFunc);

    int ExplicitKernelDataBLockLoadStore(ADORA::KernelOp Kernel);
    int EliminateOuterLoopAffineTrans(ADORA::KernelOp Kernel);

    enum AlignedBitsType { BITS_32 = 32, BITS_64 = 64, BITS_128 = 128, BITS_256 = 256 };
    SmallVector<int64_t> FillMemRefShape \
      (ArrayRef<int64_t> ToFillShape, MemRefType TargetShape, AlignedBitsType AlignedBits=AlignedBitsType::BITS_64);
  
    /// Utilities
    mlir::affine::AffineForOp constructOuterLoopNest(mlir::affine::AffineForOp &OriginforOp);
  };
} // namespace

/// @brief Find the max space accessed of a single memref load/store in affine for loop
/// @param forOp Only walk within this for-looop level, and outer level will not be considered
/// @return the access count(not footprint) of a single memref load/store * databits  
std::optional<int64_t> mlir::ADORA::getSingleMemrefAccessSpace(AffineForOp forOp){
  int64_t MaxSpace = 0;
  forOp.walk([&](mlir::Operation *op){
    if(isa<AffineLoadOp>(op) || isa<AffineStoreOp>(op)){
      unsigned BitWidth;
      if(isa<AffineLoadOp>(op))
        BitWidth = dyn_cast<AffineLoadOp>(op).getMemref().getType().getElementTypeBitWidth();
      else
        BitWidth = dyn_cast<AffineStoreOp>(op).getMemref().getType().getElementTypeBitWidth();

      assert(BitWidth % 8 == 0);

      int64_t TotalTC = 1;
      mlir::Operation* parent = op->getParentOp();
      assert(isa<AffineForOp>(parent) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
      AffineForOp parentFor = dyn_cast<AffineForOp>(*parent);
      // parentFor.dump();
      // forOp.dump();
      while(parentFor != forOp){
        TotalTC *= getConstantTripCount(parentFor).value_or(0);
        parent = parent->getParentOp();
        assert(isa<AffineForOp>(parent) && "Nest ForOp 's parent op should be AffineForOp!");
        parentFor = dyn_cast<AffineForOp>(*parent);
      }
      TotalTC *= getConstantTripCount(parentFor).value_or(0);

      assert(TotalTC != 0 && "Trip count should not be zero.");
      MaxSpace = std::max(MaxSpace, TotalTC * BitWidth / 8);
    }
  });
  assert(MaxSpace != 0 && "Trip count should not be zero.");
  return MaxSpace;
}

template <typename lsT>
static void replaceMemrefOfAffineLSop(lsT& op, mlir::Value& newmemref){
  mlir::Value oldmemref = op.getMemref();
  LLVM_DEBUG(oldmemref.getType().dump());  

  op.getOperation()->replaceUsesOfWith(oldmemref, newmemref);

  if(oldmemref.getType().isa<MemRefType>()){
    if(dyn_cast<MemRefType>(oldmemref.getType()).getShape().size() == 0){
      OpBuilder b(op.getOperation());
      AffineExpr expr = b.getAffineConstantExpr(0);
      expr.dump();
      AffineMap Map = AffineMap::get(0, 0, expr);
      op.setMap(Map);
    }
  }

  LLVM_DEBUG(op.dump());  
}

std::optional<int64_t> AdjustMemoryFootprintPass::
    getSingleMemrefFootprintBytes(AffineForOp forOp){
  auto *forInst = forOp.getOperation();
  Block &block = *forInst->getBlock();
  Block::iterator start = Block::iterator(forInst);
  Block::iterator end = std::next(Block::iterator(forInst));
  SmallDenseMap<mlir::Value, std::unique_ptr<MemRefRegion>, 4> regions;
  // Walk this 'affine.for' operation to gather all memory regions.
  auto result = block.walk(start, end, [&](Operation *opInst) -> WalkResult {
    if (!isa<AffineReadOpInterface, AffineWriteOpInterface>(opInst)) {
      // Neither load nor a store op.
      return WalkResult::advance();
    }

    // Compute the memref region symbolic in any IVs enclosing this block.
    auto region = std::make_unique<MemRefRegion>(opInst->getLoc());

    // opInst->dump();
    // region->compute(opInst,
    //                         /*loopDepth=*/getNestingDepth(forOp));
    // region->cst.dump();
    // region->memref.dump();
    // std::optional<int64_t> size = region->getRegionSize();
    // std::cout << "size:" << *size << std::endl;

    if (failed(
            region->compute(opInst,
                            /*loopDepth=*/getNestingDepth(&*block.begin())))) {
      return opInst->emitError("error obtaining memory region\n");
    }

    auto it = regions.find(region->memref);

    if (it == regions.end()) {
      regions[region->memref] = std::move(region);
    } else if (succeeded(it->second->unionBoundingBox(*region))) {
      it->second->cst.dump();
    }
    else {
      return opInst->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
    }

        // it->second->cst.dump();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return std::optional<int64_t>();

  int64_t MaxSizeInBytes = 0;
  for (const auto & region : regions) {
    // region.second->cst.dump();
    // std::optional<int64_t> size = region.second->getRegionSize();
    // errs() << " [debug] size:" << size.value() << "\n";

    mlir::SmallVector<int64_t> shape;
    region.second->getConstantBoundingSizeAndShape(&shape);

    shape = 
      FillMemRefShape(/*ToFillShape=*/(ArrayRef<int64_t>)shape, /*TargetMemref=*/ region.first.getType().cast<MemRefType>()/*,Aligned=BITS_64*/ );
 
    /// calculate size from region shape
    std::optional<int64_t> size = getMemRefIntOrFloatEltSizeInBytes(region.first.getType().cast<MemRefType>());
    if (!size.has_value())
      return std::optional<int64_t>();

    for(int i = 0; i < shape.size(); i++){
      assert(shape[i] > 0);
      *size *= shape[i];
    }
    
    MaxSizeInBytes = std::max(size.value(), MaxSizeInBytes);
  }
  return MaxSizeInBytes;
}

/// @brief traverse all lopps and fuse 2 levels if possible
/// @param topFunc 
void AdjustMemoryFootprintPass::simplifyAffileLoopLevel(func::FuncOp topFunc){
  // errs()<<"top func before simplifying:\n"; topFunc.dump();
  topFunc.walk([&](mlir::Operation *op)
  {
    // errs()<<"  op :"; op->dump();
    if(op->getName().getStringRef()== ADORA::KernelOp::getOperationName()){
      return WalkResult::advance();
    }
    else if(op->getName().getStringRef()== mlir::affine::AffineForOp::getOperationName()){
      // errs()<<"  op :"; op->dump();
      mlir::affine::AffineForOp for_outer = dyn_cast<mlir::affine::AffineForOp>(op);

      if(for_outer.getBody()->getOperations().size() == 2 // 1 affineForOp and 1 affineYieldOp 
           && for_outer.getBody()->front().getName().getStringRef() == mlir::affine::AffineForOp::getOperationName()){ 
              //1st .front() get 1st block and 2nt .front() get 1st operation
        mlir::affine::AffineForOp for_inner = dyn_cast<mlir::affine::AffineForOp>(for_outer.getBody()->front());

        if(for_inner.getLowerBoundOperands().size() == 1 && 
           for_inner.getUpperBoundOperands().size() == 1 &&
           *(for_inner.getLowerBoundOperands().begin()) == for_outer.getInductionVar() &&
           *(for_inner.getUpperBoundOperands().begin()) == for_outer.getInductionVar()){
          /// This 2 levels' affine.for can be simplified to 1
          AffineExpr Expr;
          AffineMap Map;
          int64_t step;
          ///////////////
          /// construct an identical loop with simplified level
          ///////////////

          /// Step 1: construct another loop to forOp
          // Loop bounds will be set later.
          Operation* new_op = for_inner.clone();
          op->getBlock()->push_back(new_op);
          new_op->moveAfter(for_outer);
          AffineForOp simpleLoop = dyn_cast<AffineForOp>(*new_op);
          // llvm::errs() << "[debug] Step 1:\n";Kernel.dump();

          /// Step 2: Change simpleLoop's lower/upper bound
          OpBuilder b(for_inner.getOperation());
          if (for_outer.hasConstantLowerBound()) ///lower bound
          {
            Expr = b.getAffineConstantExpr(for_outer.getConstantLowerBound());
            Map = AffineMap::get(0, 0, Expr);
            simpleLoop.setLowerBound({}, Map);
          }
          else
          {
            Expr = for_outer.getLowerBoundMap().getResult(0);
            // Expr = getConstPartofAffineExpr(for_outer.get)
            // Expr = b.getAffineDimExpr(0);
            Map = AffineMap::get(1, 0, Expr);
            simpleLoop.setLowerBound(for_outer.getLowerBoundOperands(), Map);
          }
          // errs()<<"  for_outer :"; for_outer.dump();
          if (for_outer.hasConstantUpperBound()) ///upper bound 
          {
            Expr = b.getAffineConstantExpr(for_outer.getConstantUpperBound());
            Map = AffineMap::get(0, 0, Expr);
            simpleLoop.setUpperBound({}, Map);
          }
          else
          {
            Expr = for_outer.getUpperBoundMap().getResult(0);
            Map = AffineMap::get(1, 0, Expr);
            simpleLoop.setUpperBound(for_outer.getUpperBoundOperands(), Map);
          }
          // errs()<<"  Expr for upper bound :"<< Expr << "\n";       
          /// Step 3: Set new step for simplified loop
          assert(for_outer.getStep().getSExtValue() % for_inner.getStep().getSExtValue() == 0 && "Step of inner loop should be a divisor of step of outer!");
          step = for_inner.getStep().getSExtValue();
          simpleLoop.setStep(step);
          
          for_outer.erase();
          // errs()<<"  simpleLoop :\n"; simpleLoop.dump();         
        }
      }
    }
    return WalkResult::advance(); 
  });  
  
  return;
}



/// @brief check whether a ADORA.kernel meets the limits of cachesize
/// @param Kernel
/// @return a excessFactor that how many
///           times larger memory footprint is compared to cacheSize
uint64_t AdjustMemoryFootprintPass::
    excessFactor_toCachesize(ADORA::KernelOp &Kernel)
{
  unsigned Toploop_cnt = 0;
  uint64_t excessFactor;
  
  /// get how many times larger memory footprint is compared to cacheSize
  uint64_t cacheSizeBytes = Cachesize_Kib * 1024;
  uint64_t singleArraySizeBytes;

  if(SingleArray_Size == 0){
    singleArraySizeBytes = cacheSizeBytes;
  }
  else{
    singleArraySizeBytes = SingleArray_Size * 1024;
  }
  // errs()<<" Kernel :" << Kernel << "\n";
  for (mlir::affine::AffineForOp forOp : Kernel.getOps<mlir::affine::AffineForOp>())
  {
    // errs()<<"[Info] Found a forOP:\n"; forOp.dump();
    if(AffineAccessPattern == true){
      /***********
       * If memory access is in affine pattern, the scratchpad space an array occupied
       * equals to the trip-count.
      ***********/
      std::optional<int64_t> maxSpace_singleMem = getSingleMemrefAccessSpace(forOp);
      excessFactor = llvm::divideCeil(*maxSpace_singleMem, cacheSizeBytes);
    }
    else{
      /// get memory footprint of this kernel
      LLVM_DEBUG(errs()<<" fp_singleMem :" << forOp << "\n");
      std::optional<int64_t> fp_totalMem = getMemoryFootprintBytes(forOp);
      std::optional<int64_t> fp_singleMem = AdjustMemoryFootprintPass::getSingleMemrefFootprintBytes(forOp);
      LLVM_DEBUG(errs()<<" fp_totalMem :" << fp_totalMem << "\n");
      LLVM_DEBUG(errs()<<" fp_singleMem :" << fp_singleMem << "\n");


      
      excessFactor = std::max(llvm::divideCeil(*fp_totalMem, cacheSizeBytes)
                              ,llvm::divideCeil(*fp_singleMem, singleArraySizeBytes));
    }

    Toploop_cnt++;
  }
  assert(Toploop_cnt == 1 && "Kernel have only 1 top-loop!");
  return excessFactor;
}

/// @brief traverse all kernels and check whether it meets the limits of cachesize
/// @param Func
/// @return return a kernel whose mem footprint is larger than cachesize
ADORA::KernelOp AdjustMemoryFootprintPass::
    check_AllKernelMemoryFootprint(func::FuncOp topFunc, unsigned &Part_Factor)
{
  // for (auto Kernel : topFunc.getOps<ADORA::KernelOp>()) {
  ADORA::KernelOp Kernel = NULL;
  topFunc.walk([&](mlir::Operation *op)
  {
    // errs()<<"op :" << op->getName().getStringRef() << "\n";
    if(op->getName().getStringRef()== ADORA::KernelOp::getOperationName()){
      Kernel = dyn_cast<ADORA::KernelOp>(op);
      uint64_t excessFactor = excessFactor_toCachesize(Kernel);
      Part_Factor = 1;
      if(excessFactor > 1){
        Part_Factor = excessFactor;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance(); });

  return Kernel;
}

void AdjustMemoryFootprintPass::
    outloop_partition(func::FuncOp topFunc, ADORA::KernelOp &Kernel, unsigned Part_Factor)
{
  assert(Part_Factor >= 1 && "Part_Factor is a integer larger than 1!");
  /// Tofix:
  /// Partition a loop through its outest loop may not be right
  // assert(Kernel.getOps<mlir::affine::AffineForOp>().size() == 1 && "Part_Factor is a integer larger than 1!");
  mlir::affine::AffineForOp forOp = *(Kernel.getOps<mlir::affine::AffineForOp>().begin());
  std::string kn_name = Kernel.getKernelName();

  // errs() << "forop: \n"<< forOp << "\n";
  int64_t largestDiv = getLargestDivisorOfTripCount(forOp);
  std::optional<uint64_t> mayBeConstantCount = getConstantTripCount(forOp);

  mlir::affine::AffineForOp OutforOp;
  OpBuilder b(forOp.getOperation());

  if(DisableRemainderBlock && largestDiv % Part_Factor != 0)
  {
    while(largestDiv % Part_Factor != 0 && Part_Factor < largestDiv)
    {
      Part_Factor++;
    }
  }

  if (mayBeConstantCount && Part_Factor >= mayBeConstantCount.value())
  {
    /// tripcount of outloop is less than part factor:
    /// move ADORA.kernel{...} to inner loop
    // assert(forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound() &&
    //       "We don't support Variable LowerBound now!");/// Todo : Variable LoopBound
    ///////////////
    /// adjust "ADORA.kernel" location
    ///////////////

    /// Step 1: erase old KernelOp and get corresponding op of kernel after erasing
    mlir::Operation *new_OutforOp = ADORA::eraseKernel(topFunc, Kernel);
    // llvm::errs() << "[debug] After erase:\n";topFunc.dump();

    /// Step 2: get forOp from inner for-loop of new_OutforOp
    OutforOp = dyn_cast<AffineForOp>(*new_OutforOp);

    std::vector<AffineForOp> AffineForOpVec;
    for (AffineForOp innerforOp : OutforOp.getOps<AffineForOp>())
    {
      AffineForOpVec.push_back(innerforOp);
    }

    /// Step 3: create new KernelOp
    for (AffineForOp innerforOp : AffineForOpVec)
    {
      (void)ADORA::SpecifiedAffineFortoKernel(innerforOp, kn_name);
    }
  }
  else {
    //// How to handle Irregular Access Iteration Space, e.g.
    ///     affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1)
    /// or  affine.for %arg3 = 0 to affine_map<(d0) -> (1800 - d0)>(%arg2) 
    if(!mayBeConstantCount){
      ///////
      // Verify the lower and upper bound of for op is like 0 to arg1 or 0 to 1800-arg1
      ///////
      if(!IsIterationSpaceSupported(forOp)){
        forOp.dump();
        assert(false && "Can't handle this kind of irregular iteration space for above for op");
      }

      
    }

    AffineExpr Expr;
    AffineMap Map;
    int64_t step;

    // largestDiv = 2000;

    if (largestDiv % Part_Factor != 0)
    {
      /// Need to handle isolated loop
      // Intra-tile loop ii goes from i to min(i + tileSize * stepSize, ub_i).
      // Construct the upper bound map; the operands are the original operands
      // with 'i' (tile-space loop) appended to it. Isolation bound is needed to get.

      ///////////////
      /// construct an identical loop and get isolation bound
      ///////////////

      /// Step 1: construct an identical loop to forOp
      // Loop bounds will be set later.
      Operation* new_op = forOp.clone();
      Kernel.getBody().begin()->push_back(new_op);
      new_op->moveAfter(forOp);
      AffineForOp IslatedLoop = dyn_cast<AffineForOp>(*new_op);
      // llvm::errs() << "[debug] Step 1:\n";Kernel.dump();

      /// Step 2: get isolation bound
      int64_t Isolation_count = largestDiv - largestDiv % Part_Factor;

      /// Step 3: Set lower bound of Isolation loop and 
      ///  Upper bound of original loop 
      ///  = Isolation_count * Step + lower bound;
      ///  other 2 bounds remains.
      step = forOp.getStep().getSExtValue();
      if (forOp.hasConstantLowerBound())
      {
        Expr = b.getAffineConstantExpr(forOp.getConstantLowerBound())+step*Isolation_count;
        Map = AffineMap::get(0, 0, Expr);
        IslatedLoop.setLowerBound({}, Map);
        forOp.setUpperBound({}, Map);
      }
      else
      {
        Expr = b.getAffineDimExpr(0)+step*Isolation_count;
        /// Tofix : "0" Here we assume that forOp.LowerBound() has only 1 Dim for affine
        Map = AffineMap::get(1, 0, Expr);
        IslatedLoop.setLowerBound(forOp.getLowerBoundOperands(), Map);
        forOp.setUpperBound(forOp.getLowerBoundOperands(), Map);
      }
      // llvm::errs() << "[debug] Step 3:\n";Kernel.dump();

      ///////////////
      /// Move position of "Kernel" Op
      ///////////////
      /// Step 4: create new KernelOp and erase old KernelOp
      (void)ADORA::SpecifiedAffineFortoKernel(forOp, kn_name);
      (void)ADORA::SpecifiedAffineFortoKernel(IslatedLoop, kn_name);
      new_op = ADORA::eraseKernel(topFunc, Kernel);
      Kernel = dyn_cast<ADORA::KernelOp>(*new_op);
      forOp = *(Kernel.getOps<mlir::affine::AffineForOp>().begin()); /// one Kernel should only have one top for
      largestDiv = Isolation_count;
      /// Following handling is general for both largestDiv % Part_Factor ==0 or !=0 
    }

    /// Following handling is general for both largestDiv % Part_Factor ==0 or !=0 
    /// largestDiv % Part_Factor == 0
    /// No need of the min expression.
    // llvm::errs() << "[debug] Before construct:\n";Kernel.dump();forOp.dump();
    OutforOp = constructOuterLoopNest(forOp);

    ///////////////
    /// adjust "new" top loop index
    ///////////////
    // assert(forOp.hasConstantLowerBound() && forOp.hasConstantUpperBound() &&
    //       "We don't support Variable LowerBound now!");/// Todo : Variable LoopBound

    /// Step 1: adjust lower bound to original loop lower bound
    // llvm::errs() << "[debug] Before Step 1:\n";Kernel.dump();
    if (forOp.hasConstantLowerBound())
    {
      Expr = b.getAffineConstantExpr(forOp.getConstantLowerBound());
      Map = AffineMap::get(0, 0, Expr);
      OutforOp.setLowerBound({}, Map);
    }
    else
    {
      Map = forOp.getLowerBoundMap();
      OutforOp.setLowerBound(forOp.getLowerBoundOperands(), Map);
    }
    // llvm::errs() << "[debug] Step 1:\n";Kernel.dump();

    /// Step 2: adjust upper bound to original loop upper bound (= lower_bound + strip_count)
    if (forOp.hasConstantUpperBound())
    {
      Expr = b.getAffineConstantExpr(forOp.getConstantUpperBound());
      Map = AffineMap::get(0, 0, Expr);
      OutforOp.setUpperBound({}, Map);
    }
    else
    {
      Map = forOp.getUpperBoundMap();
      OutforOp.setUpperBound(forOp.getLowerBoundOperands(), Map);
    }
    // llvm::errs() << "[debug] Step 2:\n";Kernel.dump();

    /// Step 3: adjust step = strip_count / Part_Factor
    step = llvm::divideCeil(largestDiv, Part_Factor);
    OutforOp.setStep(step);
    // llvm::errs() << "[debug] Step 3:\n";Kernel.dump();

    ///////////////
    /// adjust "original" loop index
    ///////////////

    /// Step 4: adjust lower bound to new outer loop lower index value
    Expr = b.getAffineDimExpr(0);
    Map = AffineMap::get(1, 0, Expr);
    // newIntraTileLoop.setLowerBound(lbOperands, lbMap);
    forOp.setLowerBound(OutforOp.getInductionVar(), Map);
    // llvm::errs() << "[debug] Step 4:\n";Kernel.dump();

    /// Step 5: adjust upper bound = lower_bound + step
    Map = AffineMap::get(1, 0, Expr + step);
    forOp.setUpperBound(OutforOp.getInductionVar(), Map);
    // llvm::errs() << "[debug] Step 5:\n";Kernel.dump();

    /// Note: Step of original forOp remains

    ///////////////
    /// adjust "ADORA.kernel" location
    ///////////////

    /// Step 6: erase old KernelOp and get corresponding op of kernel after erasing
    mlir::Operation *new_OutforOp = ADORA::eraseKernel(topFunc, Kernel);
    // llvm::errs() << "[debug] After erase:\n";FuncOp.dump();

    /// Step 7: get forOp from inner for-loop of new_OutforOp
    OutforOp = dyn_cast<AffineForOp>(*new_OutforOp);
    forOp = dyn_cast<AffineForOp>(OutforOp.getBody()->begin());

    /// Step 8: create new KernelOp
    (void)ADORA::SpecifiedAffineFortoKernel(forOp, kn_name);
    // llvm::errs() << "[debug] After create:\n";FuncOp.dump();
  }

  // forOp.erase();
  return;
}



/// @brief Generate Explicit Kernel Data BLock
///        Load/Store of the call of kernel.
/// @param Kernel
/// Steps to create a DataBlockLoad Op:
///   Step1: when find a LoadOp, we traverse every dim(rank) of its original memref,
///    Get InductionVar(IV) of this memrefRegion
///
///   Step2: Get Lower And Upper Bound in this rank and get the lpMap and upMap 
///    that determines the min size of space. The size should be constant.
///
///   Step3: Create DataBlockLoadOp and replace original Memref in loadOp.
///
///   Step4: Change index of loadop according to the new BlockLoadOp.
///
/// Steps to create a DataBlockStore Op similar to DataBlockLoad Op
///
int AdjustMemoryFootprintPass::ExplicitKernelDataBLockLoadStore(ADORA::KernelOp Kernel)
{
  /*****
   * A test for dependence analysis.
   */
  MemRefRegion memrefRegion(Kernel.getLoc());
  mlir::Value memref;
  SmallVector<mlir::Value, 4> IVs;
  LLVM_DEBUG(Kernel.dump());
  mlir::OpBuilder builder(Kernel.getBody().getContext());
  // LLVM_DEBUG(Kernel.dump());
  unsigned BlockLoadStoreOpId = 0;

  // Tofix:
  // If dependency within iteration
  // typedef SmallDenseMap<std::unique_ptr<MemRefRegion>, SmallVector<Value, 4>, 4>  RegionToLSop_t;
  // RegionToLSop_t RegionToLSMap;

  //////////
  /// Dependency analysis:
  ///  RAW and WAR are considered.
  //////////
  AffineForOp forop = GetOutermostLoopOfKernel(Kernel);
  SmallDenseMap<affine::AffineLoadOp, SmallVector<affine::AffineStoreOp>> WARs;
  SmallDenseMap<affine::AffineStoreOp, SmallVector<affine::AffineLoadOp>> RAWs;
  WARs = getAffineLoadToStoreDependency(forop);
  RAWs = getAffineStoreToLoadDependency(forop);
  SmallDenseMap<unsigned, SmallVector<Operation* > > ReuseGroups = getReuseGroupsForLoop(forop);
  SmallVector<Operation* > VisitedOperations;

  LLVM_DEBUG(
  for(auto elem: WARs){
    llvm::errs() << "WAR:\n";
    affine::AffineLoadOp load = elem.first;
    llvm::errs() << "load: ";
    load.dump();
    for(auto store : elem.second){
      llvm::errs() << "|-->store: ";
      store.dump();
    }
  }

  for(auto elem: RAWs){
    llvm::errs() << "RAW:\n";
    affine::AffineStoreOp store = elem.first;
    llvm::errs() << "store: ";
    store.dump();
    for(auto load : elem.second){
      llvm::errs() << "|-->load: ";
      load.dump();
    }
  }

  for(auto elem: ReuseGroups){
    llvm::errs() << "ReuseGroups " << elem.first << ":\n";
    for(auto op : elem.second){
      llvm::errs() << "|->op: ";
      op->dump();
    }
  }

  );

  //////////
  /// Handle RAWs first
  //////////
  for(auto ReuseGroupPair: ReuseGroups){
    SmallVector<Operation* > ReuseGroup = ReuseGroupPair.second;
    /// Get memrefRegion of the group
    MemRefRegion GroupMemrefRegion(Kernel.getLoc());
    for(Operation* operation : ReuseGroup){
      if(affine::AffineLoadOp loadop = dyn_cast<affine::AffineLoadOp>(operation)){
        if(succeeded(memrefRegion.compute(loadop, 
                /*loopDepth=*/getNestingDepth(Kernel.getOperation())))){
          //// TODO: verify this
          if(GroupMemrefRegion.memref != memrefRegion.memref) 
            GroupMemrefRegion = memrefRegion;
          else
            GroupMemrefRegion.unionBoundingBox(memrefRegion);
        }
      }
      else if(affine::AffineStoreOp storeop = dyn_cast<affine::AffineStoreOp>(operation)){
        if(succeeded(memrefRegion.compute(storeop, 
                /*loopDepth=*/getNestingDepth(Kernel.getOperation())))){
          //// TODO: verify this
          if(GroupMemrefRegion.memref != memrefRegion.memref) 
            GroupMemrefRegion = memrefRegion;
          else
            GroupMemrefRegion.unionBoundingBox(memrefRegion);
        }
      }
      VisitedOperations.push_back(operation);
    }


    /////////////////////////////////////////
    /// generate blockload and blockstore
    ////////////////////////////////////////

    memrefRegion = GroupMemrefRegion;
    memref = memrefRegion.memref; /// original memref Op of this loadOP
    MemRefType memRefType = memref.getType().cast<MemRefType>(); /// contains shape info of original memref
    unsigned rank = memRefType.getRank(); /// dim number of original memref
    assert(rank == memrefRegion.cst.getNumDimVars() && "inconsistent memref region");      
      
    /// To fix: memIVs should be a setVector??
    // SmallVector<Value, 4> memIVs; /// stores the Interation Variables out of kernel that determined load bound
    SmallVector<AffineExpr, 4> memExprs;
    SmallVector<int64_t, 4> newMemRefShape; /// stores the new dim shape of blockloadOp which will replace oringinal memref

    /////////////
    /// Step1: Get InductionVar(IV) of this memrefRegion
    /////////////

    memrefRegion.cst.getValues(memrefRegion.cst.getNumDimVars(),
          memrefRegion.cst.getNumDimAndSymbolVars(), &IVs);
    // assert(IVs.size() <= 1  /// To fix: if IVs.size() > 1 ?
    //       && " This kernel should only have 1 outer IV as input arguments.");

    // if(IVs.size()==1){
    //   AffineForOp iv = getForInductionVarOwner(IVs.front());
    //   memIVs.push_back(iv.getInductionVar());          
    // }

    /// For different dim of original memref
    // for(auto &iv : IVs){
    //   llvm::errs() << "iv:" << iv << "\n";
    // }

    for (unsigned r = 0; r < rank; r++) {
      AffineExpr lbExpr_minspace, ubExpr_minspace;

      /////////////
      /// Step2: Get Lower And Upper Bound in this rank and get the lpMap and upMap 
      /// that determines the min size of space. The size should be constant.
      /////////////
      AffineMap lbMap, ubMap;
      memrefRegion.getLowerAndUpperBound(r, lbMap, ubMap);
      assert(lbMap.getNumDims() == IVs.size() && ubMap.getNumDims() == IVs.size()\
              && " Num of bound's dim should be the same with num of IVs!");
      llvm::errs() << "[debug] lbMap: " << lbMap << " , ubMap: "<< ubMap << "\n";
      int64_t min_space = -1;

      for(AffineExpr lbExpr : lbMap.getResults()){ 
        for(AffineExpr ubExpr : ubMap.getResults()){
          AffineExpr diffExpr = ubExpr - lbExpr;
          diffExpr = simplifyAffineExpr(diffExpr, lbMap.getNumDims(), lbMap.getNumSymbols());
          LLVM_DEBUG(llvm::errs() << "[debug] diffExpr: " << diffExpr << "\n");
          LLVM_DEBUG(llvm::errs() << "[debug] lbExpr: " << lbExpr << ",ubExpr:" << ubExpr << "\n");
          if(diffExpr.isSymbolicOrConstant()){
            /// Found a Constant diff
            AffineConstantExpr diffExpr_const=diffExpr.dyn_cast<AffineConstantExpr>();
            if((memRefType.hasStaticShape() && diffExpr_const.getValue()==memRefType.getNumElements()) 
                && min_space==-1){
              /// This upper and lowerbound is constrained by 
              /// original memref's size and a smaller min_space
              /// is not found yet.
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
            else if(diffExpr_const.getValue() < min_space || min_space==-1){
              /// Found a smaller memory space, store the Affine Expr of lb and ub
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
            else if(diffExpr_const.getValue() == min_space && 
              !ubExpr.isSymbolicOrConstant() && !lbExpr.isSymbolicOrConstant()){
              /// This is to set the lb to the expression from
              /// the one which contains dim variables rather than a constant bound
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
          }
        }
      }
      assert(min_space != -1 && 
                  " The memory space this L/S op access has different size in different Iterations!");

      memExprs.push_back(lbExpr_minspace);
      newMemRefShape.push_back(min_space);
    }

    //// Fix the shape of newMemRefShape to burst a sequential DMA quest as possible
    SmallVector<int64_t> FilledNewMemRefShape = \
      FillMemRefShape(/*ToFillShape=*/(ArrayRef<int64_t>)newMemRefShape, /*TargetMemref=*/memRefType/*,Aligned=BITS_64*/ );
    // llvm::errs() << "FilledNewMemRefShape Shape: ";
    // for(int r = 0; r < FilledNewMemRefShape.size(); r++){
    //   llvm::errs() << FilledNewMemRefShape[r] << " ";
    // }
    // llvm::errs() << "\n";
    /////////
    /// Step 3 :Create DataBlockLoadOp and replace original Memref in loadOp 
    /////////
    // ArrayRef<int64_t> try_array = {1,2,3};
    // MemRefType tryMemRef = MemRefType::get(try_array, memRefType.getElementType());
    // llvm::errs() << "[debug] tryMemRef: ";tryMemRef.dump();      

    AffineMap memIVmap = AffineMap::get(IVs.size(), /*symbolCount=*/0, memExprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
    MemRefType newMemRef = MemRefType::get(FilledNewMemRefShape, memRefType.getElementType());
    llvm::errs() << "[debug] newMemRef: ";newMemRef.dump();

    ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                (Kernel.getLoc(), memref, memIVmap, IVs, newMemRef);
    // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
    Kernel.getOperation()->getBlock()->push_back(BlockLoad);
    BlockLoad.getOperation()->moveBefore(Kernel);
    BlockLoad.setKernelName(Kernel.getKernelName());
    BlockLoad.setId(std::to_string(BlockLoadStoreOpId));

    llvm::errs() << "[debug] Before replace Kernel: ";Kernel.dump();
    for(Operation* operation : ReuseGroup){
      LLVM_DEBUG(operation->dump());
      /////////
      /// Step 4 : Change index of loadop according to the new BlockLoadOp
      /////////
      if(affine::AffineLoadOp loadop = dyn_cast<affine::AffineLoadOp>(operation)){
        // replaceMemrefOfAffineLSop(loadop, BlockLoad.getResult());
        loadop.getOperation()->replaceUsesOfWith(memref, BlockLoad.getResult());
        LLVM_DEBUG(loadop.dump());
        assert(memExprs.size() == rank && "Each rank should have its own affine expr.");
        SmallVector<AffineExpr, 4> LoadExprs;
        AffineMap LoadMap = loadop.getAffineMapAttr().getValue();
        for (unsigned r = 0; r < rank; r++) {
          AffineExpr LoadExpr = LoadMap.getResult(r);
          llvm::errs() << "[debug] before remove LoadExpr: " << LoadExpr << "\n";
          // operand_range LoadOperands = loadop.getMapOperands();
          AffineExpr BlockLoadExpr = memExprs[r];
          llvm::errs() << "[debug] BlockLoadExpr: " << BlockLoadExpr << "\n";       
          // llvm::errs() << "[debug] BlockLoadExpr.getKind(): " << BlockLoadExpr.getKind() << "\n";    
          assert((BlockLoadExpr.getKind() == AffineExprKind::DimId ||     // like: d0
                BlockLoadExpr.getKind() == AffineExprKind::Add   ||     // like: d0 + 32
                BlockLoadExpr.getKind() == AffineExprKind::Mul   ||     // like: d0 * 32
                BlockLoadExpr.getKind() == AffineExprKind::Constant) && // like: 32
                "Only handle '%arg9' or '%arg9 + 32' like index for BlockLoadExpr right now.");
        
          if(BlockLoadExpr.getKind() == AffineExprKind::Constant){
            LoadExpr = LoadExpr - BlockLoadExpr; /// forward the head position of LoadOp by a const value
          }
          else { 
          /// If BlockLoadExpr is : "%arg9" or "%arg9 + 32", we need to judge whether loadOp's 
          /// index contains a reference of "%arg9" first. If so, LoadExpr = LoadExpr - BlockLoadExpr
          // check(LoadExpr )

            SmallDenseMap<mlir::Value, unsigned> BlkLoadOperands = getOperandInRank(BlockLoad, r);
            SmallDenseMap<mlir::Value, unsigned> LoadOperands = getOperandInRank(loadop, r);
            // assert(BlkLoadOperands.size() == 1 && "BlockLoadExpr could only contain one dim.");
            // DenseMapPair<mlir::Value, unsigned> BlkLoadOperand = *(BlkLoadOperands.begin());

            /// check whether the block Load Operand is contained by Load Operands
            // unsigned input;
            // llvm::errs() << "[debug] before simplified LoadExpr: " << LoadExpr << "\n";
            /// TODO: check this part
            for(DenseMapPair<mlir::Value, unsigned> BlkLoadOperand : BlkLoadOperands){
              for(DenseMapPair<mlir::Value, unsigned> LoadOperand : LoadOperands){
                if(LoadOperand.first == BlkLoadOperand.first){
                  assert(MultiplicatorOfDim(LoadExpr, LoadOperand.second) == 
                        MultiplicatorOfDim(BlockLoadExpr, BlkLoadOperand.second) &&
                      "Only if the multiplicator equals the dim can be eliminated.");
                  AffineExpr ConstantExpr = getAffineConstantExpr(
                      MultiplicatorOfDim(LoadExpr, LoadOperand.second), LoadExpr.getContext());
                  LoadExpr = LoadExpr -
                    getAffineDimExpr(LoadOperand.second, LoadExpr.getContext()) * ConstantExpr;
                  break;
                }
              }
              LoadExpr = simplifyAffineExpr(LoadExpr, LoadMap.getNumDims(), LoadMap.getNumSymbols());
              llvm::errs() << "[debug] after simplified LoadExpr: " << LoadExpr << "\n";
            }
            /// fixed: maybe in different dim 
          }
          LoadExprs.push_back(LoadExpr);
          llvm::errs() << "[debug] after remove LoadExpr: " << LoadExpr << "\n";
        }
        if(LoadExprs.size() == 0) LoadExprs.push_back(builder.getAffineConstantExpr(0));
        AffineMap newLoadMap = AffineMap::get(LoadMap.getNumDims(),LoadMap.getNumSymbols(),LoadExprs,builder.getContext()); 
        loadop.getOperation()->setAttr(AffineLoadOp::getMapAttrStrName(),AffineMapAttr::get(newLoadMap));
        llvm::errs() << "[debug] After replace Kernel: \n";Kernel.dump();
        // LoadToBlkLoad[loadop] = BlockLoad;
      }
      else if(affine::AffineStoreOp storeop = dyn_cast<affine::AffineStoreOp>(operation)){
        // replaceMemrefOfAffineLSop(storeop, BlockLoad.getResult());
        storeop.getOperation()->replaceUsesOfWith(memref, BlockLoad.getResult());
        assert(memExprs.size() == rank && "Each rank should have its own affine expr.");
        SmallVector<AffineExpr, 4> StoreExprs;
        AffineMap StoreMap = storeop.getAffineMapAttr().getValue();
        for (unsigned r = 0; r < rank; r++) {
          AffineExpr StoreExpr = StoreMap.getResult(r);
          llvm::errs() << "[debug] before remove StoreExpr: " << StoreExpr << "\n";
          // operand_range StoreOperands = storeop.getMapOperands();
          AffineExpr BlockLoadExpr = memExprs[r];
          llvm::errs() << "[debug] BlockLoadExpr: " << BlockLoadExpr << "\n";       
          // llvm::errs() << "[debug] BlockLoadExpr.getKind(): " << BlockLoadExpr.getKind() << "\n";    
          assert((BlockLoadExpr.getKind() == AffineExprKind::DimId ||     // like: d0
                BlockLoadExpr.getKind() == AffineExprKind::Add   ||     // like: d0 + 32
                BlockLoadExpr.getKind() == AffineExprKind::Mul   ||     // like: d0 * 32
                BlockLoadExpr.getKind() == AffineExprKind::Constant) && // like: 32
                "Only handle '%arg9' or '%arg9 + 32' like index for BlockLoadExpr right now.");
        
          if(BlockLoadExpr.getKind() == AffineExprKind::Constant){
            StoreExpr = StoreExpr - BlockLoadExpr; /// forward the head position of LoadOp by a const value
          }
          else { 
          /// If BlockLoadExpr is : "%arg9" or "%arg9 + 32", we need to judge whether storeOp's 
          /// index contains a reference of "%arg9" first. If so, LoadExpr = LoadExpr - BlockLoadExpr
          // check(LoadExpr )

            SmallDenseMap<mlir::Value, unsigned> BlkLoadOperands = getOperandInRank(BlockLoad, r);
            SmallDenseMap<mlir::Value, unsigned> StoreOperands = getOperandInRank(storeop, r);
            // assert(BlkLoadOperands.size() == 1 && "BlockLoadExpr could only contain one dim.");
            // DenseMapPair<mlir::Value, unsigned> BlkLoadOperand = *(BlkLoadOperands.begin());

            /// check whether the block Load Operand is contained by Store Operands
            // unsigned input;
            // llvm::errs() << "[debug] before simplified StoreExpr: " << StoreExpr << "\n";
            /// TODO: check this part
            for(DenseMapPair<mlir::Value, unsigned> BlkLoadOperand : BlkLoadOperands){
              for(DenseMapPair<mlir::Value, unsigned> StoreOperand : StoreOperands){
                if(StoreOperand.first == BlkLoadOperand.first){
                  assert(MultiplicatorOfDim(StoreExpr, StoreOperand.second) == 
                        MultiplicatorOfDim(BlockLoadExpr, BlkLoadOperand.second) &&
                      "Only if the multiplicator equals the dim can be eliminated.");
                  AffineExpr ConstantExpr = getAffineConstantExpr(
                      MultiplicatorOfDim(StoreExpr, StoreOperand.second), StoreExpr.getContext());
                  StoreExpr = StoreExpr -
                    getAffineDimExpr(StoreOperand.second, StoreExpr.getContext()) * ConstantExpr;
                  break;
                }
              }
              StoreExpr = simplifyAffineExpr(StoreExpr, StoreMap.getNumDims(), StoreMap.getNumSymbols());
              llvm::errs() << "[debug] after simplified StoreExpr: " << StoreExpr << "\n";
            }
            /// fixed: maybe in different dim 
          }
          StoreExprs.push_back(StoreExpr);
          llvm::errs() << "[debug] after remove LoadExpr: " << StoreExpr << "\n";
        }
        if(StoreExprs.size() == 0) StoreExprs.push_back(builder.getAffineConstantExpr(0));
        AffineMap newStoreMap = AffineMap::get(StoreMap.getNumDims(),StoreMap.getNumSymbols(),StoreExprs,builder.getContext()); 
        storeop.getOperation()->setAttr(AffineStoreOp::getMapAttrStrName(),AffineMapAttr::get(newStoreMap));
        llvm::errs() << "[debug] After replace Kernel: \n";Kernel.dump();
      }
    }
    /////////
    /// Step 5 : Generate Block Store
    /////////
    // / DataBlockStoreOp which is coupled with above DataBlockLoadOp
    ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
                (Kernel.getLoc(), BlockLoad.getResult(), memref, memIVmap, IVs);
    // ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
    //             (Kernel.getLoc(),  memref, memIVmap, IVs);
    Kernel.getOperation()->getBlock()->push_back(BlockStore);
    BlockStore.getOperation()->moveAfter(Kernel);
    BlockStore.setKernelName(Kernel.getKernelName());
    BlockStore.setId(std::to_string(BlockLoadStoreOpId++));
  }

  //////////
  /// Generate Other BlockLoadOp
  //////////
  // SmallDenseMap<AffineLoadOp, ADORA::DataBlockLoadOp> LoadToBlkLoad;
  Kernel.walk([&](AffineLoadOp loadop)->WalkResult
  {
    if(findElement(VisitedOperations, loadop.getOperation()) != -1)
      return WalkResult::advance();

    LLVM_DEBUG(llvm::errs() << "[debug] loadop: "; loadop.dump(););
    if(succeeded(memrefRegion.compute(loadop, 
                /*loopDepth=*/getNestingDepth(Kernel.getOperation())))){ /// Bind loadop and memrefRegion through compute()
      memref = memrefRegion.memref; /// original memref Op of this loadOP
      MemRefType memRefType = memref.getType().cast<MemRefType>(); /// contains shape info of original memref
      unsigned rank = memRefType.getRank(); /// dim number of original memref
      assert(rank == memrefRegion.cst.getNumDimVars() && "inconsistent memref region");      
      
      /// To fix: memIVs should be a setVector??
      // SmallVector<Value, 4> memIVs; /// stores the Interation Variables out of kernel that determined load bound
      SmallVector<AffineExpr, 4> memExprs;
      SmallVector<int64_t, 4> newMemRefShape; /// stores the new dim shape of blockloadOp which will replace oringinal memref

      /////////////
      /// Step1: Get InductionVar(IV) of this memrefRegion
      /////////////

      memrefRegion.cst.getValues(memrefRegion.cst.getNumDimVars(),
          memrefRegion.cst.getNumDimAndSymbolVars(), &IVs);
      // assert(IVs.size() <= 1  /// To fix: if IVs.size() > 1 ?
      //       && " This kernel should only have 1 outer IV as input arguments.");

      // if(IVs.size()==1){
      //   AffineForOp iv = getForInductionVarOwner(IVs.front());
      //   memIVs.push_back(iv.getInductionVar());          
      // }

      /// For different dim of original memref
      LLVM_DEBUG(for(auto &iv : IVs){
        llvm::errs() << "iv:" << iv << "\n";
      });

      for (unsigned r = 0; r < rank; r++) {
        AffineExpr lbExpr_minspace, ubExpr_minspace;

        /////////////
        /// Step2: Get Lower And Upper Bound in this rank and get the lpMap and upMap 
        /// that determines the min size of space. The size should be constant.
        /////////////
        AffineMap lbMap, ubMap;
        memrefRegion.getLowerAndUpperBound(r, lbMap, ubMap);
        assert(lbMap.getNumDims() == IVs.size() && ubMap.getNumDims() == IVs.size()\
              && " Num of bound's dim should be the same with num of IVs!");
        LLVM_DEBUG(llvm::errs() << "[debug] lbMap: " << lbMap << " , ubMap: "<< ubMap << "\n";);
        int64_t min_space = -1;

        for(AffineExpr lbExpr : lbMap.getResults()){ 
          for(AffineExpr ubExpr : ubMap.getResults()){
            AffineExpr diffExpr = ubExpr - lbExpr;
            diffExpr = simplifyAffineExpr(diffExpr, lbMap.getNumDims(), lbMap.getNumSymbols());
            LLVM_DEBUG(llvm::errs() << "[debug] diffExpr: " << diffExpr << "\n");
            LLVM_DEBUG(llvm::errs() << "[debug] lbExpr: " << lbExpr << ",ubExpr:" << ubExpr << "\n");
            if(diffExpr.isSymbolicOrConstant()){
              /// Found a Constant diff
              AffineConstantExpr diffExpr_const=diffExpr.dyn_cast<AffineConstantExpr>();
              if((memRefType.hasStaticShape() && diffExpr_const.getValue()==memRefType.getNumElements()) 
                && min_space==-1){
                /// This upper and lowerbound is constrained by 
                /// original memref's size and a smaller min_space
                /// is not found yet.
                lbExpr_minspace = lbExpr;
                ubExpr_minspace = ubExpr;
                min_space = diffExpr_const.getValue();
              }
              else if(diffExpr_const.getValue() < min_space || min_space==-1){
                /// Found a smaller memory space, store the Affine Expr of lb and ub
                lbExpr_minspace = lbExpr;
                ubExpr_minspace = ubExpr;
                min_space = diffExpr_const.getValue();
              }
              else if(diffExpr_const.getValue() == min_space && 
                !ubExpr.isSymbolicOrConstant() && !lbExpr.isSymbolicOrConstant()){
                /// This is to set the lb to the expression from
                /// the one which contains dim variables rather than a constant bound
                lbExpr_minspace = lbExpr;
                ubExpr_minspace = ubExpr;
                min_space = diffExpr_const.getValue();
              }
            }
          }
        }
        assert(min_space != -1 && 
                  " The memory space this L/S op access has different size in different Iterations!");

        memExprs.push_back(lbExpr_minspace);
        newMemRefShape.push_back(min_space);
      }

      //// Fix the shape of newMemRefShape to burst a sequential DMA quest as possible
      SmallVector<int64_t> FilledNewMemRefShape = \
        FillMemRefShape(/*ToFillShape=*/(ArrayRef<int64_t>)newMemRefShape, /*TargetMemref=*/memRefType/*,Aligned=BITS_64*/ );

      /////////
      /// Step 3 :Create DataBlockLoadOp and replace original Memref in loadOp 
      /////////
      // ArrayRef<int64_t> try_array = {1,2,3};
      // MemRefType tryMemRef = MemRefType::get(try_array, memRefType.getElementType());
      // llvm::errs() << "[debug] tryMemRef: ";tryMemRef.dump();      

      AffineMap memIVmap = AffineMap::get(IVs.size(), /*symbolCount=*/0, memExprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
      MemRefType newMemRef = MemRefType::get(FilledNewMemRefShape, memRefType.getElementType());
      llvm::errs() << "[debug] newMemRef: ";newMemRef.dump();

      ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                (Kernel.getLoc(), memref, memIVmap, IVs, newMemRef);
      // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
                // (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
      Kernel.getOperation()->getBlock()->push_back(BlockLoad);
      BlockLoad.getOperation()->moveBefore(Kernel);
      BlockLoad.setKernelName(Kernel.getKernelName());
      BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));

      // llvm::errs() << "[debug] Before replace Kernel: ";Kernel.dump();
      // replaceMemrefOfAffineLSop(loadop, BlockLoad.getResult());
      loadop.getOperation()->replaceUsesOfWith(memref, BlockLoad.getResult());
      // llvm::errs() << "[debug] BlockLoad: " << BlockLoad << "\n";
      /////////
      /// Step 4 : Change index of loadop according to the new BlockLoadOp
      /////////
      
      // llvm::errs() << "[debug] memIVmap: " << memIVmap << "\n";
      // llvm::errs() << "[debug] BlockLoad: " << BlockLoad << "\n";
      
      /** 
       * Complete a conversion for load op's index, here is an example:
       * BlockLoadOp:
       *   %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> 
       * LoadOp in kernel region:
       *   %2 = affine.load %arg1[%arg9 + %arg11, %arg12] : memref<32x32xf32>
       * 
       * %arg1 become %0,  %arg9 + %arg11 = %arg9 + %arg11 - %arg9 = %arg11
       * LoadOp after conversion should be:
       *   %2 = affine.load %0[%arg11, %arg12] : memref<1x32xf32>
       * 
       * This adjustment is caused by the change of head position of the memref.
       * 
       * But for another example:
       * 
       * BlockLoadOp:
       *   %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<8x32xf32> 
       * LoadOp in kernel region:
       *   %2 = affine.load %arg1[%arg10, %arg12] : memref<32x32xf32>
       * 
       * 
      */
      assert(memExprs.size() == rank && "Each rank should have its own affine expr.");
      SmallVector<AffineExpr, 4> LoadExprs;
      AffineMap LoadMap = loadop.getAffineMapAttr().getValue();
      for (unsigned r = 0; r < rank; r++) {
        AffineExpr LoadExpr = LoadMap.getResult(r);
        llvm::errs() << "[debug] before remove LoadExpr: " << LoadExpr << "\n";
        // operand_range LoadOperands = loadop.getMapOperands();
        AffineExpr BlockLoadExpr = memExprs[r];
        llvm::errs() << "[debug] BlockLoadExpr: " << BlockLoadExpr << "\n";       
        // llvm::errs() << "[debug] BlockLoadExpr.getKind(): " << BlockLoadExpr.getKind() << "\n";    
        assert((BlockLoadExpr.getKind() == AffineExprKind::DimId ||     // like: d0
                BlockLoadExpr.getKind() == AffineExprKind::Add   ||     // like: d0 + 32
                BlockLoadExpr.getKind() == AffineExprKind::Mul   ||     // like: d0 * 32
                BlockLoadExpr.getKind() == AffineExprKind::Constant) && // like: 32
                "Only handle '%arg9' or '%arg9 + 32' like index for BlockLoadExpr right now.");
        
        if(BlockLoadExpr.getKind() == AffineExprKind::Constant){
          LoadExpr = LoadExpr - BlockLoadExpr; /// forward the head position of LoadOp by a const value
        }
        else { 
          /// If BlockLoadExpr is : "%arg9" or "%arg9 + 32", we need to judge whether loadOp's 
          /// index contains a reference of "%arg9" first. If so, LoadExpr = LoadExpr - BlockLoadExpr
          // check(LoadExpr )

          SmallDenseMap<mlir::Value, unsigned> BlkLoadOperands = getOperandInRank(BlockLoad, r);
          SmallDenseMap<mlir::Value, unsigned> LoadOperands = getOperandInRank(loadop, r);
          // assert(BlkLoadOperands.size() == 1 && "BlockLoadExpr could only contain one dim.");
          // DenseMapPair<mlir::Value, unsigned> BlkLoadOperand = *(BlkLoadOperands.begin());

          /// check whether the block Load Operand is contained by Load Operands
          // unsigned input;
          // llvm::errs() << "[debug] before simplified LoadExpr: " << LoadExpr << "\n";
          /// TODO: check this part
          for(DenseMapPair<mlir::Value, unsigned> BlkLoadOperand : BlkLoadOperands){
            for(DenseMapPair<mlir::Value, unsigned> LoadOperand : LoadOperands){
              if(LoadOperand.first == BlkLoadOperand.first){
                assert(MultiplicatorOfDim(LoadExpr, LoadOperand.second) == 
                      MultiplicatorOfDim(BlockLoadExpr, BlkLoadOperand.second) &&
                      "Only if the multiplicator equals the dim can be eliminated.");
                AffineExpr ConstantExpr = getAffineConstantExpr(
                      MultiplicatorOfDim(LoadExpr, LoadOperand.second), LoadExpr.getContext());
                LoadExpr = LoadExpr -
                  getAffineDimExpr(LoadOperand.second, LoadExpr.getContext()) * ConstantExpr;
                break;
              }
            }
            LoadExpr = simplifyAffineExpr(LoadExpr, LoadMap.getNumDims(), LoadMap.getNumSymbols());
            llvm::errs() << "[debug] after simplified LoadExpr: " << LoadExpr << "\n";
          }
         

          /// fixed: maybe in different dim 
          // if(input != LoadOperands.size()){ /// find the block Load Operand in operand of loadOp
          //   for(unsigned dim = 0; dim < BlockLoad.getMapOperands().size(); dim++){
          //     if(BlockLoadExpr.isFunctionOfDim(dim) 
          //       &&LoadExpr.isFunctionOfDim(dim) ){
          //       LoadExpr = LoadExpr - BlockLoadExpr;
          //     }
          //   }
          //   // for(auto operand : LoadOperands){
          //   //   auto operand = LoadOperands[i];
          //   //   if(operand == BlkLoadOperand){
                
          //   //   }
          //   // }
          // }
        }
        LoadExprs.push_back(LoadExpr);
        llvm::errs() << "[debug] after remove LoadExpr: " << LoadExpr << "\n";
      }
      if(LoadExprs.size() == 0) LoadExprs.push_back(builder.getAffineConstantExpr(0));
      AffineMap newLoadMap = AffineMap::get(LoadMap.getNumDims(),LoadMap.getNumSymbols(),LoadExprs,builder.getContext()); 
      loadop.getOperation()->setAttr(AffineLoadOp::getMapAttrStrName(),AffineMapAttr::get(newLoadMap));
      llvm::errs() << "[debug] After replace Kernel: \n";Kernel.dump();
      // LoadToBlkLoad[loadop] = BlockLoad;
    } 
  });


  //////////
  /// Generate LocalMalloc-BlockStoreOp pair
  //////////
  // SmallDenseMap<AffineStoreOp, ADORA::DataBlockLoadOp, 4> StoreToBlkLoadOp; 
  SmallDenseMap<AffineStoreOp, mlir::Value, 4> StoreToMem; 
  SmallDenseMap<AffineStoreOp, mlir::Value, 4> StoreToBlockLoad; 
  SmallDenseMap<mlir::Value, std::unique_ptr<MemRefRegion>, 4> MemToRegion; 
  Kernel.walk([&](AffineStoreOp storeop)-> WalkResult
  {
    if(findElement(VisitedOperations, storeop.getOperation()) != -1)
      return WalkResult::advance();
    //////////
    /// Step 1: Get the memref region of store op. What should be noted is:
    ///  2 accesses to an identical store memref should compute their unionBoundingBox region.
    ///  But for those stores with RAW dependency, just correspond this store to its depended load.
    //////////
    llvm::errs() << "[debug] storeop: ";storeop.dump();

    auto region = std::make_unique<MemRefRegion>(storeop.getLoc());
    if(failed(region->compute(storeop, 
                /*loopDepth=*/getNestingDepth(Kernel.getOperation())))){ /// Bind loadop and memrefRegion through compute()
      return storeop->emitError("error obtaining memory region\n");
    }
    StoreToMem[storeop] = std::move(region->memref);

    auto it = MemToRegion.find(region->memref);
    if (it == MemToRegion.end()) {
      MemToRegion[region->memref] = std::move(region);
    } else if (failed(it->second->unionBoundingBox(*region))) {
      return storeop.getOperation()->emitWarning(
          "getMemoryFootprintBytes: unable to perform a union on a memory "
          "region");
    }

    return WalkResult::advance();
  });

  // for(const auto &StorePair : StoreToBlkLoadOp){
  //   DataBlockLoadOp blkload = StorePair.second;
  //   memref = blkload.getResult();
  //   // / DataBlockStoreOp which is coupled with above DataBlockLoadOp
  //   ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
  //               (Kernel.getLoc(), blkload.getResult(), blkload.getOriginalMemref(), blkload.getAffineMap(), blkload.getIndices());
  //   // ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
  //   //             (Kernel.getLoc(),  memref, memIVmap, IVs);
  //   Kernel.getOperation()->getBlock()->push_back(BlockStore);
  //   BlockStore.getOperation()->moveAfter(Kernel);
  //   BlockStore.setKernelName(Kernel.getKernelName());
  //   BlockStore.setId(blkload.getId().str());
  // }

  for (const auto &MemAndRegion : MemToRegion) {
    memrefRegion = *(MemAndRegion.getSecond());
    memref = memrefRegion.memref;
    // llvm::errs() << "[debug] memref: " << MemAndRegion.getFirst();
    // llvm::errs() << ", region: \n"; memrefRegion.dump();
    

    // Value SourceMemref = memrefRegion.memref; /// original memref Op of this loadOP
    MemRefType memRefType = memref.getType().cast<MemRefType>(); /// contains shape info of original memref
    unsigned rank = memRefType.getRank(); /// dim number of original memref
    assert(rank == memrefRegion.cst.getNumDimVars() && "inconsistent memref region");      
      
    /// To fix: memIVs should be a setVector??
    // SmallVector<Value, 4> memIVs; /// stores the Interation Variables out of kernel that determined load bound
    SmallVector<AffineExpr, 4> memExprs;
    SmallVector<int64_t, 4> newMemRefShape; /// stores the new dim shape of blockloadOp which will replace oringinal memref

    ////////////
    /// Step2: Get InductionVar(IV) of this memrefRegion
    /////////////

    memrefRegion.cst.getValues(memrefRegion.cst.getNumDimVars(),
        memrefRegion.cst.getNumDimAndSymbolVars(), &IVs);
    // assert(IVs.size() <= 1  /// To fix: if IVs.size() > 1 ?
    //       && " This kernel should only have 1 outer IV as input arguments.");

    /// For different dim of original memref
    for (unsigned r = 0; r < rank; r++) {
      AffineExpr lbExpr_minspace, ubExpr_minspace;

      /////////////
      /// Step3: Get Lower And Upper Bound in this rank and get the lpMap and upMap 
      /// that determines the min size of space. The size should be constant.
      /////////////
      AffineMap lbMap, ubMap;
      memrefRegion.getLowerAndUpperBound(r, lbMap, ubMap);
      assert(lbMap.getNumDims() == IVs.size() && ubMap.getNumDims() == IVs.size()\
            && " Num of bound's dim should be the same with num of IVs!");
      llvm::errs() << "[debug] lbMap: " << lbMap << " , ubMap: "<< ubMap << "\n";
      int64_t min_space = -1;
      for(AffineExpr lbExpr : lbMap.getResults()){
        for(AffineExpr ubExpr : ubMap.getResults()){
          AffineExpr diffExpr = ubExpr - lbExpr;
          diffExpr = simplifyAffineExpr(diffExpr, lbMap.getNumDims(), lbMap.getNumSymbols());
          LLVM_DEBUG(llvm::errs() << "[debug] diffExpr: " << diffExpr << "\n");
          if(diffExpr.isSymbolicOrConstant()){
            /// Found a Constant diff
            AffineConstantExpr diffExpr_const=diffExpr.dyn_cast<AffineConstantExpr>();
            if((memRefType.hasStaticShape() && diffExpr_const.getValue()==memRefType.getNumElements()) 
                && min_space==-1){
              /// This upper and lowerbound is constrained by 
              /// original memref's size and a smaller min_space
              /// is not found yet.
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
            else if(diffExpr_const.getValue() < min_space || min_space==-1){
              /// Found a smaller memory space, store the Affine Expr of lb and ub
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
            else if(diffExpr_const.getValue() == min_space && 
              !ubExpr.isSymbolicOrConstant() && !lbExpr.isSymbolicOrConstant()){
              /// This is to set the lb to the expression from
              /// the one which contains dim variables rather than a constant bound
              lbExpr_minspace = lbExpr;
              ubExpr_minspace = ubExpr;
              min_space = diffExpr_const.getValue();
            }
          }
        }
      }

      assert(min_space != -1 && 
          " The memory space this L/S op access has different size in different Iterations!");
      memExprs.push_back(lbExpr_minspace);
      newMemRefShape.push_back(min_space);
    }
      
    /////////
    /// Step 4 :Create LocalMemAllocOp and DataBlockStoreOp
    /////////
    /// LocalMemAllocOp which is coupled with following DataBlockStoreOp
    //// Fix the shape of newMemRefShape to burst a sequential DMA quest as possible
    SmallVector<int64_t> FilledNewMemRefShape = \
      FillMemRefShape(/*ToFillShape=*/(ArrayRef<int64_t>)newMemRefShape, /*TargetMemref=*/memRefType/*,Aligned=BITS_64*/ );
    AffineMap memIVmap = AffineMap::get(IVs.size(), /*symbolCount=*/0, memExprs, builder.getContext());   /// stores corresponding AffineMap of above memIVs
    MemRefType newMemRef = MemRefType::get( (ArrayRef<int64_t>)FilledNewMemRefShape, memRefType.getElementType());

    ADORA::LocalMemAllocOp  BlockAlloc = builder.create<ADORA::LocalMemAllocOp>\
                (Kernel.getLoc(), newMemRef);
    // ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>\
    //             (Kernel.getLoc(), memref, memIVmap, IVs, memRefType);
    Kernel.getOperation()->getBlock()->push_back(BlockAlloc);
    BlockAlloc.getOperation()->moveBefore(Kernel);
    BlockAlloc.setKernelName(Kernel.getKernelName());
    BlockAlloc.setId(std::to_string(BlockLoadStoreOpId));


    // / DataBlockStoreOp which is coupled with above DataBlockLoadOp
    ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
                (Kernel.getLoc(), BlockAlloc.getResult(), memref, memIVmap, IVs);
    // ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>\
    //             (Kernel.getLoc(),  memref, memIVmap, IVs);
    Kernel.getOperation()->getBlock()->push_back(BlockStore);
    BlockStore.getOperation()->moveAfter(Kernel);
    BlockStore.setKernelName(Kernel.getKernelName());
    BlockStore.setId(std::to_string(BlockLoadStoreOpId++));

    /////////
    /// Step 5 : replace original Memref in storeOp and Change index of storeop according to the new BlockLoadOp
    /////////
    for(auto StoreAndMem : StoreToMem){
      if(StoreAndMem.getSecond() == MemAndRegion.getFirst()){
        // replaceMemrefOfAffineLSop(StoreAndMem.getFirst(), BlockAlloc.getResult());
        StoreAndMem.getFirst().getOperation()->replaceUsesOfWith(memref, BlockAlloc.getResult());
        AffineStoreOp storeop =  StoreAndMem.getFirst();
        /** 
          * Complete a conversion for load op's index, here is an example:
          * BlockLoadOp:
          *   %0 = ADORA.BlockLoad %arg1 [%arg9, 0] : memref<32x32xf32> -> memref<1x32xf32> 
          * StoreOp in kernel region:
          *   %2 = affine.store %1, %arg1[%arg9 + %arg11, %arg12] : memref<32x32xf32>
          * 
          * %arg1 become %0,  %arg9 + %arg11 = %arg9 + %arg11 - %arg9 = %arg11
          * StoreOp after conversion should be:
          *   %2 = affine.store %1 ,%0[%arg11, %arg12] : memref<1x32xf32>
          * 
          * This adjustment is caused by the change of head position of the memref 
          * 
          */
        assert(memExprs.size() == rank && "Each rank should have its own affine expr.");
        SmallVector<AffineExpr, 4> StoreExprs;
        AffineMap StoreMap = storeop.getAffineMapAttr().getValue();
        for (unsigned r = 0; r < rank; r++) {
          AffineExpr StoreExpr = StoreMap.getResult(r);
          llvm::errs() << "[debug] before remove StoreExpr: " << StoreExpr << "\n";
          // operand_range LoadOperands = loadop.getMapOperands();
          AffineExpr BlockStoreExpr = memExprs[r];
          llvm::errs() << "[debug] before remove BlockStoreExpr: " << BlockStoreExpr << "\n";
          assert((BlockStoreExpr.getKind() == AffineExprKind::DimId ||
                  BlockStoreExpr.getKind() == AffineExprKind::Add   ||
                  BlockStoreExpr.getKind() == AffineExprKind::Mul   ||
                  BlockStoreExpr.getKind() == AffineExprKind::Constant) &&
                  "Only handle '%arg9' or '%arg9 + 32' like index for BlockStoreExpr right now.");
        
          if(BlockStoreExpr.getKind() == AffineExprKind::Constant){
            StoreExpr = StoreExpr - BlockStoreExpr; /// forward the head position of LoadOp by a const value
          }
          else { 
            /// If BlockLoadExpr is : "%arg9" or "%arg9 + 32", we need to judge whether loadOp's 
            /// index contains a reference of "%arg9" first. If so, StoreExpr = StoreExpr - BlockLoadExpr;
            // check(LoadExpr )
        
            SmallDenseMap<mlir::Value, unsigned> BlkStoreOperands = getOperandInRank(BlockStore, r);
            SmallDenseMap<mlir::Value, unsigned> StoreOperands = getOperandInRank(storeop, r);
            assert(BlkStoreOperands.size() == 1 && "BlockStoreExpr could only contain one dim.");
            DenseMapPair<mlir::Value, unsigned> BlkStoreOperand = *(BlkStoreOperands.begin());

            /// check whether the block Load Operand is contained by Load Operands
            unsigned input;
            for(DenseMapPair<mlir::Value, unsigned> StoreOperand : StoreOperands){
              if(StoreOperand.first == BlkStoreOperand.first){
                assert(MultiplicatorOfDim(StoreExpr, StoreOperand.second) == 
                      MultiplicatorOfDim(BlockStoreExpr, BlkStoreOperand.second) &&
                      "Only if the multiplicator equals the dim can be eliminated.");
                AffineExpr ConstantExpr = getAffineConstantExpr(
                      MultiplicatorOfDim(StoreExpr, StoreOperand.second), StoreExpr.getContext());
                StoreExpr = StoreExpr -
                    getAffineDimExpr(StoreOperand.second, StoreExpr.getContext()) * ConstantExpr;
                break;
              }
            }
            StoreExpr = simplifyAffineExpr(StoreExpr, StoreMap.getNumDims(),StoreMap.getNumSymbols());
          }
          StoreExprs.push_back(StoreExpr);
        }
        if(StoreExprs.size() == 0) StoreExprs.push_back(builder.getAffineConstantExpr(0));
        AffineMap newStoreMap = AffineMap::get(StoreMap.getNumDims(),StoreMap.getNumSymbols(),StoreExprs,builder.getContext()); 
        storeop.getOperation()->setAttr(AffineStoreOp::getMapAttrStrName(),AffineMapAttr::get(newStoreMap));      
      }
    }
  }
  return 1;
}


/// @brief  Eliminate the affine transformation of the upper/lower bound
///       of most-out loop in Kernel{ }
/// @param Kernel
int AdjustMemoryFootprintPass::EliminateOuterLoopAffineTrans(ADORA::KernelOp Kernel)
{
  MemRefRegion memrefRegion(Kernel.getLoc());
  mlir::Value memref;
  // Block& knBlock = Kernel.getBody().front(); /// kernel block
  mlir::OpBuilder builder(Kernel.getBody().getContext());
  // llvm::SetVector<Value> memIVs;
  llvm::SmallVector<mlir::Value, 4> IVs;

  /// Step 1:
  /// Check: whether the affine transformation of most-out loop can be eliminated.
  /// Get the IV which is out of kernel and is used in kernel
  Kernel.walk([&](Operation *op)
  { 
    if((op->getName().getStringRef()== AffineLoadOp::getOperationName()  ||
        op->getName().getStringRef()== AffineStoreOp::getOperationName())&&
        succeeded(memrefRegion.compute(op,
                                       /*loopDepth=*/getNestingDepth(Kernel.getOperation()))))
    { /// Bind loadop and memrefRegion through compute()
      // op->dump();
      // memrefRegion.dump();
      memref = memrefRegion.memref;
      MemRefType memRefType = memref.getType().cast<MemRefType>();

      unsigned rank = memRefType.getRank();
      assert(rank == memrefRegion.cst.getNumDimVars() && "inconsistent memref region");

      /////////////
      /// Get InductionVar(IV) of this memrefRegion
      /////////////
      // llvm::errs() << "cst:\n";
      // memrefRegion.cst.dump();
      llvm::SmallVector<mlir::Value, 4> _;
      memrefRegion.cst.getValues(memrefRegion.cst.getNumDimVars(),
                                 memrefRegion.cst.getNumDimAndSymbolVars(), &_);
      IVs.append(_);
      
      // llvm::errs() << "IV:\n";
      // for(auto IV: IVs){
      //   IV.dump();
      //   IV.getParentBlock()->getParentOp()->dump();
      // }
      /// eliminate unused indices of load or store op
      /// Note: It is inappropriate to eliminate unused indices right after
      /// function "ExplicitKernelDataBLockLoadStore" because we can't locate
      /// the affine transformation in for bound index which can be eliminated
      /// in that way.
      eliminateUnusedIndices(op);
    } 
    Kernel.dump();
  });

  llvm::errs() << "IV:\n";
  for(auto IV: IVs){
    IV.dump();
    IV.getParentBlock()->getParentOp()->dump();
  }

  /// Only the bound of outer-most loop in kernel can be
  /// the affine transformation of the memIVs outside kernel
  Kernel.walk([&](AffineForOp forop)
  {
    if(forop.getOperation()->getParentOp() == Kernel.getOperation()){
      /// Get an outer-most ForOp
      if(forop.hasConstantBounds())
        return WalkResult::advance();
      
      ValueRange lboperands= forop.getLowerBoundOperands();
      ValueRange upoperands= forop.getUpperBoundOperands();

      /// if bounds belongs to following 2 situation, just skip handling
      /// TODO: really?
      ///     affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1)
      /// or  affine.for %arg3 = 0 to affine_map<(d0) -> (1800 - d0)>(%arg2) 
      if((lboperands.size() != 1 || upoperands.size() != 1)) {
        if(IsIterationSpaceSupported(forop)){
          return WalkResult::advance();
        }
        else{
          LLVM_DEBUG(forop.dump());
          assert(false && "Can't handle this kind of irregular iteration space for above for op");
        }
      }

      /// if bound is an affine transformation of outer memIVs
      /// then the transormation can be eliminated
      assert(lboperands.size() == 1 && upoperands.size() == 1
              && "Only support the situation that both lower and upper bounds contain 1 operand.");
      
      mlir::Value operand = lboperands.front();      
      assert(lboperands.front() == upoperands.front()
              && "Outermost loop should take in only 1 operand!");
      
      AffineExpr lbExpr, ubExpr;
      AffineMap Map;
      if(ADORA::findElement(IVs, operand)!=-1){
        /// Operand is out of kernel so elimination is legal
        lbExpr = forop.getLowerBoundMap().getResult(0);
        ubExpr = forop.getUpperBoundMap().getResult(0);
        // llvm::errs() << "[debug] lbExpr: " << lbExpr << ", ubExpr: "<< ubExpr <<"\n";
        AffineExpr ConstantExpr;
        ConstantExpr = ::mlir::ADORA::getConstPartofAffineExpr(lbExpr);
        // llvm::errs() << "[debug] Constant: " << ConstantExpr<<"\n";
        Map = AffineMap::get(0, 0, ConstantExpr);
        forop.setLowerBound({}, Map);

        ConstantExpr = ::mlir::ADORA::getConstPartofAffineExpr(ubExpr);
        // llvm::errs() << "[debug] Constant: " << ConstantExpr<<"\n";
        Map = AffineMap::get(0, 0, ConstantExpr);
        forop.setUpperBound({}, Map);
      }
      LLVM_DEBUG(Kernel.dump());
    }
    return WalkResult::advance(); 
  });

  return 1;
}


/// @brief Fill the shape of one memref, obey to 2 rules:
///   1. DMA Request should be continuous as possible
///   2. One dma request align with the system bus aligned bits;
/// @param ToFillShape
/// @param TargetShape
/// @param AlignedBits
///
SmallVector<int64_t> AdjustMemoryFootprintPass::FillMemRefShape 
    (ArrayRef<int64_t> ToFillShape, MemRefType TargetMemref, AlignedBitsType AlignedBits/*=AlignedBitsType::BITS_64*/){
  ArrayRef<int64_t>  TargetShape = TargetMemref.getShape();
  assert(ToFillShape.size() == TargetShape.size());

  unsigned BitWidth = TargetMemref.getElementTypeBitWidth();
  unsigned SingleArray_Size_bits = SingleArray_Size * 1024 * 8;
  unsigned Aligned_count = AlignedBits/BitWidth;
  int SingleDMAlen = 1;
  
  SmallVector<int64_t> NewShape(ToFillShape.size());
  if(ToFillShape.size() == 0){
    NewShape.push_back(2);
    return NewShape;
  }

  LLVM_DEBUG(
  llvm::errs() << "ToFillShape Shape: ";
  for(int r = 0; r < ToFillShape.size(); r++){
    llvm::errs() << ToFillShape[r] << " ";
  }
  llvm::errs() << "\n";);

  for(int r = ToFillShape.size() - 1; r >= 0; r--){
    NewShape[r] = ToFillShape[r];
  }

  if(!TargetMemref.hasStaticShape()){
    return NewShape;
  }

  // llvm::errs() << "TargetShape Shape: ";
  // for(int r = TargetShape.size() - 1; r >= 0; r--){
  //   llvm::errs() << TargetShape[r] << " ";
  //   // NewShape[r] = TargetShape[r];
  // }
  // llvm::errs() << "\n";
 
  /// rule 1: DMA Request should be continuous as possible
  for(int r = ToFillShape.size() - 1; r >= 0; r--){
    if((TargetShape[r] - ToFillShape[r]) * SingleDMAlen <= IDEAL_DMA_REQ_LEN / 10 
    || (SingleDMAlen < IDEAL_DMA_REQ_LEN / 10 
        && (TargetShape[r] - ToFillShape[r]) * SingleDMAlen <= IDEAL_DMA_REQ_LEN / 4))
    /// DMA Request could be continuous
    {
      NewShape[r] = TargetShape[r];
      SingleDMAlen *= TargetShape[r];
    }
    else{
      NewShape[r] = ToFillShape[r];
      break;
    }
  }

  llvm::errs() << "NewShape Shape: ";
  for(int r = NewShape.size() - 1; r >= 0; r--){
    llvm::errs() << NewShape[r] << " ";
    // NewShape[r] = TargetShape[r];
  }
  llvm::errs() << "\n";

  /// Check whether new shape oversize the local memory limit.
  /// If oversized, NewShape is not applied.
  int NewShapeBits = BitWidth;
  for(int r = ToFillShape.size() - 1; r >= 0; r--)
    NewShapeBits *= NewShape[r];

  if(NewShapeBits > SingleArray_Size_bits){
    for(int r = ToFillShape.size() - 1; r >= 0; r--)
      NewShape[r] = ToFillShape[r];
  }


  /// rule 2: One dma request align with the system bus aligned bits;
  unsigned Elements_notAligned = 0;
  bool CannotAlign = false; //such as memref<1x231x231x3xf32>
  do {
    int r;
    if(Elements_notAligned){
      for(r = ToFillShape.size() - 1; r >= 0; r--){
        if(NewShape[r] != TargetShape[r]){
          NewShape[r] ++;
          break;
        }
      }
    }
    unsigned singleDMARequstLen = 1;
    for(r = ToFillShape.size() - 1; r >= 0; r--){
      singleDMARequstLen *= NewShape[r];
      if(NewShape[r] != TargetShape[r]){
        break;
      }
    }
    
    CannotAlign = (r >= 0 ? false: true);
    Elements_notAligned = singleDMARequstLen % Aligned_count ;
    
  } while(Elements_notAligned != 0 && !CannotAlign);

  NewShapeBits = BitWidth;
  for(int r = ToFillShape.size() - 1; r >= 0; r--)
    NewShapeBits *= NewShape[r];

  // assert(NewShapeBits <= SingleArray_Size_bits\ 
  // && "We can not handle if NewShapeBits > SingleArray_Size_bits after aligning.");
  

  LLVM_DEBUG(
  llvm::errs() << "Filled Shape Shape: ";
  for(int r = 0; r < NewShape.size(); r++){
    llvm::errs() << NewShape[r] << " ";
  }
  llvm::errs() << "\n";);

  // return (::llvm::ArrayRef<int64_t>)NewShape;
  return NewShape;
}
      



//===----------------------------------------------------------------------===//
// Utilities
//===----------------------------------------------------------------------===//
mlir::affine::AffineForOp AdjustMemoryFootprintPass::
    constructOuterLoopNest(mlir::affine::AffineForOp &OriginforOp)
{
  Location loc = OriginforOp.getLoc();

  // The outermost loops we add
  OpBuilder b(OriginforOp);

  // Loop bounds will be set later.
  AffineForOp OuterLoop = b.create<AffineForOp>(loc, 0, 0);
  // b.setInsertionPointToStart(OuterLoop.getBody());
  OuterLoop.getBody()->getOperations().splice(
      OuterLoop.getBody()->begin(), OriginforOp.getBody()->getOperations(),
      Block::iterator(OriginforOp));

  // adjust loop bounds
  // errs() << "[constructOuterLoopNest()] after OriginforOp:\n"; OriginforOp.dump();
  // errs() << "[constructOuterLoopNest()] after OuterLoop:\n"; OuterLoop.dump();
  return OuterLoop;
}

void AdjustMemoryFootprintPass::runOnOperation()
{
  for (auto FuncOp : getOperation().getOps<func::FuncOp>())
  {
    unsigned part_factor = 1; /// A kernel shoule be parted to partition_factor subkernels

    ADORA::KernelOp KernelToPart = check_AllKernelMemoryFootprint(FuncOp, part_factor);
    while (part_factor != 1)
    { // KernelToPart != NULL
      LLVM_DEBUG(errs() << "\n[debug]FuncOp:\n");
      LLVM_DEBUG(FuncOp.dump());

      outloop_partition(FuncOp, KernelToPart, part_factor);
      KernelToPart = check_AllKernelMemoryFootprint(FuncOp, part_factor);
      // break; ///for debug
    }
    
    /// simplify loop levels if possible
    simplifyAffileLoopLevel(FuncOp);


    ///////////////
    /// Generate explicit data block movement (load/store) for kernel to consume
    ///////////////
    if(ExplicitDataTrans==true){
      /// generate explicit data movement around Kernel{...}
      // FuncOp.dump();
      LLVM_DEBUG(llvm::errs() << "[dubug] Before ExplicitKernelDataBLockLoadStore: \n";FuncOp.dump(););
      FuncOp.walk([&](ADORA::KernelOp kernel)
      {
        // if(kernel->hasAttr("ADORAGemm")) return;

        ExplicitKernelDataBLockLoadStore(kernel);
      });
      // FuncOp.dump();
      LLVM_DEBUG(llvm::errs() << "[dubug] After ExplicitKernelDataBLockLoadStore: \n";FuncOp.dump(););

      /// Eliminate the affine transformation of the upper/lower bound 
      /// of most-out loop in Kernel{...}
      FuncOp.walk([&](ADORA::KernelOp kernel)
      {
        // if(kernel->hasAttr("ADORAGemm")) return;

        EliminateOuterLoopAffineTrans(kernel);
      });
      
      /// Remove unused arguments of Kernel's region
      // Kernel.walk([&](Region *region){ removeUnusedRegionArgs(*region); });
    }
  }

  
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::ADORA::createAdjustKernelMemoryFootprintPass()
{
  return std::make_unique<AdjustMemoryFootprintPass>();
}

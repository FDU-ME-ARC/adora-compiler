//===-  AffineLoopReorder.cpp -===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h" 
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/RegionUtils.h"
// #include "mlir/Transforms/LoopUtils.h"  
// #include "mlir/IR/AffineExpr.h"

#include "mlir/Support/LLVM.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Analysis/DependencyAnalysis.h"
#include "../PassDetail.h"
#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

// #include "stdc++.h"

using namespace llvm;
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define DEBUG_TYPE "affine-loop-reorder"

/// TODO: support to pass in permutation map.

namespace {
struct AffineLoopReorder : public AffineLoopReorderBase<AffineLoopReorder> {
  AffineLoopReorder() = default;

  SmallVector<SmallVector<unsigned>> findValidLoopPermutations(AffineForOp forOp);
  
  template <typename LoadOrStoreOpPointer>
    bool IsIndexOfAccess(AffineForOp for_tocheck,LoadOrStoreOpPointer accessop);

  LogicalResult ReorderOnAffineForOp(AffineForOp forOp);
  SmallVector<int64_t> GetAccessShapeAtThisLevel(AffineForOp ForLevel, SmallVector<Operation*> MemAccessOps);
  SmallVector<Operation*> Corresponding_MemAccessOps(AffineForOp forOp);
  SmallVector<std::pair<AffineLoadOp, AffineStoreOp>> getRelatedLoadStorePair(SmallVector<Operation*> Ops);
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    // llvm::errs() << "func:\n" << func << "\n";
    SmallVector<AffineForOp, 4> loops; // stores forOps from outermost to innermost
    func.walk([&](AffineForOp forOp) {
      if (getNestingDepth(forOp) == 0) {
        loops.insert(loops.begin(), forOp);
        // llvm::outs() << "For loop\n";
        // for(auto it=loops.begin();it!=loops.end();it++) { (*it).dump(); }
        ArrayRef<AffineForOp> loops_arrayRef = llvm::ArrayRef(loops);
        if (isPerfectlyNested(loops_arrayRef)) {
          ReorderOnAffineForOp(forOp);
        }
        else {
          // llvm::outs() << "Loops are not perfectly nested\n";
        }
        loops.clear();
        // if(loops.empty()) { llvm::outs() << "loops is empty\n"; }
      }
      else {
        loops.insert(loops.begin(), forOp);
      }
    });
  }
};
} // namespace

//1. For each loop l, compute number of memory accesses made when l is the innermost loop.
//        For innermost Loop, number of memory accesses = {
//            1 , when reference is loop invariant;
//            (tripCount/cacheLineSize), when reference has spatial reuse for loop l;
//            (tripCount), otherwise
//       }
//        a. For each reference group, choose a reference R.
//            i. if R has spatial reuse on loop l, add (tripCount/cacheLineSize) to number of memory accesses.
//            ii. else, if R has temporal reuse on loop l, add 1 to number of memory accesses.
//            iii. else, (add tripCount) to number of memory accesses.
//        b. Multiple the result of number of memory accesses by the tripCount of all the remaining loops.
// 2. Choose the loop with least number of memory accesses as the innermost loop, say it is L.
// 3. Find the valid loop permutation which has loop L as the innermost loop.
// 4. Find the loops which are parallel, does not carry any loop dependence.
// 5. For each loop permutation found in step 5, calculate the cost of synchronization.
//        Cost of synchronization is calculated for each parallel loop.
//        For a loop, synchronization cost = product of tripCounts of all loops which are at outer positions to this loop.
//6. Choose the permutation with the least synchronization cost as the best permutation.
LogicalResult AffineLoopReorder::ReorderOnAffineForOp(AffineForOp forOp) {
  SmallDenseMap<unsigned, SmallVector<SmallVector<Operation *>> > loop_refGroups = getReuseGroupsForEachLoop(forOp);

  AffineForOp rootForOp = forOp;
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, rootForOp);
  unsigned loopDepth = loops.size();
  if(loopDepth == 1){
    return LogicalResult::success();
  }
  SmallVector<unsigned> loopPermMap(loopDepth), OriginloopPerm(loopDepth);

  bool Interchangable = true;
  while(Interchangable){
    Interchangable = false;

    for (unsigned d=0;d < loopDepth - 1;d++) {
      loops.clear();
      getPerfectlyNestedLoops(loops, rootForOp);
      if(Interchangable) 
        break;/// traverse from root for op again
      
      /// initialize permutation map
      for (unsigned i = 0; i < loopDepth; i++) {
        OriginloopPerm[i] = i;
        if(i == d)
          loopPermMap[i] = d + 1; 
        else if(i == d + 1)
          loopPermMap[i] = d;
        else
          loopPermMap[i] = i;
      }

      AffineForOp OuterFor = loops[d];
      AffineForOp InnerFor = loops[d + 1];
      llvm::errs() << "Outer:\n"  << OuterFor ;
                    // << "Inner:\n" << InnerFor;
      /***
       * Step1: check whether two loop can be interchanged
      */
      ArrayRef<AffineForOp> loops_arrayRef = llvm::ArrayRef(loops);
      ArrayRef<unsigned> loopPermMap_arrayRef = llvm::ArrayRef(loopPermMap);
      if ( isValidLoopInterchangePermutation(loops_arrayRef,loopPermMap_arrayRef) ) {
        // llvm::errs() << "Valid interchange\n"; 
        /***
        * Step2: Find corresponding load and store operations and check 
        *       whether there exist load-store pair which access the same address in one iteration
        *       which might have the potential to be extracts as the Accumulation(ACC/AMUL) operation
        */
        SmallVector<Operation*> MemAccessOps_in = Corresponding_MemAccessOps(InnerFor);
        SmallVector<Operation*> MemAccessOps_out = Corresponding_MemAccessOps(OuterFor);
        // MemAccessOps_in.insert(MemAccessOps_in.end(), MemAccessOps_out.begin(), MemAccessOps_out.end());
        SmallVector<Operation*> MemAccessOps = ADORA::SetMergeForVector(MemAccessOps_in, MemAccessOps_out);
        SmallVector<std::pair<AffineLoadOp, AffineStoreOp>> LSPairs = getRelatedLoadStorePair(MemAccessOps);

        /***
        * Step3: If load-store pair which access the same address exits, consider temporal reuse;
        *   else, consider spatial reuse.
        */        
        if(LSPairs.size() != 0){
          /// If there exits temporal reuse, the corresponding loop should be outer.
          SmallDenseMap<std::pair<AffineLoadOp, AffineStoreOp>, bool> pair_to_interchange;
          bool skip = 0;
          for(auto lspair : LSPairs){
            if(IsIndexOfAccess(InnerFor, lspair.first)) {
              assert(IsIndexOfAccess(InnerFor, lspair.second));
              if(IsIndexOfAccess(OuterFor ,lspair.first)){
                /// If the outer loop corresponds to higher rank, then 
                /// there is no need to interchange
                std::pair<int, int> p_out = getCorrespondingRankAndMultiplicator(lspair.first, OuterFor);
                std::pair<int, int> p_in = getCorrespondingRankAndMultiplicator(lspair.first, InnerFor);
                if(p_out.first < p_in.first){
                  /// outer corresponds to higher rank, no need to permutate
                  Interchangable = false;
                  skip = true;
                }
                else if(p_out.first == p_in.first){
                  /// both corresponds to the same rank, compare Multiplicator
                  if(p_out.first > p_in.first){
                    Interchangable = false;
                    skip = true;
                  }
                  else{
                    pair_to_interchange[lspair] = true;
                  }
                }
                else{
                  pair_to_interchange[lspair] = true;
                }
              }
              else{
                pair_to_interchange[lspair] = true;
              }
            }
            else{
              pair_to_interchange[lspair] = false;
            }
          }
          if(skip){
            continue;
          }

          Interchangable = true;
          for(auto lspair_result : pair_to_interchange){
            if(lspair_result.second == false){
              /// If there is one pair do not prefer this interchange, then we won't interchange.
              Interchangable = false; 
            }
          }
          if(Interchangable){
            unsigned NewRootIndex = permuteLoops(loops, loopPermMap);
            break;
          }
        }
        /***
        * Step4: Spatial reuse is a more common situation.
        *  compare the shapes of memory access before and after interchanging
        */        
        /// Before interchanging
        SmallVector<int64_t> Shape_before = GetAccessShapeAtThisLevel(/*level=*/InnerFor, MemAccessOps);

        /// After interchanging
        unsigned NewRootIndex = permuteLoops(loops, loopPermMap);
        SmallVector<int64_t> Shape_after = GetAccessShapeAtThisLevel(/*level=*/OuterFor, MemAccessOps);
        llvm::errs() << "Shape_before:";
        for(auto d : Shape_before){
          llvm::errs() << d << " ";
        }
        llvm::errs() << ";\n";
        llvm::errs() << "Shape_after:";
        for(auto d : Shape_after){
          llvm::errs() << d << " ";
        }
        llvm::errs() << ";\n";

        for(unsigned d = 0; d < Shape_after.size() || d < Shape_before.size(); d++){
          /* We want deliver as more data to accelerator's local memory as possible,
          * so we want the last dimmension(lowest) of the shape is larger.
          */ 
          if(Shape_before[Shape_before.size()-1-d] < Shape_after[Shape_after.size()-1-d]){
            Interchangable = true;
            break;
          }
          else if(Shape_before[Shape_before.size()-1-d] > Shape_after[Shape_after.size()-1-d]){
            Interchangable = false;
            break;
          }
        }
        rootForOp = loops[NewRootIndex];
        if(!Interchangable){
          /// permute to original nested-loop levels
          loops.clear();
          getPerfectlyNestedLoops(loops, rootForOp);
          NewRootIndex = permuteLoops(loops, loopPermMap);
          rootForOp = loops[NewRootIndex];
          /// Don't interchange, check the next loop level
        }
      }
    }
  }

  return success();
}

SmallVector<Operation*> AffineLoopReorder::Corresponding_MemAccessOps(AffineForOp forOp){
  SmallVector<Operation *> loadAndStoreOpInsts;
  // llvm::errs() << "forOp: " << forOp << "\n";
  forOp.getOperation()->walk([&](Operation *opInst) {
    SmallVector<SmallVector<int> > AccessMatrix;
    unsigned arrayDimension;
    if (auto store = dyn_cast<AffineStoreOp>(opInst)) {
      auto memRefType = store.getMemRef().getType().template cast<MemRefType> ();
      arrayDimension = memRefType.getRank();
    }
    else if (auto load = dyn_cast<AffineLoadOp>(opInst)) {
      auto memRefType = load.getMemRef().getType().template cast<MemRefType> ();
      arrayDimension = memRefType.getRank();
    }
    else{
      return WalkResult::advance();
    }

    AccessMatrix = getAccessMatrix(opInst);
    for(unsigned i = 0; i <  arrayDimension; i++){
      if(AccessMatrix[i][getNestingDepth(forOp)] != 0){
        loadAndStoreOpInsts.push_back(opInst);
        // llvm::errs() << "Corresponfing: " << *opInst << "\n";
        break;
      }
    }

    return WalkResult::advance();
  });

  return loadAndStoreOpInsts;
}


SmallVector<int64_t> AffineLoopReorder::GetAccessShapeAtThisLevel(
      AffineForOp ForLevel, SmallVector<Operation*> AccessOps){
  SmallVector<int64_t> CriticalShape;
  std::optional<int64_t> BiggestSize;  
  MemRefRegion CriticalRegion(AccessOps[0]->getLoc());
  for(Operation* AccessOp : AccessOps){
    auto region = std::make_unique<MemRefRegion>(AccessOp->getLoc());
    if (failed(region->compute(AccessOp,
                  /*loopDepth=*/getNestingDepth(ForLevel)))) {
      assert(0 && "[Error] error obtaining memory region\n");
      // return AccessOp->emitError("error obtaining memory region\n");
    }
    SmallVector <int64_t> shape;
    // llvm::errs() << "region:\n";
    // region->dump();
    region->getConstantBoundingSizeAndShape(/*shape=*/&shape/*, lbs= , lbDivisors=*/);
    llvm::errs() << "op:"; AccessOp->dump();
    llvm::errs() << "shape:";
    for(auto d : shape){
      llvm::errs() << d << " ";
    }
    llvm::errs() << ";\n";
    std::optional<int64_t> size = region->getRegionSize();
    llvm::errs() << "size: "<< size <<"\n";

    if(size > BiggestSize){
      BiggestSize = size;
      CriticalShape = shape;
    }
  }
  return CriticalShape;

}



// SmallVector<int64_t> AffineLoopReorder::GetAccessCountsAtThisLevel(
//       AffineForOp ForLevel, SmallVector<Operation*> AccessOps){
//   SmallVector <int64_t> CriticalShape;
//   std::optional<int64_t> BiggestSize;  
//   MemRefRegion CriticalRegion(AccessOps[0]->getLoc());
//   for(Operation* AccessOp : AccessOps){
//     auto region = std::make_unique<MemRefRegion>(AccessOp->getLoc());
//     if (failed(region->compute(AccessOp,
//                   /*loopDepth=*/getNestingDepth(ForLevel)))) {
//       assert(0 && "[Error] error obtaining memory region\n");
//       // return AccessOp->emitError("error obtaining memory region\n");
//     }
//     SmallVector <int64_t> shape;
//     // llvm::errs() << "region:\n";
//     // region->dump();
//     region->getConstantBoundingSizeAndShape(/*shape=*/&shape/*, lbs= , lbDivisors=*/);
//     llvm::errs() << "op:"; AccessOp->dump();
//     llvm::errs() << "shape:";
//     for(auto d : shape){
//       llvm::errs() << d << " ";
//     }
//     llvm::errs() << ";\n";
//     std::optional<int64_t> size = region->getRegionSize();
//     llvm::errs() << "size: "<< size <<"\n";

//     if(size > BiggestSize){
//       BiggestSize = size;
//       CriticalShape = shape;
//     }
//   }
//   return CriticalShape;

// }

// Given a forOp, returns all the loop permutations which have lexicographically positive dependence vectors.
SmallVector<SmallVector<unsigned>> AffineLoopReorder::findValidLoopPermutations(AffineForOp forOp) {
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops,forOp);
  SmallVector<SmallVector<unsigned>> validLoopPerm;
  if (loops.size() < 2)
    return validLoopPerm;  
  unsigned maxLoopDepth = loops.size();
  unsigned arr[maxLoopDepth];
  for(unsigned i=0; i<maxLoopDepth; i++) {
    arr[i] = i;
  }
  SmallVector<unsigned> loopPermMap(maxLoopDepth);
  SmallVector<unsigned> loopPerm(maxLoopDepth);
  do {
    for (unsigned i = 0; i < maxLoopDepth; ++i) {
      loopPermMap[arr[i]] = i; // inverted, referred sinkSequentialLoops func
      loopPerm[i] = arr[i];    // not inverted
    }
    ArrayRef<AffineForOp> loops_arrayRef = llvm::ArrayRef(loops);
    ArrayRef<unsigned> loopPermMap_arrayRef = llvm::ArrayRef(loopPermMap);
    if ( isValidLoopInterchangePermutation(loops_arrayRef,loopPermMap_arrayRef) ) {
      // display(arr,maxLoopDepth);
      validLoopPerm.push_back(loopPerm); // not loopPermMap
    }
  } while(std::next_permutation(arr,arr+maxLoopDepth));
  return validLoopPerm;
}


SmallVector<std::pair<AffineLoadOp, AffineStoreOp>>
  AffineLoopReorder::getRelatedLoadStorePair(SmallVector<Operation*> ops){
  SmallVector<AffineLoadOp> loads;
  SmallVector<AffineStoreOp> stores;
  SmallVector<std::pair<AffineLoadOp, AffineStoreOp>> ls_pair;

  for(Operation* op : ops){
    if(isa<AffineLoadOp>(op))
      loads.push_back(dyn_cast<AffineLoadOp>(op));
    if(isa<AffineStoreOp>(op))
      stores.push_back(dyn_cast<AffineStoreOp>(op));
  }

  for(AffineLoadOp loadop : loads){
    /// Check whether this load occurs with a corresponding store which have
    /// the same memref and address to access. 
    for(AffineStoreOp storeop : stores){
      // llvm::errs() << "[info] loadop: " << loadop << "\n";   
      // llvm::errs() << "[info] storeop: " << storeop << "\n";    
      if(LoadStoreSameMemAddr(loadop, storeop)){
        ls_pair.push_back(std::pair(loadop, storeop));
      }
    }
  }

  return ls_pair;
}

template <typename LoadOrStoreOpPointer>
bool AffineLoopReorder::IsIndexOfAccess(AffineForOp for_tocheck,LoadOrStoreOpPointer accessop){
  Operation::operand_range Indices = accessop.getIndices();
  for(unsigned d = 0; d < Indices.size(); d++){
    // llvm::errs() << "[test] loadIndice[i]: " ; loadIndices[d].dump() ; 
    AffineForOp forop = dyn_cast<AffineForOp>(Indices[d].getParentBlock()->getParentOp());
    if(for_tocheck == forop){
      return true;
    }
  }
  
  return false;
}




std::unique_ptr<OperationPass<func::FuncOp>> mlir::ADORA::createAffineLoopReorderPass() {
  return std::make_unique<AffineLoopReorder>();
}

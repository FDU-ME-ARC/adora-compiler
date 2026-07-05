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

  LogicalResult ReorderOnAffineForOp(AffineForOp forOp);
  uint64_t ComputeMemoryAccessCost(
      ArrayRef<AffineForOp> loops,
      SmallDenseMap<unsigned, SmallVector<SmallVector<Operation *>>> &loopRefGroups,
      unsigned innerLevel);
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
          (void)ReorderOnAffineForOp(forOp);
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
/// Estimate the total number of distinct memory accesses when the loop at
/// `innerLevel` is placed as the innermost loop, following the reuse-group
/// cost model described in the comment above (steps 1-2).
/// For every reference group w.r.t. `innerLevel` a representative reference R
/// contributes, per innermost iteration:
///   - 1                        if R is loop-invariant (temporal reuse);
///   - ceildiv(trip, cacheLine) if R has spatial reuse (innermost index only
///                              appears in the last array dimension, stride 1);
///   - trip                     otherwise (strided, no reuse).
/// The per-iteration count is then multiplied by the product of the trip counts
/// of all the remaining (non-innermost) loops. A lower cost means the loop is a
/// better candidate for the innermost position.
uint64_t AffineLoopReorder::ComputeMemoryAccessCost(
    ArrayRef<AffineForOp> loops,
    SmallDenseMap<unsigned, SmallVector<SmallVector<Operation *>>> &loopRefGroups,
    unsigned innerLevel) {
  constexpr int64_t kCacheLineSize = 8;
  AffineForOp innerFor = loops[innerLevel];
  uint64_t innerTrip = getConstantTripCount(innerFor).value_or(1);

  uint64_t perIterAccesses = 0;
  for (const SmallVector<Operation *> &group : loopRefGroups[innerLevel]) {
    if (group.empty())
      continue;
    Operation *repOp = group.front();

    unsigned arrayRank = 0;
    std::pair<int, int> rankMult(-1, -1);
    if (auto load = dyn_cast<AffineLoadOp>(repOp)) {
      arrayRank = load.getMemRef().getType().cast<MemRefType>().getRank();
      rankMult = getCorrespondingRankAndMultiplicator(load, innerFor);
    } else if (auto store = dyn_cast<AffineStoreOp>(repOp)) {
      arrayRank = store.getMemRef().getType().cast<MemRefType>().getRank();
      rankMult = getCorrespondingRankAndMultiplicator(store, innerFor);
    } else {
      continue;
    }

    if (rankMult.first < 0) {
      // Innermost induction variable does not index this reference: reused.
      perIterAccesses += 1;
    } else if ((unsigned)rankMult.first == arrayRank - 1 && rankMult.second == 1) {
      // Contiguous access along the innermost loop: spatial reuse.
      perIterAccesses += (innerTrip + kCacheLineSize - 1) / kCacheLineSize;
    } else {
      // Strided access, no reuse.
      perIterAccesses += innerTrip;
    }
  }

  uint64_t remainingTrips = 1;
  for (unsigned l = 0; l < loops.size(); l++) {
    if (l == innerLevel)
      continue;
    remainingTrips *= getConstantTripCount(loops[l]).value_or(1);
  }
  return perIterAccesses * remainingTrips;
}

LogicalResult AffineLoopReorder::ReorderOnAffineForOp(AffineForOp forOp) {
  AffineForOp rootForOp = forOp;
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, rootForOp);
  unsigned loopDepth = loops.size();
  if(loopDepth == 1){
    return LogicalResult::success();
  }
  SmallVector<unsigned> loopPermMap(loopDepth), OriginloopPerm(loopDepth);

  // Reuse groups are keyed by loop depth on the original nest. The memory-access
  // cost of a loop as the innermost loop depends only on that loop's own access
  // pattern and the product of the other trip counts, both invariant to loop
  // ordering. So compute the analysis once and cache the cost per physical loop.
  SmallDenseMap<unsigned, SmallVector<SmallVector<Operation *>>> loop_refGroups =
      getReuseGroupsForEachLoop(rootForOp);
  SmallDenseMap<Operation *, uint64_t> costOf;
  for (unsigned d = 0; d < loopDepth; d++)
    costOf[loops[d].getOperation()] =
        ComputeMemoryAccessCost(loops, loop_refGroups, d);

  bool Interchangable = true;
  while(Interchangable){
    Interchangable = false;

    for (unsigned d=0;d < loopDepth - 1;d++) {
      loops.clear();
      getPerfectlyNestedLoops(loops, rootForOp);

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
      /***
       * Step1: check whether two loop can be interchanged
      */
      ArrayRef<AffineForOp> loops_arrayRef = llvm::ArrayRef(loops);
      ArrayRef<unsigned> loopPermMap_arrayRef = llvm::ArrayRef(loopPermMap);
      if ( isValidLoopInterchangePermutation(loops_arrayRef,loopPermMap_arrayRef) ) {
        /***
        * Decide the adjacent interchange with the reuse-group memory-access
        * cost model. ComputeMemoryAccessCost folds both temporal reuse (a
        * loop-invariant reference costs 1) and spatial reuse (a unit-stride
        * innermost reference costs trip/cacheLine) into a single per-loop key
        * that is invariant to loop ordering. Move the outer loop inward iff it
        * incurs strictly fewer memory accesses than its inner neighbour when
        * placed innermost; ties are left untouched so the bubbling converges.
        */
        uint64_t costOuterAsInner = costOf[OuterFor.getOperation()];
        uint64_t costInnerAsInner = costOf[InnerFor.getOperation()];
        LLVM_DEBUG(llvm::errs() << "cost[outer as inner]=" << costOuterAsInner
                                << " cost[inner as inner]=" << costInnerAsInner << "\n");
        if (costOuterAsInner < costInnerAsInner) {
          Interchangable = true;
          unsigned NewRootIndex = permuteLoops(loops, loopPermMap);
          rootForOp = loops[NewRootIndex];
          break; /// a swap happened, restart bubbling from the root
        }
      }
    }
  }

  return success();
}

std::unique_ptr<OperationPass<func::FuncOp>> mlir::ADORA::createAffineLoopReorderPass() {
  return std::make_unique<AffineLoopReorder>();
}

//===-  DependencyAnalysis.cpp -===//
/*****************************************************/
/************ Tool Functions for Dependency Analysis */
/*****************************************************/
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h" 
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"

#include "ADORA/Dialect/ADORA/Analysis/DependencyAnalysis.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

using namespace mlir::affine;

#define DEBUG_TYPE "dependency-analysis"

#define getMemrefFromOperation(op) isa<affine::AffineLoadOp>(op)? \
                                                 dyn_cast<affine::AffineLoadOp>(op).getMemref():\
                                                 dyn_cast<affine::AffineStoreOp>(op).getMemref()

#define getIndicesFromOperation(op) isa<affine::AffineLoadOp>(op)? \
                                                 dyn_cast<affine::AffineLoadOp>(op).getIndices():\
                                                 dyn_cast<affine::AffineStoreOp>(op).getIndices()

// Returns the number of outer loop common to 'src/dstDomain'.
// Loops common to 'src/dst' domains are added to 'commonLoops' if non-null.
static unsigned
getNumCommonLoops(const FlatAffineValueConstraints &srcDomain,
                  const FlatAffineValueConstraints &dstDomain,
                  SmallVectorImpl<AffineForOp> *commonLoops = nullptr) {
  // Find the number of common loops shared by src and dst accesses.
  unsigned minNumLoops =
      std::min(srcDomain.getNumDimVars(), dstDomain.getNumDimVars());
  unsigned numCommonLoops = 0;
  for (unsigned i = 0; i < minNumLoops; ++i) {
    if ((!isAffineForInductionVar(srcDomain.getValue(i)) &&
         !isAffineParallelInductionVar(srcDomain.getValue(i))) ||
        (!isAffineForInductionVar(dstDomain.getValue(i)) &&
         !isAffineParallelInductionVar(dstDomain.getValue(i))) ||
        srcDomain.getValue(i) != dstDomain.getValue(i))
      break;
    if (commonLoops != nullptr)
      commonLoops->push_back(getForInductionVarOwner(srcDomain.getValue(i)));
    ++numCommonLoops;
  }
  if (commonLoops != nullptr)
    assert(commonLoops->size() == numCommonLoops);
  return numCommonLoops;
}


bool equalMatrices(SmallVector<SmallVector<int>> srcAccessMatrix,SmallVector<SmallVector<int>> dstAccessMatrix) {
  int numRows = srcAccessMatrix.size();
  int numCols = srcAccessMatrix[0].size();
  if ((numRows != dstAccessMatrix.size()) || (numCols != dstAccessMatrix[0].size())) 
    return false;

  for( int r=0;r<numRows;r++) {
    for(int c=0;c<numCols;c++) {
      if (srcAccessMatrix[r][c] != dstAccessMatrix[r][c])
        return false;
    }
  }
  return true;
}

void printMatrix(SmallVector<SmallVector<int> > m){
  for(auto v : m){
      for(int e : v){
        llvm::errs() << e << "\t";
      }
    llvm::errs() << "\n";
  }
}

namespace mlir{
namespace ADORA{


// Adds ordering constraints to 'dependenceDomain' based on number of loops
// common to 'src/dstDomain' and requested 'loopDepth'.
// Note that 'loopDepth' cannot exceed the number of common loops plus one.
// EX: Given a loop nest of depth 2 with IVs 'i' and 'j':
// *) If 'loopDepth == 1' then one constraint is added: i' >= i + 1
// *) If 'loopDepth == 2' then two constraints are added: i == i' and j' > j + 1
// *) If 'loopDepth == 3' then two constraints are added: i == i' and j == j'
// copy from AffineAnalysis.cpp
void
addOrderingConstraints(const FlatAffineValueConstraints &srcDomain,
                       const FlatAffineValueConstraints &dstDomain,
                       unsigned loopDepth,
                       FlatAffineValueConstraints *dependenceDomain) {
  unsigned numCols = dependenceDomain->getNumCols();
  SmallVector<int64_t, 4> eq(numCols);
  unsigned numSrcDims = srcDomain.getNumDimVars();
  unsigned numCommonLoops = getNumCommonLoops(srcDomain, dstDomain);
  unsigned numCommonLoopConstraints = std::min(numCommonLoops, loopDepth);
  for (unsigned i = 0; i < numCommonLoopConstraints; ++i) {
    std::fill(eq.begin(), eq.end(), 0);
    eq[i] = -1;
    eq[i + numSrcDims] = 1;
    if (i == loopDepth - 1) {
      eq[numCols - 1] = -1;
      dependenceDomain->addInequality(eq);
    } else {
      dependenceDomain->addEquality(eq);
    }
  }
}



// For each loop l, compute groups of array references.
//        Two references ref1 and ref2 belong to same group with respect to loop
//        l if:
//            a. they refer to the same array and has exactly same access
//            function. b. or, they refer to the same array and differ only in
//            lth dimension by atmost cacheLineSize. c. or, they refer to the
//            same array and differ by at most cacheLineSize in the last
//            dimension.
// conditions (a) and (b) corresponds to group temporal reuse, (c) corresponds
// to group spatial reuse.
SmallDenseMap<unsigned, SmallVector<SmallVector<Operation* >> > getReuseGroupsForEachLoop(AffineForOp forOp) {
  // get all load and store operations
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  SmallDenseMap<Operation *, bool> visitedOp;
  forOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst)) {
      loadAndStoreOpInsts.push_back(opInst);
      visitedOp[opInst] = false;
    }
  });
  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);
  unsigned loopDepth = loops.size();

  // get groups of loadAndStoreOpInsts for each loop
  SmallDenseMap<unsigned, SmallVector<SmallVector<Operation* >> > loop_refGroups; 
  /* SmallDenseMap: key is loop (e.g.if loop  = i,j,k then key = 0 for i, key = 1 for j, key = 2 for k)
    value is collection of refGroups for the loop
  */
  for (unsigned d = 0; d < loopDepth; d++) {
    SmallVector<SmallVector<Operation* >> refGroups;
    unsigned numOps = loadAndStoreOpInsts.size();
    // mark all ops as unvisited;
    for (unsigned i = 0; i < numOps; i++) {
      visitedOp[loadAndStoreOpInsts[i]] = false;
    }
    for (unsigned i = 0; i < numOps; ++i) {
      auto *srcOpInst = loadAndStoreOpInsts[i];
      LLVM_DEBUG(llvm::errs() << "srcopinst:" << srcOpInst);
      if (visitedOp[srcOpInst]) continue; //already added to a group
      // create a group and mark visited
      visitedOp[srcOpInst] = true;
      SmallVector<Operation *> currGroup;
      currGroup.push_back(srcOpInst);
      Value srcArray; // src array name
      if (auto store = dyn_cast<AffineStoreOp>(srcOpInst)) {
        srcArray = srcOpInst->getOperand(1);
      }
      else if (auto load = dyn_cast<AffineLoadOp>(srcOpInst)) {
        srcArray = srcOpInst->getOperand(0);
      }
      for (unsigned j = 0; j < numOps; ++j) {
        auto *dstOpInst = loadAndStoreOpInsts[j];
        LLVM_DEBUG(llvm::errs() << "dstOpInst:"<< dstOpInst);
        if ((i == j) || visitedOp[dstOpInst] == true) {
          // same operation or already added to group
          continue;
        }
        Value dstArray; // dst array name
        if (auto store = dyn_cast<AffineStoreOp>(dstOpInst)) {
          dstArray = dstOpInst->getOperand(1);
        }
        else if (auto load = dyn_cast<AffineLoadOp>(dstOpInst)) {
          dstArray = dstOpInst->getOperand(0);
        }
        // srcArray.dump();
        // dstArray.dump();
        if (srcArray != dstArray) {
          continue;
        }
        else {
          // refer to same array and dstOpInst is not visited
          // check 1: they have same access matrix
          // check 2: check 1 && has group temporal reuse for loop d if they only differ in subscript having loop d by a small constant (< cache line size).
          // check 3: check 1 && has group spatial reuse if they differ in only last dimension
          // if check 2 or check 3 is satisfied, add this op to currGroup and mark it visited.
          SmallVector<SmallVector<int> > srcAccessMatrix = getAccessMatrix(srcOpInst);
          SmallVector<SmallVector<int> > dstAccessMatrix = getAccessMatrix(dstOpInst);
          printMatrix(srcAccessMatrix);
          printMatrix(dstAccessMatrix);
          if (equalMatrices(srcAccessMatrix, dstAccessMatrix)) {
            // group spatial reuse
            if (hasGroupSpatialReuse(srcOpInst, dstOpInst) || hasGroupTemporalReuse(srcOpInst, dstOpInst, d)) {
              // hasGroupSpatialReuse handles the case when array references are exactly same. (eg. A[i,j] and A[i,j])
              LLVM_DEBUG(llvm::errs() << "pushed!:" << "\n");
              currGroup.push_back(dstOpInst);
              visitedOp[dstOpInst] = true;
            }
          }
        }
      } 
      LLVM_DEBUG(llvm::errs() << "pushed!d:" << d << "\n");
      refGroups.push_back(currGroup);
    }
    loop_refGroups[d] = refGroups;
  }
  return loop_refGroups;
}

// TODO:write introduction to this.
template<typename srcT, typename dstT>
SmallDenseMap<srcT, SmallVector<dstT>> getSrctoDstDependency(AffineForOp& forOp) {
  SmallDenseMap<srcT, SmallVector<dstT>> srcToDependentdst;
  
  SmallVector<srcT, 8> srcInsts;
  SmallVector<dstT, 8> dstInsts;
  forOp.getOperation()->walk([&](srcT opInst) {
    srcInsts.push_back(opInst);
  });

  forOp.getOperation()->walk([&](dstT opInst) {
    dstInsts.push_back(opInst);
  });

  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);
  // unsigned loopDepth = getNestingDepth(forOp) + 1;
  // unsigned totalLoopLevels = getInnermostCommonLoopDepth(dstInsts);
  unsigned srcNumOps = srcInsts.size();
  unsigned dstNumOps = dstInsts.size();

  /// try
  // std::vector<SmallVector<DependenceComponent, 2>> *depCompsVec;
  // getDependenceComponents(forOp, loopDepth, depCompsVec);
  // for(auto depComps : *depCompsVec){
  //   llvm::errs() << "\nnew comp:\n";
  //   for(auto depComp : depComps){
  //     depComp.op -> dump();
  //     llvm::errs() << "lb:" << depComp.lb.value() 
  //                   << ", ub:" << depComp.ub.value() <<"\n";
  //   }
  // }

  // for (unsigned d = 0; d < totalLoopLevels; d++) {
    for (unsigned i = 0; i < srcNumOps; ++i) {
      srcT srcOpInst = srcInsts[i];
      LLVM_DEBUG(llvm::errs() << "srcopinst:" << srcOpInst);

      // SmallVector<Operation *> currGroup;
      // currGroup.push_back(srcOpInst);
      Value srcArray = srcOpInst.getMemref(); // src array name
      // if (auto store = dyn_cast<AffineStoreOp>(srcOpInst)) {
      //   srcArray = store.getMemref();
      // }
      // else if (auto load = dyn_cast<AffineLoadOp>(srcOpInst)) {
      //   srcArray = load.getMemref();
      // }
      for (unsigned j = 0; j < dstNumOps; ++j) {
        dstT dstOpInst = dstInsts[j];
        LLVM_DEBUG(llvm::errs() << "dstOpInst:" << dstOpInst);
        if ((srcOpInst == dstOpInst)) {
          // same operation
          continue;
        }
        ////////////
        // Check dependecy from outer most level to innermost level
        ////////////
        Value dstArray = dstOpInst.getMemref(); // dst array name
        // if (auto store = dyn_cast<AffineStoreOp>(srcOpInst)) {
        //   srcArray = store.getMemref();
        // }
        // else if (auto load = dyn_cast<AffineLoadOp>(srcOpInst)) {
        //   srcArray = load.getMemref();
        // }
        if (srcArray != dstArray) {
          continue;
        }
        else {
          ///// Only check whether related loop levels exist RAW
          bool inDomain = false;
          for (unsigned d = 0; d < loops.size(); d++) {
            AffineForOp dth_level = loops[d];
            dth_level.dump();

            // inDomain means that the current (and inner) loop level is within the domain relevant to the operation indices,
            // i.e., we have already reached the loop domain that requires dependency checking.
            // If not in the domain, there might be generated solutions representing fake (spurious) dependencies.
            // In other words, only after entering the relevant loop domain should dependencies be checked,
            // to avoid detecting false dependencies at unrelated loop levels.
            if(findElement(srcOpInst.getIndices(), dth_level.getInductionVar()) == NULL &&
               findElement(dstOpInst.getIndices(), dth_level.getInductionVar()) == NULL &&
               !inDomain){
              continue;
            }
            inDomain = true; 

            int depth = getNestingDepth(dth_level) + 1;
            
            MemRefAccess srcAccess(srcOpInst);
            MemRefAccess dstAccess(dstOpInst);
            // 
            // FlatAffineRelation srcRel, dstRel;
            // if (failed(srcAccess.getAccessRelation(srcRel)))
              // llvm::errs() << "Failure";
            // if (failed(dstAccess.getAccessRelation(dstRel)))
              // llvm::errs() << "Failure";
            // srcRel.dump();
            // dstRel.dump();
            // dstRel.inverse();
            // dstRel.dump();
            // dstRel.compose(srcRel);
            // dstRel.dump();

            // Add 'src' happens before 'dst' ordering constraints.
            // FlatAffineValueConstraints srcDomain = srcRel.getDomainSet();
            // FlatAffineValueConstraints dstDomain = dstRel.getDomainSet();

            // Add 'src' happens before 'dst' ordering constraints.
            // addOrderingConstraints(srcDomain, dstDomain, loopDepth + 1, &dstRel);
            // dstRel.dump();
            // 
            // if (dstRel.isEmpty()){
              // llvm::errs() << "Empty" << "\n";
            // }

            DependenceResult result =
              checkMemrefAccessDependence(srcAccess, dstAccess, depth);
            
            if(hasDependence(result)){
              srcToDependentdst[srcOpInst].push_back(dstOpInst);
            }
          }
        }
      } 
      // llvm::errs() << "pushed!d:" << d << "\n";
      // refGroups.push_back(currGroup);
    }
    // loop_refGroups[d] = refGroups;
  // }
  return srcToDependentdst;
}

//// WAR
SmallDenseMap<affine::AffineLoadOp, SmallVector<affine::AffineStoreOp>> 
                                getAffineLoadToStoreDependency(AffineForOp& forOp) {
  return getSrctoDstDependency<affine::AffineLoadOp, affine::AffineStoreOp>(forOp);
}

//// RAW
SmallDenseMap<affine::AffineStoreOp, SmallVector<affine::AffineLoadOp>> 
                                getAffineStoreToLoadDependency(AffineForOp& forOp) {
  return getSrctoDstDependency<affine::AffineStoreOp, affine::AffineLoadOp>(forOp);
}


/// @brief 
/// @param forOp  
/// @return a map, whose element is also a vector containing a group which will
///   access the same bank in pipeline.
SmallDenseMap<unsigned, SmallVector<Operation* >> 
                 getReuseGroupsForLoop(AffineForOp forOp) {
  // get all load and store operations
  SmallVector<Operation *, 8> loadAndStoreOpInsts;
  // SmallDenseMap<Operation *, bool> visitedOp;
  forOp.getOperation()->walk([&](Operation *opInst) {
    if (isa<AffineLoadOp>(opInst) || isa<AffineStoreOp>(opInst)) {
      loadAndStoreOpInsts.push_back(opInst);
      // visitedOp[opInst] = false;
    }
  });
  // SmallVector<AffineForOp, 4> loops;
  // getPerfectlyNestedLoops(loops, forOp);
  // unsigned loopDepth = loops.size();
  // unsigned loopDepth = getNestingDepth(forOp) + 1;

  SmallVector<AffineForOp, 4> loops;
  getPerfectlyNestedLoops(loops, forOp);

  // get groups of loadAndStoreOpInsts for each loop
  SmallDenseMap<unsigned, SmallVector<Operation* > > ReuseGroups;
  SmallDenseMap<Operation*, unsigned> OpToGroupNumber;
  // SmallDenseMap<unsigned, SmallVector<SmallVector<Operation* >> > loop_refGroups; 
  unsigned numOps = loadAndStoreOpInsts.size();
  // mark all ops as unvisited;
  for (unsigned i = 0, groupnum = 0; i < numOps; ++i) {
    mlir::Operation* srcOpInst = loadAndStoreOpInsts[i];
    LLVM_DEBUG(llvm::errs() << "srcOpInst:"; srcOpInst->dump());
    if(OpToGroupNumber.contains(srcOpInst))
      continue;
    mlir::Value srcArray = getMemrefFromOperation(srcOpInst);

    //// Don't have to check ops before src(lex)
    for (unsigned j = i; j < numOps; ++j) {
      mlir::Operation* dstOpInst = loadAndStoreOpInsts[j];
      if(OpToGroupNumber.contains(srcOpInst) || srcOpInst == dstOpInst)
        continue;
      LLVM_DEBUG(llvm::errs() << "|-> dstOpInst:"; dstOpInst->dump());
      mlir::Value dstArray = getMemrefFromOperation(dstOpInst);
      if (srcArray != dstArray) {
        continue;
      }
      else {
        MemRefAccess srcAccess(srcOpInst);
        MemRefAccess dstAccess(dstOpInst);
        FlatAffineValueConstraints *dependenceConstraints;
        // SmallVector<DependenceComponent, 2> *dependenceComponents;
        ///// Only check whether related loop levels exist RAW
        bool inDomain = false;
        for (unsigned d = 0; d < loops.size(); d++) {
          AffineForOp dth_level = loops[d];
          dth_level.dump();

          // inDomain means that the current (and inner) loop level is within the domain relevant to the operation indices,
          // i.e., we have already reached the loop domain that requires dependency checking.
          // If not in the domain, there might be generated solutions representing fake (spurious) dependencies.
          // In other words, only after entering the relevant loop domain should dependencies be checked,
          // to avoid detecting false dependencies at unrelated loop levels.
          if(findElement(getIndicesFromOperation(srcOpInst), dth_level.getInductionVar()) == NULL &&
              findElement(getIndicesFromOperation(dstOpInst), dth_level.getInductionVar()) == NULL &&
              !inDomain){
            continue;
          }
          inDomain = true; 

          int depth = getNestingDepth(dth_level) + 1;
          DependenceResult result =
            checkMemrefAccessDependence(srcAccess, dstAccess, depth
                                                /*,dependenceConstraints, dependenceComponents*/);
          if(hasDependence(result)){
            // dependenceConstraints->dump();
            if(!OpToGroupNumber.contains(srcOpInst)){
              OpToGroupNumber[srcOpInst] = groupnum;
              ReuseGroups[groupnum].push_back(srcOpInst);
              groupnum++;
            }
            //// get group count from src
            unsigned group_cnt =  OpToGroupNumber[srcOpInst];
            ReuseGroups[group_cnt].push_back(dstOpInst);
            OpToGroupNumber[dstOpInst] = group_cnt;
          }
        }
      } 
      // llvm::errs() << "pushed!d:" << d << "\n";
      // refGroups.push_back(currGroup);
    }
  }
  return ReuseGroups;
}



//////////////////////////////////
///// Dependency analysis for data block operations
//////////////////////////////////
static bool canonicalizeAndEqual(mlir::AffineMap map1, mlir::ValueRange ops1,
                                 mlir::AffineMap map2, mlir::ValueRange ops2) {
  llvm::SmallVector<mlir::Value, 8> v1(ops1.begin(), ops1.end());
  llvm::SmallVector<mlir::Value, 8> v2(ops2.begin(), ops2.end());

  // IMPORTANT: canonicalize the (map, operands) pair together.
  // Do NOT canonicalize only the map while comparing the original operands.
  simplifyMapWithOperands(map1, v1);
  simplifyMapWithOperands(map2, v2);

  if (map1 != map2) return false;
  if (v1.size() != v2.size()) return false;
  for (size_t i = 0; i < v1.size(); ++i)
    if (v1[i] != v2[i]) return false;
  return true;
}

static llvm::SmallVector<int64_t, 8>
getBlockShapeForLoad(ADORA::DataBlockLoadOp &op) {
  // For BlockLoad, the tile/block shape is the result memref shape.
  auto ty = op.getResult().getType().cast<mlir::MemRefType>();
  return llvm::SmallVector<int64_t, 8>(ty.getShape().begin(), ty.getShape().end());
}

static llvm::SmallVector<int64_t, 8>
getBlockShapeForStore(ADORA::DataBlockStoreOp &op) {
  // For BlockStore, the tile/block shape is typically the source memref shape
  // (the on-chip tile buffer being written back to the original array).
  auto ty = op.getSourceMemref().getType().cast<mlir::MemRefType>();
  return llvm::SmallVector<int64_t, 8>(ty.getShape().begin(), ty.getShape().end());
}

struct Interval {
  int64_t lb = 0;
  int64_t ub = -1; // inclusive
};

using Box = llvm::SmallVector<Interval, 8>;

/// Try to derive an axis-aligned rectangular region (box) with constant bounds.
/// Each dimension is represented as an inclusive interval [lb, ub].
/// Return false if any bound cannot be derived as a constant.
static bool tryGetConstantBoxFromMapAndShape(mlir::AffineMap map,
                                            mlir::ValueRange operands,
                                            llvm::ArrayRef<int64_t> shape,
                                            Box &outBox) {
  // Canonicalize (map, operands) together.
  llvm::SmallVector<mlir::Value, 8> ops(operands.begin(), operands.end());
  simplifyMapWithOperands(map, ops);

  outBox.clear();
  outBox.reserve(map.getNumResults());

  for (unsigned d = 0; d < map.getNumResults(); ++d) {
    mlir::AffineExpr e = map.getResult(d);

    // Extract constant from AffineExpr.
    auto cst = e.dyn_cast<mlir::AffineConstantExpr>();
    if (!cst)
      return false;

    // Be conservative if shape is missing or dynamic.
    if (d >= shape.size())
      return false;
    if (shape[d] == mlir::ShapedType::kDynamic)
      return false;

    int64_t lb = cst.getValue();
    int64_t ub = lb + shape[d] - 1;
    outBox.push_back(Interval{lb, ub});
  }
  return true;
}

/// Interval overlap check:
/// max(lb1, lb2) <= min(ub1, ub2)  <=>  lb1 <= ub2 && lb2 <= ub1
static bool intervalsOverlap(const Interval &a, const Interval &b) {
  return (a.lb <= b.ub) && (b.lb <= a.ub);
}

/// Two boxes overlap iff they overlap in every dimension.
/// If ranks are inconsistent, return true conservatively.
static bool boxesOverlap(const Box &a, const Box &b) {
  if (a.size() != b.size())
    return true;

  for (size_t d = 0; d < a.size(); ++d) {
    if (!intervalsOverlap(a[d], b[d]))
      return false;
  }
  return true;
}

//===----------------------------------------------------------------------===//
// DataBlock-level dependency — unified core + thin adapters.
//
// A DataBlock access is described by the quadruple (memref, map, operands,
// shape). Given two such quadruples, `mayOverlapBoxesByQuadruple` judges
// whether the two box regions may overlap. Every load-load / store-store /
// load-store / store-load overlap checker is a one-liner on top of this.
//
// Similarly, every exact-same / checker wrapper is a one-liner.
//===----------------------------------------------------------------------===//

/// Core overlap predicate — zero knowledge of LoadOp / StoreOp types.
/// Returns:
///   false  — provably disjoint (different memrefs, or disjoint const boxes)
///   true   — possibly overlapping (same memref AND (overlapping const boxes
///            OR at least one side could not be folded to a constant box))
static bool mayOverlapBoxesByQuadruple(mlir::Value memA,
                                       mlir::AffineMap mapA,
                                       mlir::ValueRange opsA,
                                       llvm::ArrayRef<int64_t> shapeA,
                                       mlir::Value memB,
                                       mlir::AffineMap mapB,
                                       mlir::ValueRange opsB,
                                       llvm::ArrayRef<int64_t> shapeB) {
  if (memA != memB)
    return false;
  Box boxA, boxB;
  if (!tryGetConstantBoxFromMapAndShape(mapA, opsA, shapeA, boxA))
    return true; // conservative
  if (!tryGetConstantBoxFromMapAndShape(mapB, opsB, shapeB, boxB))
    return true;
  return boxesOverlap(boxA, boxB);
}

// NOTE: non-const refs because generated op getters are not const-qualified.
static bool mayOverlapDataBlockRegion(ADORA::DataBlockStoreOp &s,
                                      ADORA::DataBlockLoadOp  &l) {
  return mayOverlapBoxesByQuadruple(
      s.getTargetMemref(),  s.getAffineMap(), s.getMapOperands(), getBlockShapeForStore(s),
      l.getOriginalMemref(), l.getAffineMap(), l.getMapOperands(), getBlockShapeForLoad(l));
}

static bool mayOverlapDataBlockRegion(ADORA::DataBlockLoadOp &a,
                                      ADORA::DataBlockLoadOp &b) {
  return mayOverlapBoxesByQuadruple(
      a.getOriginalMemref(), a.getAffineMap(), a.getMapOperands(), getBlockShapeForLoad(a),
      b.getOriginalMemref(), b.getAffineMap(), b.getMapOperands(), getBlockShapeForLoad(b));
}

static bool mayOverlapDataBlockRegion(ADORA::DataBlockStoreOp &a,
                                      ADORA::DataBlockStoreOp &b) {
  return mayOverlapBoxesByQuadruple(
      a.getTargetMemref(), a.getAffineMap(), a.getMapOperands(), getBlockShapeForStore(a),
      b.getTargetMemref(), b.getAffineMap(), b.getMapOperands(), getBlockShapeForStore(b));
}

static bool mayOverlapDataBlockRegion(ADORA::DataBlockLoadOp  &l,
                                      ADORA::DataBlockStoreOp &s) {
  return mayOverlapBoxesByQuadruple(
      l.getOriginalMemref(), l.getAffineMap(), l.getMapOperands(), getBlockShapeForLoad(l),
      s.getTargetMemref(),   s.getAffineMap(), s.getMapOperands(), getBlockShapeForStore(s));
}

//===----------------------------------------------------------------------===//
// Exact-same-block predicates. Memref equality + canonicalized map equality.
//===----------------------------------------------------------------------===//
bool AccessSameDataBlock(ADORA::DataBlockLoadOp &a, ADORA::DataBlockLoadOp &b) {
  return a.getOriginalMemref() == b.getOriginalMemref() &&
         canonicalizeAndEqual(a.getAffineMap(), a.getMapOperands(),
                              b.getAffineMap(), b.getMapOperands());
}

bool AccessSameDataBlock(ADORA::DataBlockStoreOp &s, ADORA::DataBlockLoadOp &l) {
  return s.getTargetMemref() == l.getOriginalMemref() &&
         canonicalizeAndEqual(s.getAffineMap(), s.getMapOperands(),
                              l.getAffineMap(), l.getMapOperands());
}

bool AccessSameDataBlock(ADORA::DataBlockStoreOp &a, ADORA::DataBlockStoreOp &b) {
  return a.getTargetMemref() == b.getTargetMemref() &&
         canonicalizeAndEqual(a.getAffineMap(), a.getMapOperands(),
                              b.getAffineMap(), b.getMapOperands());
}

//===----------------------------------------------------------------------===//
// Checker wrappers. All four follow the same policy:
//   dependent iff (exact-same block) OR (conservative overlap)
// Both sub-predicates already short-circuit on memref-inequality, so the
// wrapper is pure disjunction.
//===----------------------------------------------------------------------===//
bool checkDependencyBetweenBlockStoreAndBlockLoad(ADORA::DataBlockStoreOp &s,
                                                  ADORA::DataBlockLoadOp  &l) {
  return AccessSameDataBlock(s, l) || mayOverlapDataBlockRegion(s, l);
}

bool checkDependencyBetweenBlockLoadAndBlockLoad(ADORA::DataBlockLoadOp &a,
                                                 ADORA::DataBlockLoadOp &b) {
  return AccessSameDataBlock(a, b) || mayOverlapDataBlockRegion(a, b);
}

bool checkDependencyBetweenBlockStoreAndBlockStore(ADORA::DataBlockStoreOp &a,
                                                   ADORA::DataBlockStoreOp &b) {
  return AccessSameDataBlock(a, b) || mayOverlapDataBlockRegion(a, b);
}

bool checkDependencyBetweenBlockLoadAndBlockStore(ADORA::DataBlockLoadOp  &l,
                                                  ADORA::DataBlockStoreOp &s) {
  // Reuse the (store, load) exact-same predicate — it's argument-symmetric.
  return AccessSameDataBlock(s, l) || mayOverlapDataBlockRegion(l, s);
}

} // namespace
}
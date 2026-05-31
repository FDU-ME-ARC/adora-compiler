//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORA_DEPENDECYANALYSIS_H_
#define CGRAOPT_DIALECT_ADORA_DEPENDECYANALYSIS_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h" 
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/LoopUtils.h"


#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

// #include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/DenseMap.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
// #include "llvm/ADT/SmallSet.h" /// use std::unordered_set instead of std::list
#include<set>

//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//
using namespace llvm; // for llvm.errs()
using namespace llvm::detail;
using namespace mlir::affine;



namespace mlir {
namespace ADORA {
// template <typename LoadOrStoreOpPointer>
//     SmallVector<SmallVector<int>> getAccessMatrix(LoadOrStoreOpPointer memoryOp);

SmallDenseMap<unsigned, SmallVector<SmallVector<Operation* >> > getReuseGroupsForEachLoop(AffineForOp forOp);
// template <typename LoadOrStoreOpPointer>
//     bool hasGroupSpatialReuse(LoadOrStoreOpPointer srcOpInst, LoadOrStoreOpPointer dstOpInst);
// template <typename LoadOrStoreOpPointer>
//     bool hasGroupTemporalReuse(LoadOrStoreOpPointer srcOpInst, LoadOrStoreOpPointer dstOpInst, unsigned loop);
// template <typename LoadOrStoreOpPointer>
//     std::pair<unsigned, unsigned> getCorrespondingRankAndMultiplicator (LoadOrStoreOpPointer memoryOp, AffineForOp for_tocheck);
template <typename LoadOrStoreOpPointer>
bool hasGroupSpatialReuse(LoadOrStoreOpPointer srcOpInst, LoadOrStoreOpPointer dstOpInst) {
  // int cacheLineSize = 8;
  // FlatAffineConstraints accessConstraints;
  // getAccessConstraints(srcOpInst, dstOpInst, &accessConstraints);
  // accessConstraints.dump();
  MemRefAccess SrcAccess(srcOpInst);
  MemRefAccess DstAccess(dstOpInst);
  FlatAffineValueConstraints accessConstraints;
  unsigned loopDepth = getNestingDepth(srcOpInst); // num of columns
  assert(loopDepth == getNestingDepth(dstOpInst));
  checkMemrefAccessDependence(SrcAccess, DstAccess, loopDepth, &accessConstraints,/*dependenceComponents=*/nullptr);
  // getAccessConstraints(srcOpInst, dstOpInst, &accessConstraints);
  // srcOpInst->dump(); //print
  // dstOpInst->dump();
  // accessConstraints.dump();

  unsigned numCols = accessConstraints.getNumCols();
  unsigned arrayDimension; // num of rows
  if (auto load = dyn_cast<AffineLoadOp>(srcOpInst)) {
    auto memRefType = load.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  else if (auto store = dyn_cast<AffineStoreOp>(srcOpInst)) {
    auto memRefType = store.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  // true if constant term differs in only last dimension
  for(unsigned r=0;r<arrayDimension;r++) {
    if ((r < (arrayDimension-1)) && int64_t(accessConstraints.atEq(r,numCols-1)) != 0)
      return false;
          // TODO: No regard of cacheLineSize
    // if ( (r == (arrayDimension-1)) && accessConstraints.atEq(r,numCols-1) < cacheLineSize)
    // if ( (r == (arrayDimension-1)) /*&& accessConstraints.atEq(r,numCols-1) < cacheLineSize*/)
    //   return true;
  }
  return true; // Check this.
  // return false;
}

template <typename LoadOrStoreOpPointer>
bool hasGroupTemporalReuse(LoadOrStoreOpPointer srcOpInst, LoadOrStoreOpPointer dstOpInst, unsigned loop) {
  // int cacheLineSize = 8;
  MemRefAccess SrcAccess(srcOpInst);
  MemRefAccess DstAccess(dstOpInst);
  FlatAffineValueConstraints accessConstraints;
  unsigned loopDepth = getNestingDepth(srcOpInst); // num of columns
  assert(loopDepth == getNestingDepth(dstOpInst));
  checkMemrefAccessDependence(SrcAccess, DstAccess, loopDepth, &accessConstraints,/*dependenceComponents=*/nullptr);
  // getAccessConstraints(srcOpInst, dstOpInst, &accessConstraints);
  // srcOpInst->dump(); //print
  // dstOpInst->dump();
  // accessConstraints.dump();
  unsigned numCols = accessConstraints.getNumCols();
  unsigned arrayDimension; // num of rows
  if (auto load = dyn_cast<AffineLoadOp>(srcOpInst)) {
    auto memRefType = load.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  else if (auto store = dyn_cast<AffineStoreOp>(srcOpInst)) {
    auto memRefType = store.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  // find array dimension which is not invariant to loop: 
  // these are the rows of access Matrix which has non-zero entries in loop column.

  // unsigned loopDepth = getNestingDepth(srcOpInst); // num of columns
  SmallVector< SmallVector<int> > accessMatrix;
  for (unsigned p=0;p<arrayDimension;p++) {
    SmallVector<int> tmp;
    for (unsigned q=0;q<loopDepth;q++) {
          tmp.push_back(int64_t(accessConstraints.atEq(p,q)));
    }
    accessMatrix.push_back(tmp);
  }

  SmallVector<int> loopVariantDims(arrayDimension,0);
  for (unsigned r=0;r<arrayDimension;r++) {
    if (accessMatrix[r][loop] != 0) 
      loopVariantDims[r] = 1;
  }

  // since access matrices are same, return true if constant terms differ in only loopVariantDims.
  for(unsigned r=0;r<arrayDimension;r++) {
    if (loopVariantDims[r] == 0 && accessConstraints.atEq(r,numCols-1) != 0)
      return false;
      //TODO:
    // if (loopVariantDims[r] == 1 && accessConstraints.atEq(r,numCols-1) > cacheLineSize)
    //   return false;
  }
  return true;
}

template <typename LoadOrStoreOpPointer>
SmallVector<SmallVector<int>> getAccessMatrix(LoadOrStoreOpPointer memoryOp) {
  MemRefAccess Access(memoryOp);
  FlatAffineRelation Rel;
  if (failed(Access.getAccessRelation(Rel)))
    assert(0 && "Should not run here.");
  unsigned loopDepth = getNestingDepth(memoryOp); // num of columns
  // checkMemrefAccessDependence(Access, Access, loopDepth, &accessConstraints,/*dependenceComponents=*/nullptr);
  FlatAffineValueConstraints accessConstraints = Rel.getDomainSet();
  // getAccessConstraints(memoryOp, memoryOp, &accessConstraints);
  // memoryOp->dump();
  // accessConstraints.dump(); // printing

  unsigned arrayDimension; // num of rows
  auto load = dyn_cast<AffineLoadOp>(memoryOp);
  auto store = dyn_cast<AffineStoreOp>(memoryOp);
  if (load) {
    auto memRefType = load.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  else if (store) {
    auto memRefType = store.getMemRef().getType().template cast<MemRefType> ();
    arrayDimension = memRefType.getRank();
  }
  // llvm::outs() << "AccessMatrix:\n";
  SmallVector< SmallVector<int> > accessMatrix;
  for (unsigned p=0;p<arrayDimension;p++) {
    SmallVector<int> tmp;
    for (unsigned q=0;q<loopDepth;q++) {
          tmp.push_back(int64_t(accessConstraints.atEq(p,q)));
          // llvm::outs() << accessConstraints.atEq(p,q) << " ";
    }
    accessMatrix.push_back(tmp);
    // llvm::outs() << "\n";
  }
  return accessMatrix;
}


/// @brief return a pair which contains corresponding <rank, multiplicator> in the memory op,
///        for the forop. 
/// @param memoryOp AffineLoadOp or AffineStoreOp 
/// @param for_tocheck 
/// @return std::pair<unsigned, unsigned> : <rank, multiplicator>. If it is not valid, return (-1,-1)
template <typename LoadOrStoreOpPointer>
std::pair<unsigned, unsigned> getCorrespondingRankAndMultiplicator
(LoadOrStoreOpPointer memoryOp, AffineForOp for_tocheck){
  Operation::operand_range Indices = memoryOp.getIndices();
  unsigned d;
  for(d = 0; d < Indices.size(); d++){
    // llvm::errs() << "[test] loadIndice[i]: " ; loadIndices[d].dump() ; 
    AffineForOp forop = dyn_cast<AffineForOp>(Indices[d].getParentBlock()->getParentOp());
    if(for_tocheck == forop){
      break;
    }
  }
  int r, multiplicator;
  AffineMap map = memoryOp.getAffineMapAttr().getValue();
  for(int r = map.getResults().size() - 1; r >= 0; r--){
    AffineExpr expr = map.getResult(r);
    if(expr.isFunctionOfDim(d)){
      multiplicator = ADORA::MultiplicatorOfDim(expr, d);
      return std::pair(r, multiplicator);
    }
  }
  
  return std::pair(-1, -1);
}


void
addOrderingConstraints(const FlatAffineValueConstraints &srcDomain,
                       const FlatAffineValueConstraints &dstDomain,
                       unsigned loopDepth,
                       FlatAffineValueConstraints *dependenceDomain);

//// WAR
SmallDenseMap<affine::AffineLoadOp, SmallVector<affine::AffineStoreOp>> 
                                getAffineLoadToStoreDependency(AffineForOp& forOp);

//// RAW
SmallDenseMap<affine::AffineStoreOp, SmallVector<affine::AffineLoadOp>> 
                                getAffineStoreToLoadDependency(AffineForOp& forOp);
                                
SmallDenseMap<unsigned, SmallVector<Operation* > >
                 getReuseGroupsForLoop(AffineForOp forOp);


//////////////////////////////////
///// Dependency analysis for data block operations
//////////////////////////////////
bool checkDependencyBetweenBlockStoreAndBlockLoad(ADORA::DataBlockStoreOp& store, ADORA::DataBlockLoadOp& load) ;

bool AccessSameDataBlock(ADORA::DataBlockLoadOp& op1, ADORA::DataBlockLoadOp& op2); // load-load
bool AccessSameDataBlock(ADORA::DataBlockStoreOp& op1, ADORA::DataBlockLoadOp& op2); // store-load
bool AccessSameDataBlock(ADORA::DataBlockStoreOp& op1, ADORA::DataBlockStoreOp& op2); // store-store (P1.0)

// P1.0 — DataBlock-level RAR / WAW / WAR dependency checkers.
// Semantics: same-backing-memref && (exact-same block || constant-box overlap).
// Callers own the lexical-order check when direction matters (WAR).
bool checkDependencyBetweenBlockLoadAndBlockLoad(
    ADORA::DataBlockLoadOp& a,  ADORA::DataBlockLoadOp& b);
bool checkDependencyBetweenBlockStoreAndBlockStore(
    ADORA::DataBlockStoreOp& a, ADORA::DataBlockStoreOp& b);
bool checkDependencyBetweenBlockLoadAndBlockStore(
    ADORA::DataBlockLoadOp&  load,
    ADORA::DataBlockStoreOp& store);

} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_DEPENDECYANALYSIS_H_

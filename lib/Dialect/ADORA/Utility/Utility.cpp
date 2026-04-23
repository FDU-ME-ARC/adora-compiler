//===- Utility.cpp - Some utility tools -----------===//
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/AffineExprVisitor.h"

// #include "mlir/IR/BlockAndValueMapping.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Twine.h"


#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>

#include "../../../DFG/inc/mlir_cdfg.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/DSE.h"
#include "ADORA/Misc/DFG.h"



using namespace llvm; // for llvm.errs()
// using namespace llvm::detail;
using namespace mlir;
using namespace mlir::ADORA;

//===----------------------------------------------------------------------===//
// eraseKERNEL
//===----------------------------------------------------------------------===//

mlir::Operation* mlir::ADORA::eraseKernel(func::FuncOp& TopFunc, ADORA::KernelOp& Kernel){
  /// return the corresponding first op of kernel
  mlir::Operation* return_op;
  bool IsFirstOp = true;
  /// traverse every operation in TopFunc
  // errs()<<"TopFunc :\n";TopFunc.dump();
  TopFunc.walk([&](mlir::Operation* op){
    // errs()<<"op :" << op->getName().getStringRef() << "\n";
    if(op->getName().getStringRef()== ADORA::KernelOp::getOperationName()){
      /// Found a kernel op
      ADORA::KernelOp Kernel_captured = dyn_cast<ADORA::KernelOp>(op);
      if(Kernel_captured == Kernel){
        /// Found the kernel we need to handle
        /// traverse every op of every block
        for(auto blk_itr=Kernel.getBody().begin(); blk_itr!=Kernel.getBody().end(); blk_itr++){
          for(auto op_itr=(*blk_itr).begin(); op_itr!=(*blk_itr).end(); op_itr++){
            if(op_itr->getName().getStringRef()== ADORA::TerminatorOp::getOperationName()){
              /// "ADORA.terminator" do not need to be replicated
              continue;
            }
            /// clone a new op and move it to the following loc of "ADORA.kernel"
            mlir::Operation* newop = (*op_itr).clone();
            blk_itr->push_back(newop);
            newop->moveBefore(op);
            op->replaceAllUsesWith(newop);
            /// return the 1st newop
            if(IsFirstOp){
              return_op = newop;
              IsFirstOp = false;
            }
          }
        }
      }
    }
  });
  /// erase "ADORA.kernel"
  Kernel.erase();
  /// return the corresponding first op of kernel after erase
  return return_op;
}
 



//===----------------------------------------------------------------------===//
// getConstPartofAffineExpr(Is associated with Affine Dialect)
//===----------------------------------------------------------------------===//
/// @brief 
/// @param expr 
/// @return the constant part of this AffineExpr
AffineExpr mlir::ADORA::getConstPartofAffineExpr(AffineExpr& expr){
  AffineExpr constantpart;
  switch (expr.getKind())
  {
  case AffineExprKind::DimId :
    constantpart = getAffineConstantExpr(0, expr.getContext());
    break;

  case AffineExprKind::Add :
    constantpart = expr.dyn_cast<AffineBinaryOpExpr>().getRHS();
    // llvm::errs() << "[debug] const:"<< constantpart <<"\n"; 
    break;
  
  default:
    assert(0 && "We just support 2 kinds of expr: DimId, ADD right now!");
    break;
  }
  return constantpart;
}


//===----------------------------------------------------------------------===//
// MultiplicatorOfDim
//===----------------------------------------------------------------------===//

signed mlir::ADORA::MultiplicatorOfDim(const AffineExpr& expr, const unsigned dim){
  if(!expr.isFunctionOfDim(dim))
    return 0;

  // llvm::outs() << "expr: " << expr << "\n";
  signed Multiplicator = 0;
  expr.walk([&](AffineExpr subExpr){
    // llvm::outs() << "Sub-expression: " << subExpr << "\n";
    if(subExpr.isFunctionOfDim(dim)){
      switch (subExpr.getKind())
      {
      case AffineExprKind::DimId : // d1
        if(Multiplicator==0)
          Multiplicator = 1;
        break;

      case AffineExprKind::Mul : // d1 * 2 or 2 * d1
        if(subExpr.dyn_cast<AffineBinaryOpExpr>().getLHS()==getAffineDimExpr(dim, expr.getContext()) 
        && subExpr.dyn_cast<AffineBinaryOpExpr>().getRHS().getKind() == AffineExprKind::Constant)
        {/// if LHS of this subexpr is d1, then the multiplicattor is LHS
          
          Multiplicator = subExpr.dyn_cast<AffineBinaryOpExpr>().getRHS()
                            .dyn_cast<AffineConstantExpr>().getValue();
        }
        // constantpart = expr.dyn_cast<AffineBinaryOpExpr>().getRHS();
        // llvm::errs() << "[debug] const:"<< constantpart <<"\n"; 
        break;
  
      default:
        // assert(0 && "We just support 2 kinds of expr: DimId, ADD right now!");
        break;
      }
      // llvm::errs() << "Multiplicator of dim "<< dim << " :"  << Multiplicator << "\n";
    }
    return WalkResult::advance(); 
  });

  return Multiplicator;
}
//===----------------------------------------------------------------------===//
// removeUnusedRegionArgs
//===----------------------------------------------------------------------===//
// void mlir::ADORA::removeUnusedRegionArgs(Region &region){
//   SmallVector<BlockArgument, 4> args(region.getArguments().begin(),
//                                      region.getArguments().end());
//   for (auto arg : args) {
//     // Check if the argument is used in the region
//     llvm::errs() << "[debug] arg:" << arg <<"\n";
//     bool used = false;
//     region.walk([&](Operation* op) {
//       llvm::errs() << "[debug] op:" << *op <<"\n";
//       for (auto operand : op->getOperands()) {
//         llvm::errs() << "[debug] operand:" << operand <<"\n";
//         if (operand == arg){
//           used = true;
//         }
//       }
//     });
//     if (!used)
//       region.eraseArgument(arg.getArgNumber());
//   }
// }


//===----------------------------------------------------------------------===//
// removeUnusedRegionArgs
//===----------------------------------------------------------------------===//
// This function takes a `loadOp` or `storeOp` and eliminates unused indices in its
// index list.
void mlir::ADORA::eliminateUnusedIndices(Operation *op) {
  /// TODO: big bug here. We assume that affine map is like :(d0,d1,d2,d3) -> (d0,d1,d2,d3).
  ///     But what if affine is like : (d0,d1,d2) -> (0,0,d1,d2) ?
  // Get the affine map for the operation.
  AffineMap map;
  ValueRange mapOperands;
  ::llvm::ArrayRef<int64_t> memrefShape;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)){
    map = loadOp.getAffineMap();
    mapOperands = loadOp.getIndices();
    memrefShape = loadOp.getMemref().getType().getShape();
  }
  else if (auto storeOp = dyn_cast<AffineStoreOp>(op)){
    map = storeOp.getAffineMap();
    mapOperands = storeOp.getIndices();
    memrefShape = storeOp.getMemref().getType().getShape();
  }
  else
    assert("Operation to eliminate unused indices should be AffineStoreOp or AffineLoadOp.");
  
  // for some special affine map, such as: (d0, d1, d2) -> (0, 0, d1, d2)
  if(memrefShape.size() != mapOperands.size()){
    return;
  }

  llvm::errs() << "[debug] op:" << *op <<"\n";
  // assert(memrefShape.size() == mapOperands.size());

  llvm::errs() << "[debug] map:" << map <<"\n";
  // llvm::errs() << "[debug] map.getNumInputs():" << map.getNumInputs() <<"\n";
  // llvm::errs() << "[debug] mapOperands.size():" << mapOperands.size() <<"\n";
  // If one dim of the shape is 1, then set the index of this dim to be constant index 0]
  SmallVector<AffineExpr, 4> dimReplacements(memrefShape.size());
  unsigned j = 0;
  for (unsigned dim = 0; dim < memrefShape.size(); dim++) {
    if(memrefShape[dim] == 1){
      dimReplacements[dim] = getAffineConstantExpr(0, map.getContext());
    }
    else{
      dimReplacements[dim] = getAffineDimExpr(dim, map.getContext());
    }
    llvm::errs() << "[debug] dimReplacements:" << dimReplacements[dim] <<"\n";
  }

  
  // auto newMap = AffineMap::get(newIndexList.size(), map.getNumSymbols(), map.getResults(), op->getContext());
  map = map.replaceDimsAndSymbols(dimReplacements, {}, map.getNumDims(), map.getNumSymbols());

  // Get the operands for the operation.
  // Find which indices are used by checking which dimensions appear in the affine map.
  SmallVector<bool, 8> usedIndices(mapOperands.size(), false);
  llvm::errs() << "[debug] map:" << map <<"\n";
  // llvm::errs() << "[debug] map.getNumInputs():" << map.getNumInputs() <<"\n";
  // llvm::errs() << "[debug] mapOperands.size():" << mapOperands.size() <<"\n";
  
  assert(map.getNumInputs() == mapOperands.size() && "map.getNumInputs() should be equal to operands.size().");
  for (unsigned input = 0; input < map.getNumInputs(); ++input) {
    unsigned index;
    auto results = map.getResults();
    for( index = 0; index < results.size(); index++){
      AffineExpr result = results[index];
      llvm::errs() << "expr result:" << result << "\n";
      if(result.isFunctionOfDim(input))
        break;
    }
    if(index == results.size()){
      /// This input is not a operand of the load
      usedIndices[input] = false;
    }
    else{
      usedIndices[input] = true;
    }
  }

  SmallVector<mlir::Value, 4> newIndexList;
  SmallVector<AffineExpr, 4> dimReplacements2(map.getNumDims());
  j = 0;
  for (unsigned i = 0; i < mapOperands.size(); ++i) {
    if (usedIndices[i]){
      newIndexList.push_back(mapOperands[i]);
      dimReplacements2[i] = getAffineDimExpr(j++, map.getContext());
    }
    else{
      dimReplacements2[i] = getAffineConstantExpr(0, map.getContext());
    }
  }

  // auto newMap = AffineMap::get(newIndexList.size(), map.getNumSymbols(), map.getResults(), op->getContext());
  map = map.replaceDimsAndSymbols(dimReplacements2, {}, j, map.getNumSymbols());
  // llvm::errs() << "[debug] map after replace:" << map <<"\n";
  // llvm::errs() << "[debug] map:" << map <<"\n";
  // llvm::errs() << "[debug] map.getNumInputs():" << map.getNumInputs() <<"\n";
  // llvm::errs() << "[debug] mapOperands.size():" << mapOperands.size() <<"\n";
  // Set the new affine map and index list for the operation.
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)){
    newIndexList.insert(newIndexList.begin(),loadOp.getMemref());
    loadOp.getOperation()->setAttr(AffineLoadOp::getMapAttrStrName(),AffineMapAttr::get(map));  
    loadOp.getOperation()->setOperands(newIndexList);
  }

  else if (auto storeOp = dyn_cast<AffineStoreOp>(op)){
    newIndexList.insert(newIndexList.begin(),storeOp.getMemref());
    newIndexList.insert(newIndexList.begin(),storeOp.getValue());
    storeOp.getOperation()->setAttr(AffineStoreOp::getMapAttrStrName(),AffineMapAttr::get(map)); 
    storeOp.getOperation()->setOperands(newIndexList);  
  }
}



//===----------------------------------------------------------------------===//
// getOperandInRank
//===----------------------------------------------------------------------===//

/// @brief return the AffineMap operand used in the rank 
/// @param op AffineLoadOp or AffineStoreOp or ADORA.BlockLoadOp
/// @param rank the result rank of the affine map 
/// @return SmallDenseMap<mlir::Value, unsigned> : < implemented arguments of dim, dim>
SmallDenseMap<mlir::Value, unsigned> mlir::ADORA::getOperandInRank(Operation* op, unsigned rank){
  // Get the affine map for the operation.
  AffineMap map;
  ValueRange mapOperands;
  SmallDenseMap<mlir::Value, unsigned> usedOperands;
  if (auto loadOp = dyn_cast<AffineLoadOp>(op)){
    map = loadOp.getAffineMap();
    mapOperands = loadOp.getIndices();
  }
  else if (auto storeOp = dyn_cast<AffineStoreOp>(op)){
    map = storeOp.getAffineMap();
    mapOperands = storeOp.getIndices();
  }
  else if (auto BlockLoad = dyn_cast<ADORA::DataBlockLoadOp>(op)){
    map = BlockLoad.getAffineMap();
    mapOperands = BlockLoad.getIndices();
  }
  else if (auto BlockStore = dyn_cast<ADORA::DataBlockStoreOp>(op)){
    map = BlockStore.getAffineMap();
    mapOperands = BlockStore.getIndices();
  }
  else
    assert("Operation should be AffineStoreOp, AffineLoadOp or BlockLoadOp.");

  // Find which indices are used by checking which dimensions appear in the affine map.
  SmallVector<bool, 8> usedIndices(mapOperands.size(), false);
  llvm::errs() << "[debug] op:" << *op <<"\n";
  llvm::errs() << "[debug] map:" << map <<"\n";
  
  assert(map.getNumInputs() == mapOperands.size() && "map.getNumInputs() should be equal to operands.size().");
  auto result = map.getResult(rank);
  llvm::errs() << "[debug] result:" << result <<"\n";
  for (unsigned input = 0; input < map.getNumInputs(); ++input) {
    if(result.isFunctionOfDim(input)){
      // usedOperands.push_back(std::move(mapOperands[input]));
      usedOperands[std::move(mapOperands[input])] = input;
      llvm::errs() << "[debug] mapOperands[input]:" << mapOperands[input] << ",input:" << input <<"\n";
    }
  }

  return usedOperands;
}

/// @brief get all uses of beused in block
/// @param beused 
/// @param region 
/// @return all uses
SmallVector<mlir::Operation*> mlir::ADORA::getAllUsesInRegion(const mlir::Value beused, ::mlir::Region* region){
  SmallVector<mlir::Operation*> used;
  region->walk([&](mlir::Operation* op){
    for(mlir::Value operand : op->getOperands()){
      if(operand == beused){
        used.push_back(op);
        break;
      }
    }
  });
  return used;
}

/// @brief get all uses of beused in block
/// @param beused 
/// @param block 
/// @return all uses
SmallVector<mlir::Operation*> mlir::ADORA::getAllUsesInBlock(const mlir::Value beused, ::mlir::Block* block){
  SmallVector<mlir::Operation*> used;
  block->walk([&](mlir::Operation* op){
    // op->dump();
    for(mlir::Value operand : op->getOperands()){
      if(operand == beused){
        used.push_back(op);
        break;
      }
    }
  });
  return used;
}

//===----------------------------------------------------------------------===//
// Class ADORA::ForNode
//===----------------------------------------------------------------------===//
/// public function for class ADORA::ForNode
bool ADORA::ForNode::IsInnermost(){
  /// Check whether another for Op exists inside this forop's region
  // llvm::errs() << "[debug] forop:\n";  
  // ForOp.dump();
  auto WalkResult = ForOp.getBody()->walk([&](mlir::Operation* op){
    if(op->getName().getStringRef()== mlir::affine::AffineForOp::getOperationName()){
      mlir::affine::AffineForOp forop = dyn_cast<AffineForOp>(op);
      assert(forop != NULL);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (WalkResult.wasInterrupted())
    return false;
  else
    return true;
}

/// public function for class ADORA::ForNode
bool ADORA::ForNode::IsThisLevelPerfect(){
  /// Check whether other Op exists in the same nested level of this for opn
  assert(this->HasParentFor() && "Parent loop of this loop has not been set.");
  AffineForOp* ParentFor = &(this->getParent()->getForOp());
  // ParentFor->dump();
  unsigned OpCount = 0;
  auto ib = ParentFor->getBody()->begin();
  auto ie = ParentFor->getBody()->end();
  for(; ib != ie; ib ++ ){
    if(ib->getName().getStringRef() == mlir::affine::AffineForOp::getOperationName() ||
       ib->getName().getStringRef() == ADORA::KernelOp::getOperationName() ||
       ib->getName().getStringRef() == mlir::affine::AffineYieldOp::getOperationName())
    {
      OpCount++;
    } 
    else
      return false; /// Find other Op whose type is not AffineFor/Kernel/Yield
  }
  assert(OpCount >= 2 && "Region of parent for op contain at least 2 op (a for/kernel and a yiled).");
  if(OpCount == 2){
    return true;
  }
  else{
    return false;
  }
}


/// Identifies operations that are beneficial to sink into kernels. These
/// operations may not have side-effects, as otherwise sinking (and hence
/// duplicating them) is not legal.
static bool isSinkingBeneficiary(Operation *op)
{
  return isa<arith::ConstantOp, func::ConstantOp, memref::DimOp,
             arith::SelectOp, arith::CmpIOp>(op);
}

/***
 * The purpose of this function is to determine whether
 * it is beneficial to sink an operation op into a kernel.
 * An operation can be sunk if doing so does not
 * introduce new kernel arguments.
 *
 * The function recursively checks whether each operand
 * of op can be made available via sinking or is already
 * a dependency. If all operands of op can be made available,
 * op is added to beneficiaryOps and its results are
 * marked as now available in availableValues. If an operand
 * cannot be made available via sinking or is not
 * already a dependency, the function returns false.
 *
 * The isSinkingBeneficiary function is used to check
 * whether an operation is a candidate for sinking.
 * This function returns true if the operation is one
 * of arith::ConstantOp, func::ConstantOp, memref::DimOp,
 * arith::SelectOp, or arith::CmpIOp.
 *
 *
 */
static bool extractBeneficiaryOps(Operation *op,
                              llvm::SetVector<mlir::Value> existingDependencies,
                              llvm::SetVector<Operation *> &beneficiaryOps,
                              llvm::SmallPtrSetImpl<mlir::Value> &availableValues)
{
  if (beneficiaryOps.count(op))
    return true;

  if (!isSinkingBeneficiary(op))
    return false;

  for (mlir::Value operand : op->getOperands())
  {
    // It is already visible in the kernel, keep going.
    if (availableValues.count(operand))
      continue;
    // Else check whether it can be made available via sinking or already is a
    // dependency.
    Operation *definingOp = operand.getDefiningOp();
    if ((!definingOp ||
         !extractBeneficiaryOps(definingOp, existingDependencies,
                                beneficiaryOps, availableValues)) &&
        !existingDependencies.count(operand))
      return false;
  }
  // We will sink the operation, mark its results as now available.
  beneficiaryOps.insert(op);
  for (mlir::Value result : op->getResults())
    availableValues.insert(result);
  return true;
}


/// @brief A tool function to sink operations into kernel op
/// @param kernelOp 
/// @return 
LogicalResult mlir::ADORA::sinkOperationsIntoKernelOp(ADORA::KernelOp kernelOp)
{
  Region &KernelOpBody = kernelOp.getBody();

  // Identify uses from values defined outside of the scope of the launch
  // operation.
  llvm::SetVector<mlir::Value> sinkCandidates;
  getUsedValuesDefinedAbove(KernelOpBody, sinkCandidates);

  llvm::SetVector<Operation *> toBeSunk;
  llvm::SmallPtrSet<mlir::Value, 4> availableValues;
  for (mlir::Value operand : sinkCandidates)
  {
    Operation *operandOp = operand.getDefiningOp();
    if (!operandOp)
      continue;
    extractBeneficiaryOps(operandOp, sinkCandidates, toBeSunk, availableValues);
  }

  // Insert operations so that the defs get cloned before uses.
  mlir::IRMapping map;
  OpBuilder builder(KernelOpBody);
  for (Operation *op : toBeSunk)
  {
    Operation *clonedOp = builder.clone(*op, map);
    // Only replace uses within the launch op.
    for (auto pair : llvm::zip(op->getResults(), clonedOp->getResults()))
      replaceAllUsesInRegionWith(std::get<0>(pair), std::get<1>(pair),
                                 kernelOp.getBody());
  }
  return success();
}


//===----------------------------------------------------------------------===//
// Extract a kernel to a outlined func
//===----------------------------------------------------------------------===//

/// @brief KERNELToFunc
/// @param KernelOp
/// @param kernelFnName
/// @param operands
/// @return
func::FuncOp mlir::ADORA::
    GenKernelFunc(ADORA::KernelOp KernelOp, llvm::SetVector<mlir::Value> &operands)
{
  Location loc = KernelOp.getLoc();
  // Create a builder with no insertion point, insertion will happen separately
  // due to symbol table manipulation
  OpBuilder builder(KernelOp.getBody().getContext());
  // Contains the region of code that will be outlined
  Region &KernelOpBody = KernelOp.getBody();
  std::string kernelFnName = KernelOp.getKernelName();

  // errs() << kernelFnName << ":\n";
  // KernelOp.dump();
  // Identify uses from values defined outside of the scope of the launch
  // operation.
  getUsedValuesDefinedAbove(KernelOpBody, operands);

  // Create the func.func operation.
  SmallVector<Type, 4> kernelOperandTypes;
  kernelOperandTypes.reserve(operands.size());
  for (mlir::Value operand : operands)
  {
    // errs()  << "  operands:"; operand.dump();
    kernelOperandTypes.push_back(operand.getType());
  }
  FunctionType type =
      FunctionType::get(KernelOp.getContext(), kernelOperandTypes, {});
  func::FuncOp KernelFunc = builder.create<func::FuncOp>(loc, kernelFnName, type);
  //  std::cout << "[debug] after create:\n"; KernelFunc.dump();
  KernelFunc->setAttr(kernelFnName, builder.getUnitAttr());
  KernelFunc->setAttr("Kernel", builder.getUnitAttr());

  /// Pass func arguements outside of KernelOp
  Block *entryBlock = new Block;
  for (Type argTy : type.getInputs())
  {
    entryBlock->addArgument(argTy, loc);
  }

  KernelFunc.getBody().getBlocks().push_back(entryBlock);

  mlir::IRMapping mapping;
  // Block &entryBlock = KernelFunc.getBody().front();
  for (unsigned index = 0; index < operands.size(); index++)
  {
    // errs()  << "  operands:" << operands[index] <<"\n";
    mapping.map(operands[index], entryBlock->getArgument(index));
  }
  KernelOpBody.cloneInto(&KernelFunc.getBody(), mapping);

  Block &KernelOpEntry = KernelOpBody.front();
  Block *clonedKernelOpEntry = mapping.lookup(&KernelOpEntry);
  builder.setInsertionPointToEnd(entryBlock);
  builder.create<cf::BranchOp>(loc, clonedKernelOpEntry);

  KernelFunc.walk([](ADORA::TerminatorOp op) {
    OpBuilder replacer(op);
    replacer.create<func::ReturnOp>(op.getLoc());
    op.erase(); 
  });

  return KernelFunc;
}


/////////////////////////////////////////////
///// functions to simplify AffineApplyOp 
/////////////////////////////////////////////

/*************************************
 * //=============================================//
 * Following functions are copied from AffineOps.cpp
 * //=============================================//
*/
/// Check if `e` is known to be: 0 <= `e` < `k`. Handles the simple cases of `e`
/// being an affine dim expression or a constant.
static bool isNonNegativeBoundedBy(AffineExpr e, ArrayRef<mlir::Value> operands,
                                   int64_t k) {
  if (auto constExpr = e.dyn_cast<AffineConstantExpr>()) {
    int64_t constVal = constExpr.getValue();
    return constVal >= 0 && constVal < k;
  }
  auto dimExpr = e.dyn_cast<AffineDimExpr>();
  if (!dimExpr)
    return false;
  mlir::Value operand = operands[dimExpr.getPosition()];
  // TODO: With the right accessors, this can be extended to
  // LoopLikeOpInterface.
  if (AffineForOp forOp = getForInductionVarOwner(operand)) {
    if (forOp.hasConstantLowerBound() && forOp.getConstantLowerBound() >= 0 &&
        forOp.hasConstantUpperBound() && forOp.getConstantUpperBound() <= k) {
      return true;
    }
  }

  // We don't consider other cases like `operand` being defined by a constant or
  // an affine.apply op since such cases will already be handled by other
  // patterns and propagation of loop IVs or constant would happen.
  return false;
}

/// Returns the largest known divisor of `e`. Exploits information from the
/// values in `operands`.
static int64_t getLargestKnownDivisor(AffineExpr e, ArrayRef<mlir::Value> operands) {
  // This method isn't aware of `operands`.
  int64_t div = e.getLargestKnownDivisor();

  // We now make use of operands for the case `e` is a dim expression.
  // TODO: More powerful simplification would have to modify
  // getLargestKnownDivisor to take `operands` and exploit that information as
  // well for dim/sym expressions, but in that case, getLargestKnownDivisor
  // can't be part of the IR library but of the `Analysis` library. The IR
  // library can only really depend on simple O(1) checks.
  auto dimExpr = e.dyn_cast<AffineDimExpr>();
  // If it's not a dim expr, `div` is the best we have.
  if (!dimExpr)
    return div;

  // We simply exploit information from loop IVs.
  // We don't need to use mlir::getLargestKnownDivisorOfValue since the other
  // desired simplifications are expected to be part of other
  // canonicalizations. Also, mlir::getLargestKnownDivisorOfValue is part of the
  // LoopAnalysis library.
  mlir::Value operand = operands[dimExpr.getPosition()];
  int64_t operandDivisor = 1;
  // TODO: With the right accessors, this can be extended to
  // LoopLikeOpInterface.
  if (AffineForOp forOp = getForInductionVarOwner(operand)) {
    if (forOp.hasConstantLowerBound() && forOp.getConstantLowerBound() == 0) {
      operandDivisor = forOp.getStep().getSExtValue();
    } else {
      uint64_t lbLargestKnownDivisor =
          forOp.getLowerBoundMap().getLargestKnownDivisorOfMapExprs();
      operandDivisor = std::gcd(lbLargestKnownDivisor, forOp.getStep().getSExtValue());
    }
  }
  return operandDivisor;
}
/// Check if expression `e` is of the form d*e_1 + e_2 where 0 <= e_2 < d.
/// Set `div` to `d`, `quotientTimesDiv` to e_1 and `rem` to e_2 if the
/// expression is in that form.
static bool isQTimesDPlusR(AffineExpr e, ArrayRef<mlir::Value> operands, int64_t &div,
                           AffineExpr &quotientTimesDiv, AffineExpr &rem) {
  auto bin = e.dyn_cast<AffineBinaryOpExpr>();
  if (!bin || bin.getKind() != AffineExprKind::Add)
    return false;

  AffineExpr llhs = bin.getLHS();
  AffineExpr rlhs = bin.getRHS();
  div = getLargestKnownDivisor(llhs, operands);
  if (isNonNegativeBoundedBy(rlhs, operands, div)) {
    quotientTimesDiv = llhs;
    rem = rlhs;
    return true;
  }
  div = getLargestKnownDivisor(rlhs, operands);
  if (isNonNegativeBoundedBy(llhs, operands, div)) {
    quotientTimesDiv = rlhs;
    rem = llhs;
    return true;
  }
  return false;
}

/// Gets the constant lower bound on an `iv`.
static std::optional<int64_t> getLowerBound(mlir::Value iv) {
  AffineForOp forOp = getForInductionVarOwner(iv);
  if (forOp && forOp.hasConstantLowerBound())
    return forOp.getConstantLowerBound();
  return std::nullopt;
}

/// Gets the constant upper bound on an affine.for `iv`.
static std::optional<int64_t> getUpperBound(mlir::Value iv) {
  AffineForOp forOp = getForInductionVarOwner(iv);
  if (!forOp || !forOp.hasConstantUpperBound())
    return std::nullopt;

  // If its lower bound is also known, we can get a more precise bound
  // whenever the step is not one.
  if (forOp.hasConstantLowerBound()) {
    return forOp.getConstantUpperBound() - 1 -
           (forOp.getConstantUpperBound() - forOp.getConstantLowerBound() - 1) %
               forOp.getStep().getSExtValue();
  }
  return forOp.getConstantUpperBound() - 1;
}

/// Get a lower or upper (depending on `isUpper`) bound for `expr` while using
/// the constant lower and upper bounds for its inputs provided in
/// `constLowerBounds` and `constUpperBounds`. Return std::nullopt if such a
/// bound can't be computed. This method only handles simple sum of product
/// expressions (w.r.t constant coefficients) so as to not depend on anything
/// heavyweight in `Analysis`. Expressions of the form: c0*d0 + c1*d1 + c2*s0 +
/// ... + c_n are handled. Expressions involving floordiv, ceildiv, mod or
/// semi-affine ones will lead std::nullopt being returned.
static std::optional<int64_t>
getBoundForExpr(AffineExpr expr, unsigned numDims, unsigned numSymbols,
                ArrayRef<std::optional<int64_t>> constLowerBounds,
                ArrayRef<std::optional<int64_t>> constUpperBounds,
                bool isUpper) {
  // Handle divs and mods.
  if (auto binOpExpr = expr.dyn_cast<AffineBinaryOpExpr>()) {
    // If the LHS of a floor or ceil is bounded and the RHS is a constant, we
    // can compute an upper bound.
    if (binOpExpr.getKind() == AffineExprKind::FloorDiv) {
      auto rhsConst = binOpExpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (!rhsConst || rhsConst.getValue() < 1)
        return std::nullopt;
      auto bound = getBoundForExpr(binOpExpr.getLHS(), numDims, numSymbols,
                                   constLowerBounds, constUpperBounds, isUpper);
      if (!bound)
        return std::nullopt;
      return divideFloorSigned(*bound, rhsConst.getValue());
    }
    if (binOpExpr.getKind() == AffineExprKind::CeilDiv) {
      auto rhsConst = binOpExpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (rhsConst && rhsConst.getValue() >= 1) {
        auto bound =
            getBoundForExpr(binOpExpr.getLHS(), numDims, numSymbols,
                            constLowerBounds, constUpperBounds, isUpper);
        if (!bound)
          return std::nullopt;
        return divideCeilSigned(*bound, rhsConst.getValue());
      }
      return std::nullopt;
    }
    if (binOpExpr.getKind() == AffineExprKind::Mod) {
      // lhs mod c is always <= c - 1 and non-negative. In addition, if `lhs` is
      // bounded such that lb <= lhs <= ub and lb floordiv c == ub floordiv c
      // (same "interval"), then lb mod c <= lhs mod c <= ub mod c.
      auto rhsConst = binOpExpr.getRHS().dyn_cast<AffineConstantExpr>();
      if (rhsConst && rhsConst.getValue() >= 1) {
        int64_t rhsConstVal = rhsConst.getValue();
        auto lb = getBoundForExpr(binOpExpr.getLHS(), numDims, numSymbols,
                                  constLowerBounds, constUpperBounds,
                                  /*isUpper=*/false);
        auto ub = getBoundForExpr(binOpExpr.getLHS(), numDims, numSymbols,
                                  constLowerBounds, constUpperBounds, isUpper);
        if (ub && lb &&
            divideFloorSigned(*lb, rhsConstVal) == divideFloorSigned(*ub, rhsConstVal))
          return isUpper ? mod(*ub, rhsConstVal) : mod(*lb, rhsConstVal);
        return isUpper ? rhsConstVal - 1 : 0;
      }
    }
  }
  // Flatten the expression.
  SimpleAffineExprFlattener flattener(numDims, numSymbols);
  flattener.walkPostOrder(expr);
  ArrayRef<int64_t> flattenedExpr = flattener.operandExprStack.back();
  // TODO: Handle local variables. We can get hold of flattener.localExprs and
  // get bound on the local expr recursively.
  if (flattener.numLocals > 0)
    return std::nullopt;
  int64_t bound = 0;
  // Substitute the constant lower or upper bound for the dimensional or
  // symbolic input depending on `isUpper` to determine the bound.
  for (unsigned i = 0, e = numDims + numSymbols; i < e; ++i) {
    if (flattenedExpr[i] > 0) {
      auto &constBound = isUpper ? constUpperBounds[i] : constLowerBounds[i];
      if (!constBound)
        return std::nullopt;
      bound += *constBound * flattenedExpr[i];
    } else if (flattenedExpr[i] < 0) {
      auto &constBound = isUpper ? constLowerBounds[i] : constUpperBounds[i];
      if (!constBound)
        return std::nullopt;
      bound += *constBound * flattenedExpr[i];
    }
  }
  // Constant term.
  bound += flattenedExpr.back();
  return bound;
}

/// Determine a constant upper bound for `expr` if one exists while exploiting
/// values in `operands`. Note that the upper bound is an inclusive one. `expr`
/// is guaranteed to be less than or equal to it.
static std::optional<int64_t> getUpperBound(AffineExpr expr, unsigned numDims,
                                            unsigned numSymbols,
                                            ArrayRef<mlir::Value> operands) {
  // Get the constant lower or upper bounds on the operands.
  SmallVector<std::optional<int64_t>> constLowerBounds, constUpperBounds;
  constLowerBounds.reserve(operands.size());
  constUpperBounds.reserve(operands.size());
  for (mlir::Value operand : operands) {
    constLowerBounds.push_back(getLowerBound(operand));
    constUpperBounds.push_back(getUpperBound(operand));
  }

  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>())
    return constExpr.getValue();

  return getBoundForExpr(expr, numDims, numSymbols, constLowerBounds,
                         constUpperBounds,
                         /*isUpper=*/true);
}

/// Determine a constant lower bound for `expr` if one exists while exploiting
/// values in `operands`. Note that the upper bound is an inclusive one. `expr`
/// is guaranteed to be less than or equal to it.
static std::optional<int64_t> getLowerBound(AffineExpr expr, unsigned numDims,
                                            unsigned numSymbols,
                                            ArrayRef<mlir::Value> operands) {
  // Get the constant lower or upper bounds on the operands.
  SmallVector<std::optional<int64_t>> constLowerBounds, constUpperBounds;
  constLowerBounds.reserve(operands.size());
  constUpperBounds.reserve(operands.size());
  for (mlir::Value operand : operands) {
    constLowerBounds.push_back(getLowerBound(operand));
    constUpperBounds.push_back(getUpperBound(operand));
  }

  std::optional<int64_t> lowerBound;
  if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
    lowerBound = constExpr.getValue();
  } else {
    lowerBound = getBoundForExpr(expr, numDims, numSymbols, constLowerBounds,
                                 constUpperBounds,
                                 /*isUpper=*/false);
  }
  return lowerBound;
}

/// Copy from AffineOps.cpp
/// Replace all occurrences of AffineExpr at position `pos` in `map` by the
/// defining AffineApplyOp expression and operands.
/// When `dimOrSymbolPosition < dims.size()`, AffineDimExpr@[pos] is replaced.
/// When `dimOrSymbolPosition >= dims.size()`,
/// AffineSymbolExpr@[pos - dims.size()] is replaced.
/// Mutate `map`,`dims` and `syms` in place as follows:
///   1. `dims` and `syms` are only appended to.
///   2. `map` dim and symbols are gradually shifted to higher positions.
///   3. Old `dim` and `sym` entries are replaced by nullptr
/// This avoids the need for any bookkeeping.
static LogicalResult replaceDimOrSym(AffineMap *map,
                                     unsigned dimOrSymbolPosition,
                                     SmallVectorImpl<mlir::Value> &dims,
                                     SmallVectorImpl<mlir::Value> &syms) {
  MLIRContext *ctx = map->getContext();
  bool isDimReplacement = (dimOrSymbolPosition < dims.size());
  unsigned pos = isDimReplacement ? dimOrSymbolPosition
                                  : dimOrSymbolPosition - dims.size();
  mlir::Value &v = isDimReplacement ? dims[pos] : syms[pos];
  if (!v)
    return failure();

  auto affineApply = v.getDefiningOp<AffineApplyOp>();
  if (!affineApply)
    return failure();

  // At this point we will perform a replacement of `v`, set the entry in `dim`
  // or `sym` to nullptr immediately.
  v = nullptr;

  // Compute the map, dims and symbols coming from the AffineApplyOp.
  AffineMap composeMap = affineApply.getAffineMap();
  assert(composeMap.getNumResults() == 1 && "affine.apply with >1 results");
  SmallVector<mlir::Value> composeOperands(affineApply.getMapOperands().begin(),
                                     affineApply.getMapOperands().end());
  // Canonicalize the map to promote dims to symbols when possible. This is to
  // avoid generating invalid maps.
  canonicalizeMapAndOperands(&composeMap, &composeOperands);
  AffineExpr replacementExpr =
      composeMap.shiftDims(dims.size()).shiftSymbols(syms.size()).getResult(0);
  mlir::ValueRange composeDims =
      ArrayRef<mlir::Value>(composeOperands).take_front(composeMap.getNumDims());
  mlir::ValueRange composeSyms =
      ArrayRef<mlir::Value>(composeOperands).take_back(composeMap.getNumSymbols());
  AffineExpr toReplace = isDimReplacement ? getAffineDimExpr(pos, ctx)
                                          : getAffineSymbolExpr(pos, ctx);

  // Append the dims and symbols where relevant and perform the replacement.
  dims.append(composeDims.begin(), composeDims.end());
  syms.append(composeSyms.begin(), composeSyms.end());
  *map = map->replace(toReplace, replacementExpr, dims.size(), syms.size());

  return success();
}


/// Copy from AffineOps.cpp
/// Iterate over `operands` and fold away all those produced by an AffineApplyOp
/// iteratively. Perform canonicalization of map and operands as well as
/// AffineMap simplification. `map` and `operands` are mutated in place.
static void composeAffineMapAndOperands(AffineMap *map,
                                        SmallVectorImpl<mlir::Value> *operands) {
  if (map->getNumResults() == 0) {
    canonicalizeMapAndOperands(map, operands);
    *map = simplifyAffineMap(*map);
    return;
  }

  MLIRContext *ctx = map->getContext();
  SmallVector<mlir::Value, 4> dims(operands->begin(),
                             operands->begin() + map->getNumDims());
  SmallVector<mlir::Value, 4> syms(operands->begin() + map->getNumDims(),
                             operands->end());

  // Iterate over dims and symbols coming from AffineApplyOp and replace until
  // exhaustion. This iteratively mutates `map`, `dims` and `syms`. Both `dims`
  // and `syms` can only increase by construction.
  // The implementation uses a `while` loop to support the case of symbols
  // that may be constructed from dims ;this may be overkill.
  while (true) {
    bool changed = false;
    for (unsigned pos = 0; pos != dims.size() + syms.size(); ++pos)
      if ((changed |= succeeded(replaceDimOrSym(map, pos, dims, syms))))
        break;
    if (!changed)
      break;
  }

  // Clear operands so we can fill them anew.
  operands->clear();

  // At this point we may have introduced null operands, prune them out before
  // canonicalizing map and operands.
  unsigned nDims = 0, nSyms = 0;
  SmallVector<AffineExpr, 4> dimReplacements, symReplacements;
  dimReplacements.reserve(dims.size());
  symReplacements.reserve(syms.size());
  for (auto *container : {&dims, &syms}) {
    bool isDim = (container == &dims);
    auto &repls = isDim ? dimReplacements : symReplacements;
    for (const auto &en : llvm::enumerate(*container)) {
      mlir::Value v = en.value();
      if (!v) {
        assert(isDim ? !map->isFunctionOfDim(en.index())
                     : !map->isFunctionOfSymbol(en.index()) &&
                           "map is function of unexpected expr@pos");
        repls.push_back(getAffineConstantExpr(0, ctx));
        continue;
      }
      repls.push_back(isDim ? getAffineDimExpr(nDims++, ctx)
                            : getAffineSymbolExpr(nSyms++, ctx));
      operands->push_back(v);
    }
  }
  *map = map->replaceDimsAndSymbols(dimReplacements, symReplacements, nDims,
                                    nSyms);

  // Canonicalize and simplify before returning.
  canonicalizeMapAndOperands(map, operands);
  *map = simplifyAffineMap(*map);
}

/// Copy from AffineOps.cpp
/// Simplify `expr` while exploiting information from the values in `operands`.
static void simplifyExprAndOperands(AffineExpr &expr, unsigned numDims,
                                    unsigned numSymbols,
                                    ArrayRef<mlir::Value> operands) {
  // We do this only for certain floordiv/mod expressions.
  auto binExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binExpr)
    return;

  // Simplify the child expressions first.
  AffineExpr lhs = binExpr.getLHS();
  AffineExpr rhs = binExpr.getRHS();
  simplifyExprAndOperands(lhs, numDims, numSymbols, operands);
  simplifyExprAndOperands(rhs, numDims, numSymbols, operands);
  expr = getAffineBinaryOpExpr(binExpr.getKind(), lhs, rhs);

  binExpr = dyn_cast<AffineBinaryOpExpr>(expr);
  if (!binExpr || (expr.getKind() != AffineExprKind::FloorDiv &&
                   expr.getKind() != AffineExprKind::CeilDiv &&
                   expr.getKind() != AffineExprKind::Mod)) {
    return;
  }

  // The `lhs` and `rhs` may be different post construction of simplified expr.
  lhs = binExpr.getLHS();
  rhs = binExpr.getRHS();
  auto rhsConst = dyn_cast<AffineConstantExpr>(rhs);
  if (!rhsConst)
    return;

  int64_t rhsConstVal = rhsConst.getValue();
  // Undefined exprsessions aren't touched; IR can still be valid with them.
  if (rhsConstVal <= 0)
    return;

  // Exploit constant lower/upper bounds to simplify a floordiv or mod.
  MLIRContext *context = expr.getContext();
  std::optional<int64_t> lhsLbConst =
      getLowerBound(lhs, numDims, numSymbols, operands);
  std::optional<int64_t> lhsUbConst =
      getUpperBound(lhs, numDims, numSymbols, operands);
  if (lhsLbConst && lhsUbConst) {
    int64_t lhsLbConstVal = *lhsLbConst;
    int64_t lhsUbConstVal = *lhsUbConst;
    // lhs floordiv c is a single value lhs is bounded in a range `c` that has
    // the same quotient.
    if (binExpr.getKind() == AffineExprKind::FloorDiv &&
        divideFloorSigned(lhsLbConstVal, rhsConstVal) ==
            divideFloorSigned(lhsUbConstVal, rhsConstVal)) {
      expr = getAffineConstantExpr(
          divideFloorSigned(lhsLbConstVal, rhsConstVal), context);
      return;
    }
    // lhs ceildiv c is a single value if the entire range has the same ceil
    // quotient.
    if (binExpr.getKind() == AffineExprKind::CeilDiv &&
        divideCeilSigned(lhsLbConstVal, rhsConstVal) ==
            divideCeilSigned(lhsUbConstVal, rhsConstVal)) {
      expr = getAffineConstantExpr(divideCeilSigned(lhsLbConstVal, rhsConstVal),
                                   context);
      return;
    }
    // lhs mod c is lhs if the entire range has quotient 0 w.r.t the rhs.
    if (binExpr.getKind() == AffineExprKind::Mod && lhsLbConstVal >= 0 &&
        lhsLbConstVal < rhsConstVal && lhsUbConstVal < rhsConstVal) {
      expr = lhs;
      return;
    }
  }

  // Simplify expressions of the form e = (e_1 + e_2) floordiv c or (e_1 + e_2)
  // mod c, where e_1 is a multiple of `k` and 0 <= e_2 < k. In such cases, if
  // `c` % `k` == 0, (e_1 + e_2) floordiv c can be simplified to e_1 floordiv c.
  // And when k % c == 0, (e_1 + e_2) mod c can be simplified to e_2 mod c.
  AffineExpr quotientTimesDiv, rem;
  int64_t divisor;
  if (isQTimesDPlusR(lhs, operands, divisor, quotientTimesDiv, rem)) {
    if (rhsConstVal % divisor == 0 &&
        binExpr.getKind() == AffineExprKind::FloorDiv) {
      expr = quotientTimesDiv.floorDiv(rhsConst);
    } else if (divisor % rhsConstVal == 0 &&
               binExpr.getKind() == AffineExprKind::Mod) {
      expr = rem % rhsConst;
    }
    return;
  }

  // Handle the simple case when the LHS expression can be either upper
  // bounded or is a known multiple of RHS constant.
  // lhs floordiv c -> 0 if 0 <= lhs < c,
  // lhs mod c -> 0 if lhs % c = 0.
  if ((isNonNegativeBoundedBy(lhs, operands, rhsConstVal) &&
       binExpr.getKind() == AffineExprKind::FloorDiv) ||
      (getLargestKnownDivisor(lhs, operands) % rhsConstVal == 0 &&
       binExpr.getKind() == AffineExprKind::Mod)) {
    expr = getAffineConstantExpr(0, expr.getContext());
  }
}

/// Copy from AffineOps.cpp
/// Simplify the map while exploiting information on the values in `operands`.
//  Use "unused attribute" marker to silence warning stemming from the inability
//  to see through the template expansion.
void LLVM_ATTRIBUTE_UNUSED
ADORA::simplifyMapWithOperands(AffineMap &map, ArrayRef<mlir::Value> operands) {
  assert(map.getNumInputs() == operands.size() && "invalid operands for map");
  SmallVector<AffineExpr> newResults;
  newResults.reserve(map.getNumResults());
  for (AffineExpr expr : map.getResults()) {
    simplifyExprAndOperands(expr, map.getNumDims(), map.getNumSymbols(),
                            operands);
    newResults.push_back(expr);
  }
  map = AffineMap::get(map.getNumDims(), map.getNumSymbols(), newResults,
                       map.getContext());
}

/*************************************
 * //=============================================//
 * Above functions are copied from AffineOps.cpp
 * //=============================================//
*/

/************
 * A tool function to check whether an affine map of one affine operation is constant:
 * For example:
 * %c0 = arith.constant 0 : index
 * %8 = affine.apply affine_map<(d0) -> (d0 + 1)>(%c0)
 * or
 * %0 = ADORA.BlockLoad %arg0 [0, 0, 0, %c0]
*/
template <typename AffineT>
bool IsConstantAffineOp(AffineT affineop){
  AffineMap map = affineop.getAffineMap();
  if(map.isConstant())
    return true;

  for(unsigned d = 0; d < affineop.getMapOperands().size(); d++){
    mlir::Value operand = affineop.getMapOperands()[d];
    operand.dump();
    if(!operand.isa<BlockArgument>() && isa<arith::ConstantOp>(operand.getDefiningOp()))
      continue;
    else{
            /// Maybe an iteration variable in for op
      for(AffineExpr expr : map.getResults()){
        if(expr.isFunctionOfDim(d)){
          return false;
        }
      }
    }

    // operand.dump();
  }


  return true;
}

void ADORA::simplifyConstantAffineApplyOpsInRegion(mlir::Region& region){
  /////////
  /// Simplify affine.apply operation generated from unrolling.
  /// Refer to AffineOps.cpp: SimplifyAffineOp
  ////////
  SmallVector<affine::AffineApplyOp> ApplyOpsToDel;
  region.walk([&](affine::AffineApplyOp affineOp)
  {
    OpBuilder b(affineOp.getContext());
    // affineOp.dump();
    AffineMap map = affineOp.getAffineMap();
    if(IsConstantAffineOp(affineOp)){
      // map.dump();
      affineOp.getOperation()->getBlock()->dump();
      auto oldOperands = affineOp.getMapOperands();
      SmallVector<mlir::Value, 8> resultOperands(oldOperands);
      composeAffineMapAndOperands(&map, &resultOperands);
      canonicalizeMapAndOperands(&map, &resultOperands);
      simplifyMapWithOperands(map, resultOperands);
      affineOp.getOperation()->getBlock()->dump();
      // map.dump();
      // region.dump();
      assert(map.getResults().size() == 1);
      AffineConstantExpr ApplyExpr = map.getResult(0).dyn_cast<AffineConstantExpr>();
      // ApplyExpr.dump();
      int64_t bias = ApplyExpr.getValue();
    
      IntegerAttr constAttr = b.getIndexAttr(bias);
      arith::ConstantOp newconst = b.create<arith::ConstantOp>(region.getLoc(), b.getIndexType() , constAttr);
      // affine::AffineApplyOp newapply = b.create<affine::AffineApplyOp>(affineOp.getLoc(), map , resultOperands);
      affineOp.getOperation()->getBlock()->push_back(newconst);
      newconst.getOperation()->moveBefore(affineOp);
      // newconst.dump();
      affineOp.getOperation()->replaceAllUsesWith(newconst);
      ApplyOpsToDel.push_back(affineOp);

      affineOp.getOperation()->getBlock()->dump();
    }


    // if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
    //                             resultOperands.begin()))
    // return failure();
    // replaceAffineOp(rewriter, affineOp, map, resultOperands);
  });
  for(auto applyop : ApplyOpsToDel){
    applyop.getOperation()->erase();
  }

}

inline bool ADORA::opIsContainedByKernel(mlir::Operation* op){
  if(isa<ADORA::KernelOp>(op->getParentOp()))
    return true;
  else if(isa<func::FuncOp>(op->getParentOp()))
    return false;
  else
    return opIsContainedByKernel(op->getParentOp());
}

void ADORA::simplifyAddAffineApplyOpsInRegionButOutOfKernel(mlir::Region& region){
  /////////
  /// Simplify affine.apply operation generated from unrollingAndJam.
  /// Refer to AffineOps.cpp: SimplifyAffineOp
  ////////
  SmallVector<affine::AffineApplyOp> ApplyOpsToDel;
  region.walk([&](affine::AffineApplyOp affineOp)
  { 
    if(!opIsContainedByKernel(affineOp.getOperation())){
      OpBuilder b(affineOp.getContext());
      affineOp.getOperation()->getBlock()->dump();
      // affineOp.dump();
      AffineMap map = affineOp.getAffineMap();
      assert(map.getResults().size() == 1 && map.getResult(0).getKind() == AffineExprKind::Add);
      assert(affineOp.getMapOperands().size() == 1);
      mlir::Value operand = affineOp.getMapOperands()[0];
  
      auto expr = map.getResult(0).dyn_cast<AffineBinaryOpExpr>();
      AffineConstantExpr constantexpr;
      // AffineDimExpr dimexpr;
      if(  expr.getLHS().getKind() == AffineExprKind::DimId 
        && expr.getRHS().getKind() == AffineExprKind::Constant){
        // dimexpr = expr.getLHS().dyn_cast<AffineDimExpr>();
        constantexpr = expr.getRHS().dyn_cast<AffineConstantExpr>();
      }
      else if(expr.getRHS().getKind() == AffineExprKind::DimId 
        && expr.getLHS().getKind() == AffineExprKind::Constant){
        // dimexpr = expr.getRHS().dyn_cast<AffineDimExpr>();
        constantexpr = expr.getLHS().dyn_cast<AffineConstantExpr>();
      }
      else{
        assert(false && "This AffineApplyOp is not a DimId + Constant style.");
      }
  
      int64_t v = constantexpr.getValue();
      arith::ConstantOp newconst = b.create<arith::ConstantOp>(affineOp.getLoc(), b.getIndexAttr(v));
      affineOp.getOperation()->getBlock()->push_back(newconst);
      newconst.getOperation()->moveBefore(affineOp);
      arith::AddIOp newadd = b.create<arith::AddIOp>(affineOp.getLoc(), operand, newconst.getResult());
      affineOp.getOperation()->getBlock()->push_back(newadd);
      newadd.getOperation()->moveAfter(affineOp);
  
      affineOp.getOperation()->replaceAllUsesWith(newadd); 
      ApplyOpsToDel.push_back(affineOp);
      
      // if(/*IsConstantAffineOp(affineOp)*/true){
      //   // map.dump();
      //   auto oldOperands = affineOp.getMapOperands();
      //   SmallVector<mlir::Value, 8> resultOperands(oldOperands);
      //   composeAffineMapAndOperands(&map, &resultOperands);
      //   canonicalizeMapAndOperands(&map, &resultOperands);
      //   simplifyMapWithOperands(map, resultOperands);
      //   affineOp.getOperation()->getBlock()->dump();
      //   // map.dump();
      //   // region.dump();
      //   assert(map.getResults().size() == 1);
      //   AffineConstantExpr ApplyExpr = map.getResult(0).dyn_cast<AffineConstantExpr>();
      //   // ApplyExpr.dump();
      //   int64_t bias = ApplyExpr.getValue();
      
      //   IntegerAttr constAttr = b.getIndexAttr(bias);
      //   arith::ConstantOp newconst = b.create<arith::ConstantOp>(region.getLoc(), b.getIndexType() , constAttr);
      //   // affine::AffineApplyOp newapply = b.create<affine::AffineApplyOp>(affineOp.getLoc(), map , resultOperands);
      //   affineOp.getOperation()->getBlock()->push_back(newconst);
      //   newconst.getOperation()->moveBefore(affineOp);
      //   // newconst.dump();
      //   affineOp.getOperation()->replaceAllUsesWith(newconst);
      //   ApplyOpsToDel.push_back(affineOp);
      // }
  
      // if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
      //                             resultOperands.begin()))
      // return failure();
      // replaceAffineOp(rewriter, affineOp, map, resultOperands);
    }
  });
  for(auto applyop : ApplyOpsToDel){
    applyop.getOperation()->erase();
  }
}

void ADORA::simplifyLoadAndStoreOpsInRegion(mlir::Region& region){
  /////////
  /// Simplify load store operation.
  /// Refer to AffineOps.cpp: SimplifyAffineOp
  ////////
  SmallVector<mlir::Operation*> OpsToDel;
  IRRewriter rewriter(region.getContext());

  auto result = region.walk([&](affine::AffineLoadOp affineOp) -> WalkResult
  {
    IRRewriter rewriter(affineOp.getContext());
    OpBuilder b(affineOp.getContext());
    // affineOp.dump();
    AffineMap map = affineOp.getAffineMap();
    // map.dump();
    AffineMap oldMap = map;
    affineOp.getOperation()->getBlock()->dump();
    auto oldOperands = affineOp.getMapOperands();
    SmallVector<mlir::Value, 8> resultOperands(oldOperands);

    composeAffineMapAndOperands(&map, &resultOperands);

    canonicalizeMapAndOperands(&map, &resultOperands);

    simplifyMapWithOperands(map, resultOperands);
    // map.dump();
    // for(auto v : resultOperands)  v.dump();

    // affineOp.getOperation()->getBlock()->dump();

    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                              resultOperands.begin()))
      return WalkResult::advance();

    AffineLoadOp newaffineop = b.create<AffineLoadOp>(affineOp.getLoc(), affineOp.getMemRef(), map,
                                          resultOperands);
    affineOp.getOperation()->getBlock()->push_back(newaffineop);
    newaffineop.getOperation()->moveAfter(affineOp);

    affineOp.getOperation()->replaceAllUsesWith(newaffineop); 
    // newaffineop.dump();
    OpsToDel.push_back(affineOp.getOperation());
  });
  // region.front().dump();

  result = region.walk([&](affine::AffineStoreOp affineOp) -> WalkResult
  {
    IRRewriter rewriter(affineOp.getContext());
    OpBuilder b(affineOp.getContext());
    affineOp.dump();
    AffineMap map = affineOp.getAffineMap();
    // map.dump();
    AffineMap oldMap = map;
    affineOp.getOperation()->getBlock()->dump();
    auto oldOperands = affineOp.getMapOperands();
    SmallVector<mlir::Value, 8> resultOperands(oldOperands);

    composeAffineMapAndOperands(&map, &resultOperands);

    canonicalizeMapAndOperands(&map, &resultOperands);

    simplifyMapWithOperands(map, resultOperands);
    // map.dump();
    // for(auto v : resultOperands)  v.dump();

    // affineOp.getOperation()->getBlock()->dump();

    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                              resultOperands.begin()))
      return WalkResult::advance();

    AffineStoreOp newaffineop = b.create<AffineStoreOp>(affineOp.getLoc(), affineOp.getValue(), affineOp.getMemRef(), map,
                                          resultOperands);
    affineOp.getOperation()->getBlock()->push_back(newaffineop);
    newaffineop.getOperation()->moveAfter(affineOp);

    affineOp.getOperation()->replaceAllUsesWith(newaffineop); 
    // newaffineop.dump();
    OpsToDel.push_back(affineOp.getOperation());
  });
  // region.front().dump();

  for(auto op : OpsToDel){
    op->erase();
  }
  OpsToDel.clear();
  // region.front().dump();

  region.walk([&](affine::AffineApplyOp affineOp) -> WalkResult
  {
    if(getAllUsesInRegion(affineOp.getResult(), &region).size() == 0){
      OpsToDel.push_back(affineOp.getOperation());
      return WalkResult::advance();
    }

    OpBuilder b(affineOp.getContext());
    // affineOp.dump();
    AffineMap map = affineOp.getAffineMap();
    AffineMap oldMap = map;
    // oldMap.dump();
    affineOp.getOperation()->getBlock()->dump();
    auto oldOperands = affineOp.getMapOperands();
    SmallVector<mlir::Value, 8> resultOperands(oldOperands);
    composeAffineMapAndOperands(&map, &resultOperands);
    // map.dump();
    canonicalizeMapAndOperands(&map, &resultOperands);
    simplifyMapWithOperands(map, resultOperands);
    affineOp.getOperation()->getBlock()->dump();

    if (map == oldMap && std::equal(oldOperands.begin(), oldOperands.end(),
                              resultOperands.begin()))
      return WalkResult::advance();

    // rewriter.replaceOpWithNewOp<AffineLoadOp>(affineOp, affineOp.getMemRef(), map,
    //                                       resultOperands);
    rewriter.replaceOpWithNewOp<AffineApplyOp>(affineOp, map, resultOperands);
    
    // affineOp.getOperation()->getBlock()->dump();

    return WalkResult::advance();
  });
  
  for(auto op : OpsToDel){
    op->erase();
  }
  // OpsToDel.clear();
}

/////////////////////////////////////////////
///// End of Simplify AffineApplyOp functions
/////////////////////////////////////////////

/////////////////////////////////////////////
///// simplifyLoopLevelsInRegion functions
/////////////////////////////////////////////
LogicalResult mlir::ADORA::simplifyLoopLevelsInRegion(mlir::Region& region, bool donttouchkernel){
  mlir::SmallVector<AffineForOp> to_erase;
  region.walk([&](AffineForOp forop) -> WalkResult  {
    if(donttouchkernel && isa<ADORA::KernelOp>(forop.getOperation()->getParentOp())){
      return WalkResult::advance();
    }
    mlir::OpBuilder b(forop.getOperation());
    Location loc = forop.getLoc();
    int tc = getConstantTripCount(forop).value_or(0);
    AffineBound lb = forop.getLowerBound();
    AffineMap lbMap = lb.getMap();
    bool isSingleTrip = (tc == 1);
    if (!isSingleTrip) {
      AffineBound ub = forop.getUpperBound();
      AffineMap ubMap = ub.getMap();
      if (lbMap.getNumResults() == 1 &&
          ubMap.getNumResults() == 1 &&
          lbMap.getNumDims() == ubMap.getNumDims() &&
          lbMap.getNumSymbols() == ubMap.getNumSymbols() &&
          lb.getNumOperands() == ub.getNumOperands()) {
        bool sameOperands = true;
        for (unsigned i = 0; i < lb.getNumOperands(); ++i) {
          if (lb.getOperand(i) != ub.getOperand(i)) {
            sameOperands = false;
            break;
          }
        }
        if (sameOperands) {
          AffineExpr lbExpr = lbMap.getResult(0);
          AffineExpr ubExpr = ubMap.getResult(0);
          isSingleTrip = (ubExpr == lbExpr + 1);
        }
      }
    }

    if (isSingleTrip) {
      if (lbMap.getNumResults() == 1) {
        Value ivValue;
        if (lb.getNumOperands() == 0 &&
            lbMap.getResult(0).getKind() == AffineExprKind::Constant) {
          // Replace the IV with the constant lower bound.
          int lbValue = lbMap.getResult(0).dyn_cast<AffineConstantExpr>().getValue();
          ivValue = b.create<arith::ConstantOp>(loc, b.getIndexAttr(lbValue));
        } else {
          // For non-constant bounds (e.g. %i to %i + 1), the single iteration
          // still executes with IV = lower_bound.
          ivValue = b.create<AffineApplyOp>(loc, lbMap, lb.getOperands());
        }
        forop.getInductionVar().replaceAllUsesWith(ivValue);
        
        Block* loopBlock = forop.getBody();
        auto yieldOp = dyn_cast<AffineYieldOp>(loopBlock->getTerminator());
        if (!yieldOp)
          return WalkResult::advance();

        // For loops with iter_args, a single-trip loop enters the body with
        // region iter-args equal to init values.
        ValueRange initOperands = forop.getInits();
        ValueRange regionIterArgs = forop.getRegionIterArgs();
        for (size_t idx = 0; idx < regionIterArgs.size(); ++idx) {
          regionIterArgs[idx].replaceAllUsesWith(initOperands[idx]);
        }
        // assert(isa<AffineYieldOp>(loopBlock->getTerminator()));
        // loopBlock->getTerminator()->erase();
  
        /// move the body of the loop to the parent block
        // mlir::Region* parentRegion = forop->getBlock()->getParent();
  
        /// Move the loop body to the parent block, placing it before the forOp.
        mlir::SmallVector<mlir::Operation*> ops_tomove;
        for(mlir::Operation& operation : *loopBlock){
          if(&operation == forop.getOperation())
            break;
          if(&operation == loopBlock->getTerminator())
            continue;
          ops_tomove.push_back(&operation);
        }
  
        for(auto operation : ops_tomove){
          operation->moveBefore(forop);
        }

        // Replace for-results with yielded values in the single executed
        // iteration.
        for (unsigned idx = 0; idx < forop.getNumResults(); ++idx) {
          forop.getResult(idx).replaceAllUsesWith(yieldOp.getOperand(idx));
        }
        // forop->getBlock()->dump();
        /// Erase the forOp itself.
        to_erase.push_back(forop);
  
        /// simplify affine maps
        // simplifyLoadAndStoreOpsInRegion(*parentRegion);

        // parentRegion->dump();

      }
    }
    return WalkResult::advance();
  });

  for(AffineForOp& op : to_erase)
    op.erase();

  return success();
}



//===----------------------------------------------------------------------===//
// For loop unroll and dse
//===----------------------------------------------------------------------===//
/// @brief This function is to generate DFG from a kernel in affine dialect
///        with linux command.
///        Users need to contain path of cgra-opt, mlir-translate and 
///        LLVM opt in linux system's search path, for example:
///        modify ~/.bashrc:        
///         $PATH="~/ADORA/app-compiler/cgra-opt/build/bin:$PATH"
///         $PATH="~/llvm16-project/build/bin:$PATH"
/// @param KernelsDir  The directory containing the kernel to generate DFG
/// @param kernelFnName The kernel to generate DFG
/// @return Absolute path of DFG
std::string mlir::ADORA::GenDFGfromAffinewithCMD
    (std::string KernelsDir, std::string kernelFnName, std::string llvmCDFGPass)
{
  /// Set cmd execute paths
  std::filesystem::path oldPath = std::filesystem::current_path();
  std::filesystem::current_path(KernelsDir);

  /// Lower to llvm dialect
  std::string sys_cmd = \
    "cgra-opt \
        --arith-expand --memref-expand\
        --affine-simplify-structures\
        -lower-affine --scf-for-loop-canonicalization  -convert-scf-to-cf\
        --finalize-memref-to-llvm=use-opaque-pointers \
        --convert-math-to-llvm --convert-math-to-libm\
        --convert-arith-to-llvm\
        -convert-func-to-llvm=use-bare-ptr-memref-call-conv\
        -reconcile-unrealized-casts \
        --canonicalize "
      + KernelsDir+"/"+ kernelFnName + ".mlir"
      + " -o " + KernelsDir+"/"+ kernelFnName + "_ll.mlir";

  int result = system(sys_cmd.c_str());
  if(result != 0){
    assert(false && "[Error] Lowering to LLVM dialect with cgra-opt falied! ");
    return "";
  }

  /// mlir-translate *_ll.mlir to llvm IR
  sys_cmd = \
    "mlir-translate --mlir-to-llvmir "
    + KernelsDir+"/"+ kernelFnName + "_ll.mlir" 
    + " -o " 
    + KernelsDir+"/"+ kernelFnName + ".ll";

  result = system(sys_cmd.c_str());
  if(result != 0){
    assert(false && "[Error] Fail to translate to LLVM IR with mlir-translate! ");
    return "";
  }     

  /// Generate optimized LLVM IR
  sys_cmd = \
    "opt \
      -O2 --disable-loop-unrolling \
      -disable-vector-combine -slp-max-vf=1 "
      + KernelsDir + "/" + kernelFnName + ".ll" \
      + " -S -o " 
      + KernelsDir + "/" + kernelFnName + "_opt.ll";

  result = system(sys_cmd.c_str());
  if(result != 0){
    assert(false && "[Error] Optimizing LLVM IR with LLVM opt falied! ");
    return "";
  }       

  sys_cmd = \
    "opt \
      --loop-rotate -gvn -mem2reg -memdep -memcpyopt -lcssa -loop-simplify \
      -licm -loop-deletion -indvars -simplifycfg\
      -mergereturn -indvars -instnamer "
      + KernelsDir + "/" + kernelFnName + "_opt.ll" \
      + " -S -o " 
      + KernelsDir + "/" + kernelFnName + "_gvn.ll";

  result = system(sys_cmd.c_str());
  if(result != 0){
    assert(false && "[Error] Optimizing LLVM IR with LLVM opt falied! ");
    return "";
  }       

  /// Generate DFG
  sys_cmd = \
    "opt -load "
      + llvmCDFGPass + " \"-mapping-all=true\" "
      + " --cdfg "
      + KernelsDir + "/" + kernelFnName + "_gvn.ll"
      + " -S -o " 
      + KernelsDir + "/" + kernelFnName + "_cdfg.ll" 
      + " -enable-new-pm=0";
      
  result = system(sys_cmd.c_str());
  if(result != 0){
    assert(false && "[Error] Generating DFG with llvmCDFGPass.so falied! ");
    return "";
  }       

  /// Get DFG Path
  std::string affinedot = KernelsDir + "/affine.dot";
  if(!std::filesystem::exists(affinedot)){
    assert(false && "[Error] Generating affine.dot failed! ");
    return "";
  }
  std::filesystem::rename(affinedot, KernelsDir + "/" + kernelFnName +".dot");
  std::filesystem::current_path(oldPath);  // Come back to old path
  return KernelsDir + "/" + kernelFnName +".dot";
}



/// @brief 
/// @param DFGPath Absolute path of dot file
/// @return ALU and LSU number of DFG
ADORA::DFGInfo mlir::ADORA::GetDFGinfo(std::string DFGPath)
{
  DFGInfo dfginfo;
  std::ifstream  DFGdotStream;  

  DFGdotStream.open(DFGPath);
  if (!DFGdotStream.is_open()) {
    assert(0 && "DFG .dot can not be open.");
  } 

 	std::stringstream  DFGStrstream;
  DFGStrstream << DFGdotStream.rdbuf();
  DFGdotStream.close();
  std::string strline;
  std::smatch match;

  std::regex node_pattern("([a-zA-Z0-9]+)\\[opcode=([a-zA-Z0-9]+)"); //eg. FACC3283[opcode=FACC32, acc_params="0, 4, 1, 12", acc_first=1];
  std::regex edge_pattern("([a-zA-Z0-9]+) -> ([a-zA-Z0-9]+)\\[operand = ([0-9]+)"); //eg. const1->sub3[operand=0];
  /// read in every line
  while (getline(DFGStrstream, strline)){
    // std::cout << "strline:" << strline << std::endl;
    bool found = regex_search(strline, match, node_pattern);
    /**** Found a node ****/
    if(found){ 
      std::string opcode;
      opcode = match.str(2);
      if(opcode == "Input" || opcode == "Output"){
        dfginfo.Num_LSU ++;
      }
      else{
        dfginfo.Num_ALU ++;
      }
    }
    // found = regex_search(strline, match, edge_pattern);
    // /**** Found an edge ****/
    // if(found){
    // }
  }
      	// DFGdotstr = DFGStrstream.str();
      	// std::cout<<DFGdotstr<<std::endl
  return dfginfo;
}       

/// @brief 
/// @param CDFG cdfg generated from mlir or llvm
/// @return ALU and LSU number of DFG
ADORA::DFGInfo mlir::ADORA::GetDFGinfo(LLVMCDFG* CDFG)
{
  DFGInfo dfginfo;
  auto _nodes = CDFG->nodes();
  for(auto &elem : _nodes){
    int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    std::string opcode = node->getTypeName();
    if(opcode == "Input" || opcode == "Output"){
      dfginfo.Num_LSU ++;
    }
    else{
      dfginfo.Num_ALU ++;
    }
  }

  return dfginfo;
}       

/// @brief 
/// @param Node 
/// @param CurrentDesignSpace 
/// @return An updated DesignSpace depending on this Node
SmallVector<DesignPoint> mlir::ADORA::
        ExpandTilingAndUnrollingFactors(ADORA::ForNode Node, SmallVector<DesignPoint> CurrentDesignSpace){
  SmallVector<DesignPoint> NewAllDesignSpace;
  
  // Design Space is empty
  if(CurrentDesignSpace.size()==0){
    DesignPoint emptypoint;
    CurrentDesignSpace.push_back(emptypoint);
  }

  for(DesignPoint point : CurrentDesignSpace){
    // for( unsigned tilingFactor : Node.TilingFactors )
    /// TODO: Tiling is not considered right now. 
    point.push_back(1); // No tiling right now
    if(Node.UnrollFactors.size()==0){
      point.push_back(1);
      NewAllDesignSpace.push_back(point);
    }
    for(unsigned UnrollFactor : Node.UnrollFactors ){
      DesignPoint newpoint = point;
      newpoint.push_back(UnrollFactor);
      NewAllDesignSpace.push_back(newpoint);
    }
  }
  return NewAllDesignSpace;
}

/// @brief find all unrolling factors (which divides trip count)
/// @param Node a ForNode in loop-nest tree
/// @return vector which contains all unrolling factors
SmallVector<unsigned> ADORA::FindUnrollingFactors(ADORA::ForNode& Node){
  // assert(Node.IsInnermost()&&"Only innermost loop-nest can be unrolled.");
  auto optionalTripCount = getConstantTripCount(Node.getForOp());
  assert(optionalTripCount&&"Variable loop bound!");
  // SmallVector<unsigned> validFactors;
  unsigned factor = 1;
  unsigned tripCount = optionalTripCount.value();

  Node.UnrollFactors.clear();
  while (factor <= tripCount) {
    /// Push back the current factor.
    /// unrolling factor = 1 means no unrolling applied
    Node.UnrollFactors.push_back(factor);

    // Find the next possible size.
    ++factor;
    while (factor <= tripCount && tripCount % factor != 0)
      ++factor;
  }
  // Node.UnrollFactors = std::move(validFactors);
  return Node.UnrollFactors;
}


/// @brief construct a AffineForTree with for-nodes and set the parent-child relationship
/// @param topfunc 
/// @return all for-nodes
SmallVector<ADORA::ForNode> mlir::ADORA::createAffineForTree(func::FuncOp topfunc){
  SmallVector<ADORA::ForNode> ForNodeVec;
  topfunc.walk([&](mlir::Operation* op){
    if(op->getName().getStringRef()== mlir::affine::AffineForOp::getOperationName()){
      mlir::affine::AffineForOp forop = dyn_cast<AffineForOp>(op);
      assert(forop != NULL);
      ADORA::ForNode newForNode(forop);
      ForNodeVec.push_back(newForNode);
    }
  });

  auto TopForOps = topfunc.getOps<AffineForOp>();
  auto targetLoops =
      SmallVector<AffineForOp, 4>(TopForOps.begin(), TopForOps.end());
  for (AffineForOp loop : targetLoops) {
      ADORA::ForNode* rootForNode = findTargetLoopNode(ForNodeVec, loop);
      rootForNode->setLevel(0);
      NestedGenTree(rootForNode, ForNodeVec);
  }

  return ForNodeVec;
}

/// @brief construct a AffineForTree with for-nodes and set the parent-child relationship inside a kernel
/// @param kernel 
/// @return all for-nodes from outermost to innermost
SmallVector<ADORA::ForNode> mlir::ADORA::createAffineForTreeInsideKernel(ADORA::KernelOp kernel){
  SmallVector<ADORA::ForNode> ForNodeVec;
  kernel.walk([&](mlir::Operation* op){
    if(op->getName().getStringRef()== mlir::affine::AffineForOp::getOperationName()){
      mlir::affine::AffineForOp forop = dyn_cast<AffineForOp>(op);
      forop.dump();
      assert(forop != NULL);
      ADORA::ForNode newForNode(forop);
      ForNodeVec.push_back(newForNode);
    }
  });

  auto TopForOps = kernel.getOps<AffineForOp>();
  auto targetLoops =
      SmallVector<AffineForOp, 4>(TopForOps.begin(), TopForOps.end());
  for (AffineForOp loop : targetLoops) {
      ADORA::ForNode* rootForNode = findTargetLoopNode(ForNodeVec, loop);
      rootForNode->dumpForOp();
      rootForNode->setLevel(0);
      NestedGenTree(rootForNode, ForNodeVec);
      rootForNode->dumpTree();
  }

  return ForNodeVec;
}

/// @brief construct a AffineForTree with for-nodes and set the parent-child relationship 
///       outside and inside a kernel. 
/// @param kernel 
/// @return all for-nodes from outermost to innermost. Contains only one level if a parent for op
///       exists out of kernel.
SmallVector<ADORA::ForNode> mlir::ADORA::createAffineForTreeAroundKernel(ADORA::KernelOp kernel){
  SmallVector<ADORA::ForNode> forNodeVec;
  forNodeVec = createAffineForTreeInsideKernel(kernel);
  
  ///// push parent for op 
  affine::AffineForOp Parentfor = dyn_cast_or_null<affine::AffineForOp> 
                                    (kernel.getOperation()->getParentOp()); 

  if(Parentfor != nullptr){
    ADORA::ForNode parentForNode(Parentfor, -1/*Level: out of kernel*/);
    parentForNode.setOutOfKernel(true);
    forNodeVec.push_back(parentForNode);
    // NestedGenTree(parentForNode, forNodeVec);
  }
  
  return forNodeVec;
}


/**
 * return the single kernel in one func:FuncOp
*/
ADORA::KernelOp ADORA::getSingleKernelFromFunc(func::FuncOp func){
  SmallVector<ADORA::KernelOp> kernels;
  func.walk([&](ADORA::KernelOp kernel){
    kernels.push_back(kernel);
  });
  assert(kernels.size() == 1);
  return kernels[0];
}


/**
 * return the corresponding kernel in the copied module op
*/
ADORA::KernelOp ADORA::getKernelFromCopiedModule(ModuleOp ModuleOp, ADORA::KernelOp kernel){
  if(kernel.hasKernelName()){
    SmallVector<ADORA::KernelOp> copied_kernels;
    ModuleOp.walk([&](ADORA::KernelOp copied_kernel){
      if(copied_kernel.hasKernelName() && copied_kernel.getKernelName() == kernel.getKernelName()){
        copied_kernels.push_back(copied_kernel);
      }
    });
    assert(copied_kernels.size() == 1);
    return copied_kernels[0];
  }

  return getSingleKernelFromFunc(*(ModuleOp.getOps<func::FuncOp>().begin()));;
}

/// @brief 
/// @param NodeVec a small vector which contains all For Node
/// @param forop a target loop we want to find 
/// @return the pointer to the target Loop Node which is the for op we want to find
ADORA::ForNode* mlir::ADORA::findTargetLoopNode(SmallVector<ADORA::ForNode>& NodeVec, mlir::affine::AffineForOp forop)
{
  ForNode* ib = NodeVec.begin();
  ForNode* ie = NodeVec.end();
  for(; ib != ie; ib++){
    if(ib->getForOp() == forop)
      return ib;
  }
  return nullptr;
}

/// @brief Add relationship between parent and child nodes
/// @param rootNode A parent node whose children have not been set
/// @param NodeVec A small vector which contains all For Node
void mlir::ADORA::NestedGenTree(ADORA::ForNode* rootNode, SmallVector<ADORA::ForNode>& NodeVec){
  unsigned Level = rootNode->getLevel() + 1;

  AffineForOp For = rootNode->getForOp();
  llvm::SmallVector<ForNode*> ChildrenVec;
  auto ib = For.getBody()->begin();
  auto ie = For.getBody()->end();
  for(; ib != ie; ib ++ ){
    if(ib->getName().getStringRef() == mlir::affine::AffineForOp::getOperationName())
    {
      mlir::affine::AffineForOp NestFor = dyn_cast<AffineForOp>(ib);
      // ADORA::ForNode ChildForNode(NestFor, /*Level=*/Level);
      ADORA::ForNode* ChildForNode = findTargetLoopNode(NodeVec, NestFor);
      ChildForNode->setParent(rootNode);
      ChildForNode->setLevel(Level);
      NestedGenTree(ChildForNode, NodeVec);
      ChildrenVec.push_back(ChildForNode);
    } 
    if(ib->getName().getStringRef() == ADORA::KernelOp::getOperationName())
    {
      ADORA::KernelOp NestKernel = dyn_cast<ADORA::KernelOp>(ib);
      auto kn_ib = NestKernel.getBody().front().begin();
      auto kn_ie = NestKernel.getBody().front().end();
      for(; kn_ib != kn_ie; kn_ib ++ ){
        /// search nested loop in KernelOp
        if(ib->getName().getStringRef() == mlir::affine::AffineForOp::getOperationName())
        {
          mlir::affine::AffineForOp NestFor = dyn_cast<AffineForOp>(ib);
          ADORA::ForNode* ChildForNode = findTargetLoopNode(NodeVec, NestFor);
          ChildForNode->setParent(rootNode);
          ChildForNode->setLevel(Level);
          NestedGenTree(ChildForNode, NodeVec);
          ChildrenVec.push_back(ChildForNode);
        }
      }
    } 
  }

  rootNode->setChildren(ChildrenVec);
}


/// @brief A design point is a number sequence:
///        tilefactor(loop0),unrollfactor(loop0),tilefactor(loop1),unrollfactor(loop1).......
/// for example:
///       (1,1),(1,1),(1,3) represents a 3-level nested loop and the innermost loop is unrolled by 3
/// @param ForNodes All for loops
/// @return A SmallVector containing all design point
SmallVector<SmallVector<unsigned>> mlir::ADORA::ConstructUnrollSpace(SmallVector<ADORA::ForNode> ForNodes){
  using namespace llvm; // for llvm.errs()
  SmallVector<SmallVector<unsigned>> AllDesignSpace;

  // unsigned point_num = 1;
  for (int i = 0; i < ForNodes.size(); i++) {
    /// Find all unroll factor
    FindUnrollingFactors(ForNodes[i]);

    /// From innermost to outermost
    ADORA::ForNode currentlevel = ForNodes[i];
    UnrollDesignPoint point;
    // assert(currentlevel.IsThisLevelPerfect());

    /// For loops outer than this level, all factor will be 1.
    for(int j = 0; j < i; j++){
      point.push_back(ForNodes[j].getMaxUnrollFactor());
    }
    point.push_back(0); // point[i]
    /// For loops inner than this level, all factor will be tripcounts(fully unroll).
    for(int j = i + 1; j < ForNodes.size(); j++){
      point.push_back(1);
    }

    /// For this level loop,store all feasible factoe.
    for(auto UF: ForNodes[i].UnrollFactors){
      point[i] = UF;
      if(findElement(AllDesignSpace, point) == -1){
        AllDesignSpace.push_back(point);
      }
    }
  }

  // /*** Print Design Space ***/
  llvm::errs() <<"//-----------  Design Space  ----------//\n";
  llvm::errs() <<"Loop-nested structure:\n";
  for (ADORA::ForNode Node : ForNodes) {
    if (!Node.HasParentFor())/// outermost loop
      Node.dumpTree();
  }

  llvm::errs() <<"Design Space:\n";
  for(UnrollDesignPoint point : AllDesignSpace){
    for (int i = 0; i < ForNodes.size(); i++) {
      llvm::errs() << point[i] << " ";
    }
    llvm::errs() << "\n";
  }

  llvm::errs() <<"//-------------------------------------//\n";
  return AllDesignSpace;
}

/// @brief Check whether the irregular Iteration Space of for op is supported   
///   Lower and upper bound and step will be checked.
///   Only following types of irregular Iteration Space are supported.
///         affine.for %arg2 = 0 to affine_map<(d0) -> (d0)>(%arg1)
///     or  affine.for %arg3 = 0 to affine_map<(d0) -> (1800 - d0)>(%arg2) 
/// @param forOp
bool mlir::ADORA::IsIterationSpaceSupported(mlir::affine::AffineForOp &forOp){
  OpBuilder b(forOp);
  /// Check step
  int64_t step = forOp.getStep().getSExtValue();
  if( (!forOp.hasConstantLowerBound() 
    || !forOp.hasConstantUpperBound())
    && step != 1) 
    return false;

  /// Check lower bound
  if (forOp.hasConstantLowerBound()) ///lower bound
  {
    AffineExpr Expr = b.getAffineConstantExpr(forOp.getConstantLowerBound());
    if(Expr.dyn_cast<AffineConstantExpr>().getValue() != 0)
      return false;
  }
  else
  {
    return false; /// Really?
  }

  /// Check upper bound
  if (!forOp.hasConstantUpperBound()) ///upper bound 
  { 
    AffineMap map = forOp.getUpperBoundMap();
    if(forOp.getUpperBoundOperands().size() != 1 
      || map.getResults().size() != 1) 
      return false;
  
    AffineExpr Expr = map.getResult(0);
    if(Expr.getKind() != AffineExprKind::DimId){
      
      if(Expr.getKind() == AffineExprKind::Add ||
         Expr.getKind() == AffineExprKind::Mul){
          AffineExpr LHS = Expr.dyn_cast<AffineBinaryOpExpr>().getLHS();
          AffineExpr RHS = Expr.dyn_cast<AffineBinaryOpExpr>().getRHS();
          
          if(LHS.getKind() != AffineExprKind::DimId 
            || LHS.getKind() != AffineExprKind::Constant){
              return false;
          }
      }
      
      else{
        return false;
      }
    }
  }

  return true;
}


/// @brief check whether the loadop and storeop are accessing the same memref
///        and the same address
/// @param loadop 
/// @param storeop 
/// @return 
bool mlir::ADORA::LoadStoreSameMemAddr(AffineLoadOp loadop, AffineStoreOp storeop)
{
  return TwoAccessSameMemAddr(loadop, storeop);

  //// Following part has been coded as a template function in Passes.h
    // mlir::Value loadMemref = loadop.getMemref();
    // AffineMapAttr loadMapAttr = loadop.getAffineMapAttr();
    // Operation::operand_range loadIndices = loadop.getIndices();
    // Operation* loadMemrefOp = loadMemref.getDefiningOp();
    
    // mlir::Value storeMemref = storeop.getMemref();
    // AffineMapAttr storeMapAttr = storeop.getAffineMapAttr(); 
    // Operation::operand_range storeIndices = storeop.getIndices();
    // Operation* storeMemrefOp = storeMemref.getDefiningOp();  

    // mlir::Value storeValue = storeop.getValue();
    // if(!isa<BlockArgument>(storeValue) && isa<arith::ConstantOp>(storeValue.getDefiningOp()))
    //   return false;

    // // llvm::errs() << "[info] loadop: " << loadop << "\n";   
    // // llvm::errs() << "[info] loadMemref: " << loadMemref << "\n";   
    // // llvm::errs() << "[info] loadMapAttr: " << loadMapAttr << "\n";   

    // // llvm::errs() << "[info] storeop: " << storeop << "\n"; 
    // // llvm::errs() << "[info] storeMemref: " << storeMemref << "\n"; 
    // // llvm::errs() << "[info] storeMapAttr: " << storeMapAttr << "\n"; 

    
    // if(loadMemref == storeMemref && loadMapAttr == storeMapAttr
    //           && loadIndices.size() == storeIndices.size())
    // {
    //   for(unsigned i = 0; i < loadIndices.size(); i++){
    //     if(loadIndices[i] != storeIndices[i])
    //       return false;
    //   }
    //   return true;
    // }
    // else if(
    //   !isa<BlockArgument>(loadMemref) && !isa<BlockArgument>(storeMemref)
    //   && isa<ADORA::DataBlockLoadOp>(loadMemrefOp) && isa<ADORA::DataBlockLoadOp>(storeMemrefOp)
    //   && dyn_cast<ADORA::DataBlockLoadOp>(loadMemrefOp).getOriginalMemref() 
    //   == dyn_cast<ADORA::DataBlockLoadOp>(storeMemrefOp).getOriginalMemref())
    // {
    //   for(unsigned i = 0; i < loadIndices.size(); i++){
    //     if(loadIndices[i] != storeIndices[i])
    //       return false;
    //   }
    //   return true;      
    // }
    
    // else if(!isa<BlockArgument>(loadMemref) && !isa<BlockArgument>(storeMemref)
    //     &&isa<ADORA::DataBlockLoadOp>(loadMemrefOp) 
    //     &&isa<ADORA::LocalMemAllocOp>(storeMemrefOp)){
    //   ADORA::LocalMemAllocOp AllocOp = dyn_cast<ADORA::LocalMemAllocOp>(storeMemrefOp);
    //   SmallVector<mlir::Operation*> Consumers = 
    //     getAllUsesInBlock(AllocOp, AllocOp.getOperation()->getBlock());
    //   AllocOp.getOperation()->getBlock()->dump();
    //   // AllocOp.getOperation()->getBlock()->getRegion()->dump();
    //   SmallVector<ADORA::DataBlockStoreOp> BLKStoreConsumers;
    //   for(auto comsumer: Consumers) {
    //     if(isa<ADORA::DataBlockStoreOp>(comsumer))
    //       BLKStoreConsumers.push_back(dyn_cast<ADORA::DataBlockStoreOp>(comsumer));
    //   }
    //     // comsumer->dump();
    //   assert(BLKStoreConsumers.size() == 1);
    //   storeMemref = BLKStoreConsumers[0].getTargetMemref();

    //   if(dyn_cast<ADORA::DataBlockLoadOp>(loadMemrefOp).getOriginalMemref() == storeMemref)
    //   {
    //     for(unsigned i = 0; i < loadIndices.size(); i++){
    //       if(loadIndices[i] != storeIndices[i])
    //         return false;
    //     }
    //     return true;      
    //   }
    //   else 
    //     return false;
    // }
     
    // else{
    //   return false;
    // }
}

/// @param ld a block load or localmemalloc operation
/// @return The block store operation that accesses the same tile as a block load or local allocation.
///         If no such store operation exists, return nullopt.
template<typename ldormalloc>
static std::optional<ADORA::DataBlockStoreOp> GetBlockStoreOfSameTile(ldormalloc ld, SmallVector<ADORA::DataBlockStoreOp> stores){
  for(auto& blkst : stores){
    if(isa<ldormalloc>(blkst.getSourceMemref().getDefiningOp())
      && dyn_cast<ldormalloc>(blkst.getSourceMemref().getDefiningOp()) == ld)
      return blkst;
  }

  //// TODO: judge index and memory shape
  //// TODO: judge index and memory shape
  //// TODO: judge index and memory shape
  
  return std::nullopt;
}

// template<typename ldormalloc>
// static SmallVector<ADORA::DataBlockStoreOp> GetBlockStoreOfSameTile(ldormalloc ld, SmallVector<ADORA::DataBlockStoreOp> stores){
//   SmallVector<ADORA::DataBlockStoreOp> results;
//   for(auto& blkst : stores){
//     if(isa<ldormalloc>(blkst.getSourceMemref().getDefiningOp())
//         && dyn_cast<ldormalloc>(blkst.getSourceMemref().getDefiningOp()) == ld
//         && )
//     {      
//       results.push_back(blkst);
//     }
//   }

//   //// TODO: judge index and memory shape
//   //// TODO: judge index and memory shape
//   //// TODO: judge index and memory shape
  
//   return std::nullopt;
// }

/// @brief Retrieves the block store operation that accesses the same tile as the provided block load or local memory allocation operation.
/// 
/// This function searches for all block load and local memory allocation operations within the same parent function
/// as the given block store operation. It checks if any of these operations share the same kernel name and ID
/// as the provided store operation. If exactly one matching operation is found, it is returned; otherwise, an assertion will trigger.
///
/// @param store A block store operation whose corresponding block load or local memory allocation operation is to be found.
/// 
/// @return The block load operation or local memory allocation operation that accesses the same tile as the input store operation.
///         If no such operation exists, the behavior is undefined due to the assertion.
mlir::Operation* ::mlir::ADORA::GetTheSourceOperationOfBlockStore(ADORA::DataBlockStoreOp store){
  func::FuncOp parentfunc = store.getOperation()->getParentOfType<func::FuncOp>();
  SmallVector<mlir::Operation*> allLoadsMalloc;
  SmallVector<mlir::Operation*> result;

  parentfunc.walk([&](mlir::Operation* op){
    if(isa<ADORA::DataBlockLoadOp>(op) || isa<ADORA::LocalMemAllocOp>(op))
      allLoadsMalloc.push_back(op);
  });  

  for(auto& op : allLoadsMalloc){
    if(isa<ADORA::DataBlockLoadOp>(op)){
      ADORA::DataBlockLoadOp load = dyn_cast<ADORA::DataBlockLoadOp>(op);
      if(findElement(load.getKernelNameAsStrVector(), store.getKernelName().str()) != -1
        && load.getId() == store.getId()){
        result.push_back(op);
      }
    }
    else if(isa<ADORA::LocalMemAllocOp>(op)){
      ADORA::LocalMemAllocOp alloc = dyn_cast<ADORA::LocalMemAllocOp>(op);
      if(findElement(alloc.getKernelNameAsStrVector(), store.getKernelName().str()) != -1
        && alloc.getId() == store.getId()){
        result.push_back(op);
      }
    }
  }

  assert(result.size() == 1);
  return result[0];
}

/// @brief reset all Ids in blockload/blockstore/localalloc op
void ADORA::ResetIndexOfBlockAccessOpInFunc(func::FuncOp& func){
  /// For every block store
  SmallVector<ADORA::DataBlockStoreOp> blkstores;
  SmallDenseMap<ADORA::DataBlockStoreOp, int> BlkstoreToId;
  func.walk([&](ADORA::DataBlockStoreOp op){
    blkstores.push_back(op);
  });  

  /// get all block load and block allocation operation
  // SmallVector<mlir::Operation*> LoadsAndLocalAllocs;
  int current_id = 0;
  func.walk([&](mlir::Operation* op){
    if(isa<ADORA::DataBlockLoadOp>(op)){
      ADORA::DataBlockLoadOp blkload = dyn_cast<ADORA::DataBlockLoadOp>(op);
      std::optional<ADORA::DataBlockStoreOp> blkstore = GetBlockStoreOfSameTile(blkload, blkstores);
      if(blkstore.has_value()){
        if(BlkstoreToId.find(*blkstore) != BlkstoreToId.end()){
          blkload.setId(BlkstoreToId[*blkstore]);
        }
        else{
          blkload.setId(current_id);
          (*blkstore).setId(current_id);
          BlkstoreToId[(*blkstore)] = current_id++;
        }
      }
      else{
        blkload.setId(current_id++);
      }
    }
    else if(isa<ADORA::LocalMemAllocOp>(op)){
      ADORA::LocalMemAllocOp blkalloc = dyn_cast<ADORA::LocalMemAllocOp>(op);
      std::optional<ADORA::DataBlockStoreOp> blkstore = GetBlockStoreOfSameTile(blkalloc, blkstores);
      assert(blkstore.has_value());
      blkalloc.setId(current_id);
      (*blkstore).setId(current_id);
      BlkstoreToId[(*blkstore)] = current_id++;
    }

  });
}



// /// Interleaver 2 SmallVectors and only reserve 1 element for same elements, just like set merge 
// template <typename T>
// SmallVector<T> mlir::ADORA::SetInterleaverForVector(const llvm::SmallVector<T>& v1, const llvm::SmallVector<T>& v2){
//   SmallVector<T> v;
//   for(auto e : v1){
//     if(findElement(v, e) == -1){
//       v.push_back(e);
//     }
//   }
//   for(auto e : v2){
//     if(findElement(v, e) == -1){
//       v.push_back(e);
//     }
//   }
//   return v;
// }

/// @brief Get the corresponding dimension of one dim var.
///   For example: d0:arg7 map: [0, %arg7, %arg8, 0], then return {1}.
///   Note: this map must be simplified. 
///    [0, %arg7 + %arg6, %arg8, 0] This kind of map may not be analyzed.
/// @param dim 
/// @param map 
/// @return 
SmallVector<int> mlir::ADORA::getOperandDimensionsInMap(const int dim, const AffineMap map){
  SmallVector<int> r;
  for(int i = 0; i < map.getResults().size(); i++){
    AffineExpr expr = map.getResults()[i];
    assert(expr.getKind() == AffineExprKind::DimId 
        || expr.getKind() == AffineExprKind::Constant);
    if(expr.isFunctionOfDim(dim)){
      r.push_back(i);
    }
  }
  return r;
}

/// @brief get the instance number from CGRA adg
/// @param CGRAadg file path of CGRA adg
/// @param instype the kind of instance to be counted 
/// @return 
unsigned mlir::ADORA::getInstanceNumFromADG  (const std::string& CGRAadg,const std::string& instype_to_count){
  unsigned count = 0;
  // Read target adg JSON file.
  std::string errorMessage;
  auto ADGFile = mlir::openInputFile(CGRAadg, &errorMessage);
  if (!ADGFile) {
    llvm::errs() << errorMessage << "\n";
    // signalPassFailure();
    assert(false && "[Error] ADGFile open failed! ");
    return -1;
  }

  // Parse JSON file into memory.
  auto ADGjson = llvm::json::parse(ADGFile->getBuffer());
  if (!ADGjson) {
    llvm::errs() << "failed to parse the target cgra adg json file\n";
    // signalPassFailure();
    assert(false && "[Error] ADGFile.json parsing failed! ");
    return -1;
  }
  auto ADGObj = ADGjson.get().getAsObject();
  if (!ADGObj) {
    llvm::errs() << "support an object in the target spec json file, found "
                    <<    "something else\n";
    // signalPassFailure();
    assert(false && "[Error] ADGFile.json do not contain an ADGObj! ");
    return -1;
  }

  // unsigned SizeSpadBank = ADGObj->getInteger("iob_spad_bank_size").value_or(8192);
  ///TODO: Size Scratchpad 
 
  auto instances = ADGObj->getArray("instances");   
  for(auto insJson : *instances){
    auto insJsonObj = insJson.getAsObject();
    auto insType = insJsonObj->getString("type").value();
    // llvm::errs() << "[info] type:" << insType << "\n";      
    if(insType == instype_to_count)
      count++;
  }

  return count;
}

/// @brief get a string attribute from CGRA adg
/// @param CGRAadg file path of CGRA adg
/// @param key the JSON field name to query
/// @return value of the key if exists, otherwise empty string
int64_t mlir::ADORA::getIntegerAttrFromADG(const std::string &CGRAadg,
                                              const std::string &key) {
  // Read target adg JSON file.
  std::string errorMessage;
  auto ADGFile = mlir::openInputFile(CGRAadg, &errorMessage);
  if (!ADGFile) {
    llvm::errs() << errorMessage << "\n";
    assert(false && "[Error] ADGFile open failed! ");
    abort();
  }

  // Parse JSON file into memory.
  auto ADGjson = llvm::json::parse(ADGFile->getBuffer());
  if (!ADGjson) {
    llvm::errs() << "failed to parse the target cgra adg json file\n";
    assert(false && "[Error] ADGFile.json parsing failed! ");
    abort();
  }

  auto ADGObj = ADGjson.get().getAsObject();
  if (!ADGObj) {
    llvm::errs() << "support an object in the target spec json file, found "
                    << "something else\n";
    assert(false && "[Error] ADGFile.json does not contain an ADGObj! ");
    abort();
  }

  // Get value
  auto val = ADGObj->getInteger(key);
  if (!val) {
    llvm::errs() << "[Warning] key \"" << key << "\" not found in ADG file\n";
    abort();
  }

  return val.value();
}


ADORA::IselOp mlir::ADORA::ReplaceValueWithNewIselOp(OpBuilder b, Location loc, mlir::Value value){
  ADORA::IselOp isel = b.create<ADORA::IselOp>(loc, value);
  value.replaceAllUsesExcept(isel, isel.getOperation());
  return isel;
}

ADORA::IselOp mlir::ADORA::ReplaceLoopCarryValueWithNewIselOp(affine::AffineForOp& forop, int IterRegionOperandIdx){
  OpBuilder b(forop);
  forop.getBody()->front().dump();
  Location loc = forop.getBody()->front().getLoc();
  ADORA::IselOp isel = ReplaceValueWithNewIselOp(b, loc, forop.getRegionIterArgs()[IterRegionOperandIdx]);
  isel.getOperation()->moveBefore(&(forop.getBody()->front()));
  forop.dump();
  forop.getRegionIterArgs()[IterRegionOperandIdx].dump();
  return isel;
}



PositionRelationInLoop mlir::ADORA::getPositionRelationship(Operation* lhs, Operation* rhs)
{
  Operation* lhs_parent = lhs->getParentOp();
  Operation* rhs_parent = rhs->getParentOp(); 
  
  if(lhs_parent == rhs_parent && lhs_parent->getName().getStringRef() == ADORA::KernelOp::getOperationName() )
    return PositionRelationInLoop::SameLevel;

  else if( lhs_parent->getName().getStringRef() != AffineForOp::getOperationName() ||
      rhs_parent->getName().getStringRef() != AffineForOp::getOperationName() )
    return PositionRelationInLoop::NotInSameLoopNest;

  AffineForOp lhsParentForOp = dyn_cast<AffineForOp>(*lhs_parent);
  AffineForOp rhsParentForOp = dyn_cast<AffineForOp>(*rhs_parent);

  /// lhs and rhs are in AffineForOp region
  if(lhsParentForOp == rhsParentForOp){
    return PositionRelationInLoop::SameLevel;
  }
  else {
    WalkResult result = lhsParentForOp.walk([&](Operation* op) -> WalkResult {
      // llvm::errs() << "[info] op name: "  << op->getName() <<"\n";
      if(op == rhs)
        return WalkResult::interrupt();
      return WalkResult::advance(); });

    if(result == WalkResult::interrupt()){
      /// Found rhs in region of lhs's parent forop,
      /// which means lhs is in outer level than rhs is.
      return PositionRelationInLoop::LhsOuter;
    }
    
    result = rhsParentForOp.walk([&](Operation* op) -> WalkResult {
      if(op == lhs)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if(result == WalkResult::interrupt()){
      /// Found lhs in region of rhs's parent forop,
      
      /// which means r     s is in outer level than lhs is.
      return PositionRelationInLoop::RhsOuter;
    }
  }
  
  return PositionRelationInLoop::NotInSameLoopNest;
}

/// @brief Move a pair of load store op outer, generate loop-carried iteration variables
/// @param loadop 
/// @param storeop 
/// @return success or not
std::optional<AffineForOp> mlir::ADORA::MoveLoadStorePairOut(AffineLoadOp loadop, AffineStoreOp storeop){ 
  PositionRelationInLoop PosRelation =
    getPositionRelationship(loadop.getOperation(), storeop.getOperation());
  bool Success = false;
  switch (PosRelation)
  {
    case PositionRelationInLoop::SameLevel :
    {
      // Loadop and store op are in same level so 
      // both should be hoisted.
      Success = true;

      /// Get value to be yielded
      mlir::Value toYield = storeop.getValue();

      /// Get ValueRanges of old for op
      SmallVector<mlir::Value, 4> dupInitOperands, dupIterArgs, dupYieldOperands;
      Operation* ParentOp = loadop.getOperation()->getParentOp();
      if(isa<ADORA::KernelOp>(ParentOp)){
        Success = false;
        break;
      }

      AffineForOp oldForOp = dyn_cast<AffineForOp>(*ParentOp);
      OpBuilder builder(oldForOp.getContext());

      // Move load-store pair
      loadop.getOperation()->moveBefore(oldForOp);           
      storeop.getOperation()->moveAfter(oldForOp);

      ValueRange oldInitOperands = oldForOp.getInits();
      // ValueRange oldIterArgs = oldForOp.getRegionIterArgs();
      ValueRange oldYieldOperands =
          cast<AffineYieldOp>(oldForOp.getBody()->getTerminator()).getOperands();
      // dupInitOperands.append(oldInitOperands.begin(), oldInitOperands.end());
      // dupIterArgs.append(oldIterArgs.begin(), oldIterArgs.end());
      // dupYieldOperands.append(oldYieldOperands.begin(), oldYieldOperands.end());

      // /// Add new mlir::Value to be yielded to dupInitOperands and dupYieldOperands
      dupInitOperands.push_back(loadop);
      // dupIterArgs.push_back(toYield);
      dupYieldOperands.push_back(toYield);

      // // Create a new loop with additional iterOperands, iter_args and yield
      // // operands. This new loop will take the loop body of the original loop.
      // AffineForOp newForOp = replaceForOpWithNewYields(
      //     builder, oldForOp, dupIterOperands, dupYieldOperands, dupIterArgs); 
      // oldForOp.getOperation()->erase();
      IRRewriter rewriter(oldForOp->getContext());
      

      AffineForOp newForOp =
        cast<AffineForOp>(*oldForOp.replaceWithAdditionalYields(
          rewriter, dupInitOperands, /*replaceInitOperandUsesInLoop=*/true,
          [&](OpBuilder &b, Location loc, ArrayRef<BlockArgument> newBbArgs) {
            return dupYieldOperands;
          }));

      // Change the input of the new forop
      LLVM_DEBUG(newForOp.dump());
      // unsigned newOperandIndex = newForOp.getNumIterOperands();
      // newForOp.getOperation()->setOperand(newOperandIndex, loadop);
      // LLVM_DEBUG(newForOp.dump());
      // Replace all uses of loadop with new iter_arg of forop
      // replaceAllUsesInRegionWith( loadop.getResult(), 
      //                             newForOp.getRegionIterArgs()[newOperandIndex],
      //                             newForOp.getRegion());
      // Change the input of the store op
      unsigned newResultIndex = newForOp.getNumIterOperands() - 1;
      Value newForResult = newForOp.getOperation()->getResult(newResultIndex);
      storeop.getOperation()
                ->setOperand(storeop.getStoredValOperandIndex(),newForResult);
      LLVM_DEBUG(newForOp.dump());
      return newForOp;
      // break;
    }
    case PositionRelationInLoop::LhsOuter :
    {
      /// load op is in outer level
      /// hoist store op only
      Success = true;
      Operation* storeParentOp = storeop.getOperation()->getParentOp();
      if(isa<ADORA::KernelOp>(storeParentOp)){
        Success = false;
        break;
      }
      storeop.getOperation()->moveAfter(storeParentOp);
      
      return dyn_cast<AffineForOp>(storeParentOp);
      // break;
    }
    case PositionRelationInLoop::RhsOuter : 
    {
      /// store op is in outer level
      /// hoist load op only
      Success = true;
      Operation* loadopParentOp = loadop.getOperation()->getParentOp();
      loadop.getOperation()->moveBefore(loadopParentOp);
      if(isa<ADORA::KernelOp>(loadopParentOp)){
        Success = false;
        break;
      }
      
      return dyn_cast<AffineForOp>(loadopParentOp);
      // break;
    }
    
    default:
      return std::nullopt;
      // break;
  }
   return std::nullopt;
}



ADORA::KernelOp mlir::ADORA::findTheOnlyKernelInNestedLoop(affine::AffineForOp forOp) {
  ADORA::KernelOp result = nullptr;

  forOp.getBody()->walk([&](ADORA::KernelOp kernelOp) {
    if (result) {
      llvm::errs() << "Error: Multiple KernelOps found inside nested loop!\n";
      forOp.dump();
      assert(false && "Expected only one KernelOp in nested loop");
    }
    result = kernelOp;
  });

  return result;
}
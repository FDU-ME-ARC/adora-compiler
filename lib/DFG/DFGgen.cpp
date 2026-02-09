#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/OperationSupport.h"
// #include "mlir/IR/OpDefinition.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/APFloat.h"            // llvm::APFloat
#include "llvm/ADT/APInt.h"              // llvm::APInt

#include <iostream>
#include <string>
#include <bit>
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
// #include "PassDetail.h"
// #include "ADORA/Dialect/ADORA/Transforms/Passes.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "inc/mlir_cdfg.h"
#include "ADORA/Misc/DFG.h"

// For Block handle 
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// For op transformation
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

// DFG
ADORA::KernelOp* _kernel_toDFG;
int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::vector;
using namespace mlir::ADORA;

#define DEBUG_TYPE "adora-dfg-gen"

/// @brief 
/// @param yieldop 
/// @param value 
/// @return the value must be in yieldop's operands
static int GetYieldIndexFromValue(affine::AffineYieldOp yield, mlir::Value v){
  int outidx;
  for(outidx = 0; outidx < yield.getOperands().size(); outidx++){
    if(yield.getOperand(outidx) == v){
      break;
    }
  }
  // assert(outidx < yield.getOperands().size() && "Outer affinefor do not yield this op.");
  if(outidx == yield.getOperands().size())
    return -1;
  else
    return outidx;
}

/// @brief 
/// @param yieldop 
/// @param value 
/// @return the index of the value in yieldop's operands
static int getYieldIndexOfValue(affine::AffineYieldOp yieldop, mlir::Value value){
  int index;
  for(index = 0; index < yieldop.getOperands().size(); index++ ){
    if(value == yieldop.getOperand(index))
      break;
  }
  if(index == yieldop.getOperands().size()){
    LLVM_DEBUG(llvm::errs() << value << "is not an operand of yieldop: " << yieldop);

    assert(isa<affine::AffineForOp>(value.getDefiningOp()->getParentOp()));
    affine::AffineForOp parentforop = dyn_cast<affine::AffineForOp>(value.getDefiningOp()->getParentOp());
    assert(parentforop != dyn_cast<affine::AffineForOp>(yieldop.getOperation()->getParentOp()));

    AffineYieldOp yieldInThisLevel = dyn_cast<AffineYieldOp>(parentforop.getBody()->getTerminator());
    int yieldindex = getYieldIndexOfValue(yieldInThisLevel, value);
    mlir::Value OuterValue = parentforop.getResult(yieldindex);
    return getYieldIndexOfValue(yieldop, OuterValue);
  }
  else{
    return index;
  }
}

/// @brief 
/// @param forop 
/// @param value 
/// @return the index of the value in forop's operands
static int getInitIndexOfValue(affine::AffineForOp forop, mlir::Value value){
  int index;
  for(index = 0; index < forop.getInits().size(); index++ ){
    if(value == forop.getInits()[index])
      break;
  }
  if(index == forop.getInits().size()){
    assert(0);
    // LLVM_DEBUG(llvm::errs() << value << " is not an operand of yieldop: " << forop);

    // assert(isa<affine::AffineForOp>(value.getDefiningOp()->getParentOp()));
    // affine::AffineForOp parentforop = dyn_cast<affine::AffineForOp>(value.getDefiningOp()->getParentOp());
    // assert(parentforop != dyn_cast<affine::AffineForOp>(yieldop.getOperation()->getParentOp()));

    // AffineYieldOp yieldInThisLevel = dyn_cast<AffineYieldOp>(parentforop.getBody()->getTerminator());
    // int yieldindex = getYieldIndexOfValue(yieldInThisLevel, value);
    // mlir::Value OuterValue = parentforop.getResult(yieldindex);
    // return getYieldIndexOfValue(yieldop, OuterValue);
    return -1;
  }
  else{
    return index;
  }
}

/// @brief 
/// @param value 
/// @param op 
/// @return check whether the value is in op's operands
static bool ValueIsInOperands(mlir::Value value, mlir::Operation* op){
  int index;
  for(index = 0; index < op->getOperands().size(); index++ ){
    if(value == op->getOperand(index))
      return true;
  }
  return false;
}

static void SetACCOperandIdx(LLVMCDFGNode* node){
  assert(node->isAcc());
  assert(node->inputNodes().size() <= 2);
  for(auto innode : node->inputNodes()){
    if(isa<AffineForOp>(innode->operation())){
      continue;
    }
    else{
      node->setInputIdx(innode, 0);
      node->setInputPort(innode, 0);
    }
  }
}


template <typename DataT>
DataT DataAttrValue2NewType(mlir::Attribute constattr){
  DataT result;
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    if(floatattr.getType().isF64()){
      result = (DataT)value;
    } 
    else if(floatattr.getType().isF32()){
      result = (DataT)(float) value;
    }
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    if(intattr.getType().isInteger(16)){    
      int value = intattr.getInt();   
      int16_t value_16 = (int16_t) value;       
      result = (DataT)value_16;
    }
    else if(intattr.getType().isInteger(32)){
      int value = intattr.getInt();   
      int32_t value_32 = (int32_t) value;       
      result = (DataT)value_32;
    }
    else if(intattr.getType().isInteger(64)){ 
      int value = intattr.getInt();   
      int64_t value_64 = (int64_t) value;       
      result = (DataT)value_64;
    }
    else if(intattr.getType().isUnsignedInteger(16)){    
      unsigned int value = intattr.getUInt();   
      int16_t value_16 = (int16_t) value;       
      result = (DataT)value_16;
    }
    else if(intattr.getType().isUnsignedInteger(32)){
      unsigned int value = intattr.getUInt();   
      int32_t value_32 = (int32_t) value;       
      result = (DataT)value_32;
    }
    else if(intattr.getType().isUnsignedInteger(64)){ 
      unsigned int value = intattr.getUInt();   
      int64_t value_64 = (int64_t) value;       
      result = (DataT)value_64;
    }
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    result = (DataT)value;
  }
  return result;
}

/////
// A tool function to remove constant truncf
// void RemoveConstantTruncF(func::FuncOp Func){
void RemoveConstantTruncF(ADORA::KernelOp kernel){
  OpBuilder b(kernel);
  kernel.walk([&](arith::TruncFOp truncf){
    Operation* in = truncf.getIn().getDefiningOp();
    mlir::Type outtype = truncf.getOut().getType();
    // in->dump();
    // llvm::errs() << "value: "<< truncf.getOut() << " getType: " << truncf.getOut().getType();

    if(isa<arith::ConstantOp>(in)){
      arith::ConstantOp constin = dyn_cast<arith::ConstantOp>(in);
      mlir::Attribute constattr = in->getAttr(constin.getValueAttrName());
      // llvm::errs() << "constattr: "<< constattr << "\n";
      arith::ConstantOp newconst;
      if(outtype.isF32() || outtype.isF16() || outtype.isBF16()){
        float r = DataAttrValue2NewType<float>(constattr);
        FloatAttr outAttr = FloatAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isF64()){
        double r = DataAttrValue2NewType<double>(constattr);
        FloatAttr outAttr = FloatAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(16)){
        int16_t r = DataAttrValue2NewType<int16_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(32)){
        int32_t r = DataAttrValue2NewType<int32_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      else if(outtype.isInteger(64)){
        int64_t r = DataAttrValue2NewType<int64_t>(constattr);
        IntegerAttr outAttr = IntegerAttr::get(outtype, r);
        newconst = b.create<arith::ConstantOp>(kernel.getLoc(), outtype , outAttr);
      }
      // else if(isa<IntegerAttr>(constattr)){

      // }
      // kernel.getOperation()->getBlock()->push_back(newconst);
      newconst.getOperation()->moveAfter(truncf);
      truncf.getOperation()->replaceAllUsesWith(newconst);
    }
  //    
  //     mlir::Attribute constattr = op->getAttr(constop.getValueAttrName());
  //
  //   }
  });
  // kernel.dump();
}

/// @brief A tool function to get another operand's index for a binary operation
/// @param Op 
/// @param operandA 
/// @return index for another operand
static unsigned getAnotherOperandIdx(mlir::Operation* op, mlir::Value operandA){
  assert(op->getOperands().size() == 2);
  int idxA;
  for(idxA = 0; idxA < op->getOperands().size(); idxA++){
    if(op->getOperands()[idxA] == operandA){
      break;
    }
  }
  assert(idxA != op->getOperands().size() && "This operandA value doesn't belong to this op.");
  return op->getOperands().size() - 1 - idxA;
}

// static bool checkAccumulationChain(affine::AffineForOp forop, mlir::Value tocheck, int yieldidx = -1);

/// @brief A tool function to check whether this op is in an accumulation chain.
///     A acc chain is like: blockarg-addf-addf-addf-yield
/// @param forop 
/// @param tocheck 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(affine::AffineForOp forop, mlir::Value tocheck, int yieldidx = -1){
  forop.dump();
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(tocheck, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    use->dump();
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(forop, dyn_cast<AccT>(use).getResult(), yieldidx);

    else if(isa<affine::AffineYieldOp>(use)){
      assert(yieldidx!=-1 || dyn_cast<affine::AffineYieldOp>(use).getOperands().size() == 1);
      if(dyn_cast<affine::AffineYieldOp>(use).getOperand(yieldidx) == tocheck)
        return true;
      else
        return false;
    }

    else if(isa<affine::AffineForOp>(use)){
      affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
      int index = getInitIndexOfValue(innerforop, tocheck);
      assert(index != -1);

      mlir::Value carry_v = innerforop.getRegionIterArgs()[index];
      carry_v.dump();
      return checkAccumulationChain<AccT>(innerforop, carry_v, index);
    }

    else 
      return false;
  }
}

/// @brief A tool function to check whether this op is in an accumulation chain.
///     A acc chain is like: blockarg-addf-addf-addf-yield
/// @param forop 
/// @param tocheck 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(affine::AffineForOp forop, int operandidx){
  mlir::Value blockarg_v = forop.getRegionIterArgs()[operandidx];
  blockarg_v.dump();

  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(blockarg_v, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(forop, dyn_cast<AccT>(use).getResult(), operandidx);
    else if(isa<affine::AffineYieldOp>(use)){
      /// blockarg_v has been directly yield.
      return false;
    }
    // else if(isa<affine::AffineForOp>(use)){
    //   affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
    //   int index = getInitIndexOfValue(innerforop, blockarg_v);
    //   assert(index != -1);
    //   return checkAccumulationChain<AccT>(innerforop, index);
    // }
    else 
      return false;
  }
}

/// @brief A tool function to check whether the operation chain between load and store op is in an accumulation chain.
///     A acc chain is like: loadop-addf-addf-addf-storeop
/// @param beused 
/// @param region 
/// @return true or false
template <typename AccT>
static bool checkAccumulationChain(mlir::Operation* op, affine::AffineStoreOp storeop){
  if(op->getNumResults() != 1)
    return false;
  
  AffineForOp forop = dyn_cast<AffineForOp>(op->getParentOp());
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(op->getResult(0), forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    return false;
  }
  else {
    mlir::Operation* use = uses[0];
    if(isa<AccT>(use))
      return checkAccumulationChain<AccT>(use, storeop);
    else if(isa<affine::AffineYieldOp>(use)){
      //// check whether this yield is used by storeop
      int yieldidx = GetYieldIndexFromValue(dyn_cast<affine::AffineYieldOp>(use), op->getResult(0));
      assert(yieldidx != -1);
      if(storeop.getValue() == forop.getResult(yieldidx))
        return true;
      else
        return false;
    }
    else 
      return false;
  }

  return false;
}

// template <typename AccT>
// static bool checkAccumulationChain(affine::AffineLoadOp loadop, affine::AffineStoreOp storeop){
//   return checkAccumulationChain<AccT>(loadop.getOperation(), storeop);
// }

/// @brief hoist load store op to get loop-carried value, which helps acc extractions 
///   ACC operations: ADD MUL ADDF MULF SEL
/// @param kernel 
void HoistLoadStoreInKernelOp(ADORA::KernelOp kernel){
  HoistLoadStoreOpsInOp(kernel);
}

/**
 * A tool function to check whether a mlir value is the initial value of a accumulation chain.
 * */
template <typename AccT>
bool IsAccumulationInitialValue(affine::AffineForOp forop, mlir::Value InitValue){
  int idx;
  for(idx = 0; idx < forop.getInits().size(); idx++){
    if(forop.getInits()[idx] == InitValue){
      break;
    }
  }
  assert(idx != forop.getInits().size() && "This init value doesn't belong to this for op.");
  mlir::Value RegionIterArg = forop.getRegionIterArgs()[idx];
  SmallVector<mlir::Operation*> uses = getAllUsesInBlock(RegionIterArg, forop.getBody());
  // RegionIterArg.dump();
  if(uses.size() != 1){
    /// not an accumulation chain.
    /// TODO: really?
    return false;
  }
  else{
    mlir::Operation* use = uses[0];
    if(isa<affine::AffineForOp>(use)){
      /// an initial value of loop carried value of inner affine for.
      affine::AffineForOp innerforop = dyn_cast<affine::AffineForOp>(use);
      return IsAccumulationInitialValue<AccT>(innerforop, RegionIterArg);
    }
    else if(isa<AccT>(use)){
      return checkAccumulationChain<AccT>(forop, RegionIterArg, idx);
    }
    else {
      return false;
    }
  }
}

/**
 * A tool function to move the initial value computing of accumulation to the outer most level.
 * */
template <typename AccT>
bool MoveAccumulationInitialValue(affine::AffineForOp forop, mlir::Value InitValue){
  int idx;
  for(idx = 0; idx < forop.getInits().size(); idx++){
    if(forop.getInits()[idx] == InitValue){
      break;
    }
  }
  assert(idx != forop.getInits().size() && "This init value doesn't belong to this for op.");
  OpBuilder b(forop);
  mlir::Type datatype = InitValue.getType();
  arith::ConstantOp newconst;
  if(datatype.isF32() || datatype.isF16() || datatype.isBF16()){
    float constvalue = 0;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);
  }
  else if(datatype.isF64()){
    double constvalue = 0;
    FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);
  }
  else {
    int64_t constvalue = 0;
    IntegerAttr constAttr = IntegerAttr::get(datatype, constvalue);
    newconst = b.create<arith::ConstantOp>(forop.getLoc(), datatype , constAttr);    
  }
  // datatype.dump();
  // forop.dump();
  // forop.getOperation()->getBlock()->dump();
  InitValue.getDefiningOp()->replaceAllUsesWith(newconst);
  AccT newcompute = b.create<AccT>(forop.getLoc(), forop.getResults()[idx] , InitValue); 
  newcompute.getOperation()->moveAfter(forop);
  // forop.dump();
  // forop.getOperation()->getBlock()->dump();

  forop.getResults()[idx].replaceAllUsesWith(newcompute);
  newcompute.setOperand(0, forop.getResults()[idx]);

  // forop.getOperation()->getBlock()->dump();
  // forop.dump();
}

/**
 * 
 * A tool function to move the initial value computing of accumulation to the outer most level.
 * For example:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, 0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 *  Can change to:
 *            affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %cst = arith.const 0 : f32
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %cst) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, 0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                %6 = %10 = arith.addf %cst, %5 : f32
                affine.store %6, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 * 
*/
void MoveLoopCarriedInitailValue(ADORA::KernelOp kernel){
  // OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    for(mlir::Value IterOperand : forop.getInits()){
      // IterOperand.dump();
      AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      if(!IterOperand.isa<BlockArgument>() 
        && yieldop.getOperands().size() != 0){
        if(IsAccumulationInitialValue<arith::AddIOp>(forop, IterOperand) 
          && !IterOperand.getDefiningOp<arith::ConstantOp>()){
          MoveAccumulationInitialValue<arith::AddIOp>(forop, IterOperand);
        }
        else if(IsAccumulationInitialValue<arith::AddFOp>(forop, IterOperand)
          && !IterOperand.getDefiningOp<arith::ConstantOp>()){
          MoveAccumulationInitialValue<arith::AddFOp>(forop, IterOperand);
        }
        // else if(IsAccumulationInitialValue<arith::MulIOp>(forop, IterOperand)){
        //   MoveAccumulationInitialValue<arith::MulIOp>(forop, IterOperand);;
        // }
        // else if(IsAccumulationInitialValue<arith::MulFOp>(forop, IterOperand)){
        //   MoveAccumulationInitialValue<arith::MulFOp>(forop, IterOperand);;
        // }
      }
    }
  });
  // kernel.dump();
}



/**
 * 
 * A tool function to For accumulation chain, move accumulation operation to the last using commutative law of addition/multiplication
 * For example:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c0] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, %c0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32
                    %10 = arith.addf %arg11, %9 : f32
                    %c1 = arith.constant 1 : index
                    %11 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c1] : memref<1x3x7x62xf32>
                    %12 = affine.load %1[0, %arg8, %arg10, %c1] : memref<2x3x7x7xf32>
                    %13 = arith.mulf %11, %12 : f32
                    %14 = arith.addf %10, %13 : f32
                    %c2 = arith.constant 2 : index
                    %15 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c2] : memref<1x3x7x62xf32>
                    %16 = affine.load %1[0, %arg8, %arg10, %c2] : memref<2x3x7x7xf32>
                    %17 = arith.mulf %15, %16 : f32
                    %18 = arith.addf %14, %17 : f32
                    %c3 = arith.constant 3 : index
                    %19 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c3] : memref<1x3x7x62xf32>
                    %20 = affine.load %1[0, %arg8, %arg10, %c3] : memref<2x3x7x7xf32>
                    %21 = arith.mulf %19, %20 : f32
                    %22 = arith.addf %18, %21 : f32
                    %c4 = arith.constant 4 : index
                    %23 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c4] : memref<1x3x7x62xf32>
                    %24 = affine.load %1[0, %arg8, %arg10, %c4] : memref<2x3x7x7xf32>
                    %25 = arith.mulf %23, %24 : f32
                    %26 = arith.addf %22, %25 : f32
                    %c5 = arith.constant 5 : index
                    %27 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c5] : memref<1x3x7x62xf32>
                    %28 = affine.load %1[0, %arg8, %arg10, %c5] : memref<2x3x7x7xf32>
                    %29 = arith.mulf %27, %28 : f32
                    %30 = arith.addf %26, %29 : f32
                    %c6 = arith.constant 6 : index
                    %31 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c6] : memref<1x3x7x62xf32>
                    %32 = affine.load %1[0, %arg8, %arg10, %c6] : memref<2x3x7x7xf32>
                    %33 = arith.mulf %31, %32 : f32
                    %34 = arith.addf %30, %33 : f32
                    affine.yield %34 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 *  Can change to:
              affine.for %arg7 = 0 to 28 {
                %4 = affine.load %2[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
                %5 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %4) -> (f32) {
                  %6 = affine.for %arg10 = 0 to 7 iter_args(%arg11 = %arg9) -> (f32) {
                    %7 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c0] : memref<1x3x7x62xf32>
                    %8 = affine.load %1[0, %arg8, %arg10, %c0] : memref<2x3x7x7xf32>
                    %9 = arith.mulf %7, %8 : f32

                    %c1 = arith.constant 1 : index
                    %11 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c1] : memref<1x3x7x62xf32>
                    %12 = affine.load %1[0, %arg8, %arg10, %c1] : memref<2x3x7x7xf32>
                    %13 = arith.mulf %11, %12 : f32
                    %14 = arith.addf %9, %13 : f32
                    %c2 = arith.constant 2 : index
                    %15 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c2] : memref<1x3x7x62xf32>
                    %16 = affine.load %1[0, %arg8, %arg10, %c2] : memref<2x3x7x7xf32>
                    %17 = arith.mulf %15, %16 : f32
                    %18 = arith.addf %14, %17 : f32
                    %c3 = arith.constant 3 : index
                    %19 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c3] : memref<1x3x7x62xf32>
                    %20 = affine.load %1[0, %arg8, %arg10, %c3] : memref<2x3x7x7xf32>
                    %21 = arith.mulf %19, %20 : f32
                    %22 = arith.addf %18, %21 : f32
                    %c4 = arith.constant 4 : index
                    %23 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c4] : memref<1x3x7x62xf32>
                    %24 = affine.load %1[0, %arg8, %arg10, %c4] : memref<2x3x7x7xf32>
                    %25 = arith.mulf %23, %24 : f32
                    %26 = arith.addf %22, %25 : f32
                    %c5 = arith.constant 5 : index
                    %27 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c5] : memref<1x3x7x62xf32>
                    %28 = affine.load %1[0, %arg8, %arg10, %c5] : memref<2x3x7x7xf32>
                    %29 = arith.mulf %27, %28 : f32
                    %30 = arith.addf %26, %29 : f32
                    %c6 = arith.constant 6 : index
                    %31 = affine.load %0[0, %arg8, %arg10, %arg7 * 2 + %c6] : memref<1x3x7x62xf32>
                    %32 = affine.load %1[0, %arg8, %arg10, %c6] : memref<2x3x7x7xf32>
                    %33 = arith.mulf %31, %32 : f32
                    %34 = arith.addf %30, %33 : f32

                    %10 = arith.addf %arg11, %34 : f32
                    affine.yield %10 : f32
                  }
                  affine.yield %6 : f32
                }
                affine.store %5, %3[0, 0, 0, %arg7] : memref<1x1x1x28xf32>
              }
 * 
*/
void MoveAccumulationToLast(ADORA::KernelOp kernel){
  OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    int IterRegionOperandIdx;
    for(IterRegionOperandIdx = 0; IterRegionOperandIdx < forop.getNumRegionIterArgs(); IterRegionOperandIdx++){
      mlir::Value IterRegionOperand = forop.getRegionIterArgs()[IterRegionOperandIdx];
      if(checkAccumulationChain<arith::AddFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        // IterArgConsumer->dump();
        if(isa<affine::AffineYieldOp>(IterArgConsumer))
          continue;
        assert(isa<arith::AddFOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::AddFOp NewYieldProducer = b.create<arith::AddFOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::AddIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];

        // what is this for?
        // if(isa<affine::AffineForOp>(IterArgConsumer->getParentOp()))
        //   continue;   
        if(isa<affine::AffineYieldOp>(IterArgConsumer))
          continue;
        
        assert(isa<arith::AddIOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::AddIOp NewYieldProducer = b.create<arith::AddIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::MulFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        
        if(isa<affine::AffineYieldOp>(IterArgConsumer))
          continue;

        assert(isa<arith::MulFOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::MulFOp NewYieldProducer = b.create<arith::MulFOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      else if(checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        /// Adjust consumer of iter operand
        assert(getAllUsesInBlock(IterRegionOperand, forop.getBody()).size() == 1);
        mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
        
        if(isa<affine::AffineYieldOp>(IterArgConsumer))
          continue;
                  
        assert(isa<arith::MulIOp>(IterArgConsumer));
        mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
        IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
        IterArgConsumer->erase();

        /// Adjust producer of yield
        AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
        mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
        mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
        arith::MulIOp NewYieldProducer = b.create<arith::MulIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
        NewYieldProducer.getOperation()->moveBefore(yieldop);
        OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
        NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      }
      // else if(checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
      //   /// Adjust consumer of iter operand
      //   mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
      //   assert(isa<arith::MulIOp>(IterArgConsumer));
      //   mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
      //   IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
      //   IterArgConsumer->erase();

      //   /// Adjust producer of yield
      //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      //   mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
      //   mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
      //   arith::MulIOp NewYieldProducer = b.create<arith::MulIOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
      //   NewYieldProducer.getOperation()->moveBefore(yieldop);
      //   OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
      //   NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      // }
      else{
        //// Insert ISEL operation
        // (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
        continue;
      }
      // else if(checkAccumulationChain<arith::SelectOp>(forop, IterRegionOperand)){
      //   /// Adjust consumer of iter operand
      //   mlir::Operation* IterArgConsumer = getAllUsesInBlock(IterRegionOperand, forop.getBody())[0];
      //   assert(isa<arith::SelectOp>(IterArgConsumer));
      //   mlir::Value AnotherOperand = IterArgConsumer->getOperand(getAnotherOperandIdx(IterArgConsumer, IterRegionOperand));
      //   IterArgConsumer->replaceAllUsesWith(AnotherOperand.getDefiningOp());
      //   IterArgConsumer->erase();

      //   /// Adjust producer of yield
      //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
      //   mlir::Value OldYieldProducer_Value = yieldop.getOperand(IterRegionOperandIdx);
      //   mlir::Operation* OldYieldProducer = OldYieldProducer_Value.getDefiningOp();
      //   arith::SelectOp NewYieldProducer = b.create<arith::SelectOp>(OldYieldProducer->getLoc(), OldYieldProducer_Value, IterRegionOperand); 
      //   NewYieldProducer.getOperation()->moveBefore(yieldop);
      //   OldYieldProducer->replaceAllUsesWith(NewYieldProducer);
      //   NewYieldProducer.setOperand(0, OldYieldProducer_Value);
      // }
    }
    //   // IterOperand.dump();
    //   AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
    //   if(!IterOperand.isa<BlockArgument>() 
    //     && yieldop.getOperands().size() != 0){
    //     if(IsAccumulationInitialValue<arith::AddIOp>(forop, IterOperand)){
    //       MoveAccumulationInitialValue<arith::AddIOp>(forop, IterOperand);
    //     }
    //     else if(IsAccumulationInitialValue<arith::AddFOp>(forop, IterOperand)){
    //       MoveAccumulationInitialValue<arith::AddFOp>(forop, IterOperand);;
    //     }
    //   }
    // }
  });
  // kernel.dump();
}


// hjy
// =================== Begin If to Select with Store Sinking Support =============
// auxiliary struct to hold store information
struct StoreInfo {
    mlir::Operation* op;
    mlir::Value valueToStore;
    mlir::Value memref;
    llvm::SmallVector<mlir::Value, 4> indices;
};

// auxiliary function to check if two store operations are the same
static bool isSameLocation(const StoreInfo& a, const StoreInfo& b) {
    if (a.memref != b.memref) return false;
    if (a.indices.size() != b.indices.size()) return false;
    for (size_t i = 0; i < a.indices.size(); ++i) {
        if (a.indices[i] != b.indices[i]) return false;
    }
    return true;
}

// auxiliary function to find existing load value before ifOp
static mlir::Value findExistingLoadValue(scf::IfOp ifOp, mlir::Value memref, 
                                          llvm::ArrayRef<mlir::Value> indices) {
    // Get the block containing the ifOp
    mlir::Block* block = ifOp->getBlock();
    
    // Iterate backwards from ifOp to find matching load
    for (auto it = block->begin(); it != Block::iterator(ifOp); ++it) {
        mlir::Operation* op = &(*it);
        
        //Check if the operation is a load
        if (auto loadOp = dyn_cast<affine::AffineLoadOp>(op)) {
            if (loadOp.getMemref() == memref) {
                auto loadIndices = loadOp.getMapOperands();
                if (loadIndices.size() == indices.size()) {
                    bool match = true;
                    for (size_t i = 0; i < indices.size(); ++i) {
                        if (loadIndices[i] != indices[i]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        return loadOp.getResult();
                    }
                }
            }
        }
        // Check if the operation is a store
        else if (auto loadOp = dyn_cast<memref::LoadOp>(op)) {
            if (loadOp.getMemref() == memref) {
                auto loadIndices = loadOp.getIndices();
                if (loadIndices.size() == indices.size()) {
                    bool match = true;
                    for (size_t i = 0; i < indices.size(); ++i) {
                        if (loadIndices[i] != indices[i]) {
                            match = false;
                            break;
                        }
                    }
                    if (match) {
                        return loadOp.getResult();
                    }
                }
            }
        }
    }
    
    return mlir::Value(); // not found
}

// The main function to lower scf.if to select with store sinking support
static void lowerSCFIfToSelect(ADORA::KernelOp kernel) {
    llvm::SmallVector<scf::IfOp, 4> ifOps;
    kernel.walk([&](scf::IfOp op) {
        ifOps.push_back(op);
    });

    for (auto ifOp : ifOps) {
        OpBuilder builder(ifOp);
        
        // 1. handle return values first
        if (ifOp.getNumResults() > 0) {
            mlir::Block *thenBlock = &ifOp.getThenRegion().front();
            mlir::Block *elseBlock = &ifOp.getElseRegion().front();
            
            // move all non-yield operations out of the ifOp
            for (auto &op : llvm::make_early_inc_range(thenBlock->getOperations())) {
                if (!isa<scf::YieldOp>(op)) op.moveBefore(ifOp);
            }
            for (auto &op : llvm::make_early_inc_range(elseBlock->getOperations())) {
                if (!isa<scf::YieldOp>(op)) op.moveBefore(ifOp);
            }

            auto thenYield = cast<scf::YieldOp>(thenBlock->getTerminator());
            auto elseYield = cast<scf::YieldOp>(elseBlock->getTerminator());

            for (unsigned i = 0; i < ifOp.getNumResults(); ++i) {
                auto selectOp = builder.create<mlir::arith::SelectOp>(
                    ifOp.getLoc(), ifOp.getCondition(), thenYield.getOperand(i), elseYield.getOperand(i));
                ifOp.getResult(i).replaceAllUsesWith(selectOp.getResult());
            }
            ifOp.erase();
            continue; // skip to next ifOp
        }

        // 2. handle store operations when there is no return value
        if (ifOp.getNumResults() == 0) {
            mlir::Block *thenBlock = &ifOp.getThenRegion().front();
            mlir::Block *elseBlock = ifOp.getElseRegion().empty() ? nullptr : &ifOp.getElseRegion().front();

            // 2.1 separate store operations from other operations
            llvm::SmallVector<StoreInfo, 4> thenStores;
            llvm::SmallVector<StoreInfo, 4> elseStores;

            // Lambda to collect only store operations
            auto collectStoresOnly = [&](mlir::Block* block, llvm::SmallVector<StoreInfo, 4>& stores) {
                if (!block) return;
                for (auto &op : block->getOperations()) {
                    if (auto storeOp = dyn_cast<affine::AffineStoreOp>(op)) {
                        llvm::SmallVector<mlir::Value, 4> indices(storeOp.getMapOperands().begin(), 
                                                                   storeOp.getMapOperands().end());
                        stores.push_back({&op, storeOp.getValue(), storeOp.getMemref(), indices});
                    }
                    else if (auto storeOp = dyn_cast<memref::StoreOp>(op)) {
                        llvm::SmallVector<mlir::Value, 4> indices(storeOp.getIndices().begin(), 
                                                                   storeOp.getIndices().end());
                        stores.push_back({&op, storeOp.getValue(), storeOp.getMemref(), indices});
                    }
                }
            };

            // Lambda to move non-store operations out of the ifOp
            auto moveNonStoreOps = [&](mlir::Block* block) {
                if (!block) return;
                for (auto &op : llvm::make_early_inc_range(block->getOperations())) {
                    if (!isa<scf::YieldOp>(op) && 
                        !isa<affine::AffineStoreOp>(op) && 
                        !isa<memref::StoreOp>(op)) {
                        op.moveBefore(ifOp);
                    }
                }
            };

            // 2.1.1 collect store operations
            collectStoresOnly(thenBlock, thenStores);
            collectStoresOnly(elseBlock, elseStores);

            // 2.1.2 move non-store operations out of the ifOp
            moveNonStoreOps(thenBlock);
            moveNonStoreOps(elseBlock);

            // 2.2 handle all involved stores
            llvm::DenseSet<mlir::Operation*> processedElseOps;

            for (const auto& tStore : thenStores) {
                mlir::Value valTrue = tStore.valueToStore;
                mlir::Value valFalse;
                mlir::Operation* matchedElseOp = nullptr;

                // search for matching store in elseStores
                for (const auto& eStore : elseStores) {
                    if (isSameLocation(tStore, eStore)) {
                        valFalse = eStore.valueToStore;
                        matchedElseOp = eStore.op;
                        processedElseOps.insert(eStore.op);
                        break;
                    }
                }

                // if no matching store in elseStores, create a new load
                if (!matchedElseOp) {
                    // find existing load value
                    valFalse = findExistingLoadValue(ifOp, tStore.memref, tStore.indices);
                    
                    // if not found, create a new load
                    if (!valFalse) {
                        if (isa<affine::AffineStoreOp>(tStore.op)) {
                            auto origStore = cast<affine::AffineStoreOp>(tStore.op);
                            auto loadOp = builder.create<affine::AffineLoadOp>(
                                ifOp.getLoc(), origStore.getMemref(), 
                                origStore.getAffineMap(), origStore.getMapOperands());
                            valFalse = loadOp.getResult();
                        } else {
                            auto loadOp = builder.create<memref::LoadOp>(
                                ifOp.getLoc(), tStore.memref, tStore.indices);
                            valFalse = loadOp.getResult();
                        }
                    }
                }

                // create Select
                auto selectOp = builder.create<mlir::arith::SelectOp>(
                    ifOp.getLoc(), ifOp.getCondition(), valTrue, valFalse);

                // create new unconditional Store (after If)
                if (isa<affine::AffineStoreOp>(tStore.op)) {
                    auto origStore = cast<affine::AffineStoreOp>(tStore.op);
                    builder.create<affine::AffineStoreOp>(
                        ifOp.getLoc(), selectOp.getResult(), origStore.getMemref(),
                        origStore.getAffineMap(), origStore.getMapOperands());
                } else {
                    builder.create<memref::StoreOp>(
                        ifOp.getLoc(), selectOp.getResult(), tStore.memref, tStore.indices);
                }
            }

            // handle elseStores that were not processed
            for (const auto& eStore : elseStores) {
                if (processedElseOps.count(eStore.op)) continue;

                mlir::Value valTrue; 
                mlir::Value valFalse = eStore.valueToStore;

                // search for matching store in thenStores
                valTrue = findExistingLoadValue(ifOp, eStore.memref, eStore.indices);
                
                // if not found, create a new load
                if (!valTrue) {
                    if (isa<affine::AffineStoreOp>(eStore.op)) {
                        auto origStore = cast<affine::AffineStoreOp>(eStore.op);
                        auto loadOp = builder.create<affine::AffineLoadOp>(
                            ifOp.getLoc(), origStore.getMemref(),
                            origStore.getAffineMap(), origStore.getMapOperands());
                        valTrue = loadOp.getResult();
                    } else {
                        auto loadOp = builder.create<memref::LoadOp>(
                            ifOp.getLoc(), eStore.memref, eStore.indices);
                        valTrue = loadOp.getResult();
                    }
                }

                // create Select
                auto selectOp = builder.create<mlir::arith::SelectOp>(
                    ifOp.getLoc(), ifOp.getCondition(), valTrue, valFalse);

                // create new unconditional Store (after If)
                if (isa<affine::AffineStoreOp>(eStore.op)) {
                    auto origStore = cast<affine::AffineStoreOp>(eStore.op);
                    builder.create<affine::AffineStoreOp>(
                        ifOp.getLoc(), selectOp.getResult(), origStore.getMemref(),
                        origStore.getAffineMap(), origStore.getMapOperands());
                } else {
                    builder.create<memref::StoreOp>(
                        ifOp.getLoc(), selectOp.getResult(), eStore.memref, eStore.indices);
                }
            }

            // 2.3 erase the ifOp
            ifOp.erase();
        }
    }
}

/**
 * 
 * A tool function to Insert ISEL operator if init-xxxxx-yield chain is not complete
 * For example:
              xxx
 * 
 * There are two circumstances that need to add Isel to yield op.
 * 1: If RegionOperand is yield immediately, and IterRegionOperandIdx is not paired with yield
 * 2: If compute op before yield is select
*/
void InsertIselForLoopCarry(ADORA::KernelOp kernel, bool verbose){
  OpBuilder b(kernel);
  kernel.walk([&](affine::AffineForOp forop){
    if(verbose) forop.dump();
    int IterRegionOperandIdx;
    for(IterRegionOperandIdx = 0; IterRegionOperandIdx < forop.getNumRegionIterArgs(); IterRegionOperandIdx++){
      mlir::Value IterRegionOperand = forop.getRegionIterArgs()[IterRegionOperandIdx];
      if(verbose) IterRegionOperand.dump();

      /// get yieldop 
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());

      /// check RegionOperand is yield immediately
      int YieldIndex = GetYieldIndexFromValue(yieldop, IterRegionOperand);

      /// If IterRegionOperandIdx and yield is not paired
      if(YieldIndex != -1 && IterRegionOperandIdx != YieldIndex){
        (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
      }
      /// If compute op before yield is not acc type, insert isel op
      else if(YieldIndex == -1
          && !checkAccumulationChain<arith::AddIOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::AddFOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::MulIOp>(forop, IterRegionOperand, IterRegionOperandIdx)
          && !checkAccumulationChain<arith::MulFOp>(forop, IterRegionOperand, IterRegionOperandIdx)){
        (void)ReplaceLoopCarryValueWithNewIselOp(forop, IterRegionOperandIdx);
      }
    }
  });
  // kernel.dump();
}

SmallVector<affine::AffineForOp> SortForVec_InToOutLevels(SmallVector<affine::AffineForOp> ForVec){
  SmallVector<affine::AffineForOp> NewForVec;
  // bubble sort
  for(int i = 0; i < ForVec.size() - 1; i ++){
    for(int j = 0; j < ForVec.size() - 1 - i; j++){ 
      //  ForVec[i].dump();
      //  ForVec[j].dump();
      auto result = ForVec[j].walk([&](affine::AffineForOp f)-> WalkResult 
      {
        if(f == ForVec[j + 1])// loop j + 1 is inside loop j, put j + 1 to tail
          return WalkResult::interrupt();
        else
          return WalkResult::advance();
      });
      if(result == WalkResult::interrupt()){
        // change position
        auto temp = ForVec[j];
        ForVec[j] = ForVec[j + 1];
        ForVec[j + 1] = temp;
      }
    } 
  }
  // for(auto for_ : ForVec){
  //   for_.dump();
  // }
  return ForVec;
}

std::string LinearAccessToStr(mlir::SmallVector<std::pair<int64_t, int64_t>> Vec){
  std::string str;
  for(auto elem : Vec){
    str += std::to_string(elem.first) + "," +  std::to_string(elem.second) + ",";
    // llvm::errs() << str;
  }
  return str.substr(0, str.length() - 1); // delete the last ", "
}

std::string LinearAccessToStr(mlir::SmallVector<std::pair<std::string, std::string>> Vec){
  std::string str;
  for(auto elem : Vec){
    str += elem.first + "," +  elem.second + ",";
    // llvm::errs() << str;
  }
  return str.substr(0, str.length() - 1); // delete the last ", "
}


template <typename LoadOrStoreOp>
  int64_t GetInitAddr(LoadOrStoreOp lsop, std::map<mlir::Operation*, int> For_loop_level){
  OpBuilder b(lsop);
  Operation::operand_range loadIndices = lsop.getIndices();
  ::mlir::AffineMap map = lsop.getAffineMapAttr().getValue();

  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  ArrayRef<int64_t>  Shape = memRefType.getShape();
  int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
  // map.dump();
  SmallDenseMap<AffineForOp, int64_t> ForToLb;
  mlir::SmallVector<int64_t> initPosition_eachrank;
  for(unsigned r = 0; r < map.getResults().size(); r++ ){
    AffineExpr expr = map.getResult(r);
    AffineMap lb_new_map;
    SmallVector<AffineExpr> dim_to_expr;
    for(unsigned d = 0; d < loadIndices.size(); d++){
      // loadIndices[d].dump();
      if(expr.isFunctionOfDim(d)){
        if (loadIndices[d].isa<BlockArgument>()){
          //// a block arguement of for op
          AffineForOp forop = dyn_cast<AffineForOp>(loadIndices[d].getParentBlock()->getParentOp());
          assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
          // get lower bound of this dim
          assert(forop.getLowerBoundMap().getResults().size() == 1);
          AffineExpr lbExpr = forop.getLowerBoundMap().getResult(0);
          assert(lbExpr.getKind() == AffineExprKind::Constant);
          int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
          dim_to_expr.push_back(b.getAffineConstantExpr(lb));
        }
        else if(!loadIndices[d].isa<BlockArgument>() && isa<arith::ConstantOp>(loadIndices[d].getDefiningOp())){
          arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(loadIndices[d].getDefiningOp());
          mlir::Attribute constattr = loadIndices[d].getDefiningOp()->getAttr(constop.getValueAttrName());
          assert(isa<IntegerAttr>(constattr));
          IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
          int64_t value = intattr.getInt();   
          dim_to_expr.push_back(b.getAffineConstantExpr(value));        
        }
        else if(!loadIndices[d].isa<BlockArgument>() && isa<affine::AffineApplyOp>(loadIndices[d].getDefiningOp())){
          affine::AffineApplyOp applyop = dyn_cast<affine::AffineApplyOp>(loadIndices[d].getDefiningOp());
          mlir::AffineMap applymap = applyop.getAffineMap();
          // loadIndices[d].dump();
          // applymap.dump();
          assert(applymap.getResults().size() == 1 && applymap.getNumDims() == 1);
          // mlir::AffineExpr constpart = getConstPartofAffineExpr(applymap.results()[0]);
          SmallVector<AffineExpr, 4> dimReplacements(applymap.getNumDims());
          for(unsigned d = 0; d < applyop.getMapOperands().size(); d++){
            mlir::Value operand = applyop.getMapOperands()[d];
            assert(operand.isa<BlockArgument>());
            AffineForOp forop = dyn_cast<AffineForOp>(operand.getParentBlock()->getParentOp());
            assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
            // get lower bound of this dim
            assert(forop.getLowerBoundMap().getResults().size() == 1);
            AffineExpr lbExpr = forop.getLowerBoundMap().getResult(0);
            assert(lbExpr.getKind() == AffineExprKind::Constant);
            // int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
            dimReplacements[d] = lbExpr;
          }
          applymap = applymap.replaceDimsAndSymbols(dimReplacements, {}, applymap.getNumDims(), applymap.getNumSymbols());
          // applymap.dump();
          dim_to_expr.push_back(applymap.getResults()[0]);
        }
        else{
          assert(0 && "Only supported arith::ConstantOp and affineForOp now.");
        }

      }
      else {
        dim_to_expr.push_back(b.getAffineDimExpr(d));
      }
    }
    // lb_new_map = 
    lb_new_map = map.compose(AffineMap::get(loadIndices.size(), 0, ArrayRef<AffineExpr>(dim_to_expr), b.getContext()));
    // llvm::errs() << "[test] lb_new_map: " ; lb_new_map.dump();
    assert(lb_new_map.getResult(r).getKind() == AffineExprKind::Constant);
    int64_t init_position_thisrank = lb_new_map.getResult(r).dyn_cast<AffineConstantExpr>().getValue();
    initPosition_eachrank.push_back(init_position_thisrank);
  }

  int64_t init_position = 0;
  for(unsigned r = 0; r < Shape.size(); r++ ){
    int64_t elements_each_step_inner = 1;
    for (unsigned i = r + 1; i < Shape.size(); i++){
      // llvm::errs()<<"Shape[i]: " << Shape[i] << "\n";
      elements_each_step_inner *= Shape[i];
    }
    init_position += initPosition_eachrank[r] * elements_each_step_inner;
    // llvm::errs()<<"init_position: " << init_position 
    //             <<",initPosition[r]: " << initPosition_eachrank[r] 
    //             <<",elements_each_step_inner: " << elements_each_step_inner
    //             << "\n";
  }
  // return LinearAccess;
  return init_position * ElementBytes;
}

template <typename LoadOrStoreOp>
  int64_t GetMemrefSize(LoadOrStoreOp lsop){
  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  ArrayRef<int64_t>  Shape = memRefType.getShape();
  int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
  // map.dump();

  int64_t elements = 1;
  for(unsigned r = 0; r < Shape.size(); r++ ){
    elements *= Shape[r];
  }

  return elements * ElementBytes;
}

/// Get the string of compare operation's type, for example:
///      %7 = arith.cmpf ugt, %arg10, %6 : f32
/// return: ugt
std::string GetCMPTypeStr(mlir::Operation* op){
  std::string cmptype;
  if(isa<arith::CmpIOp>(op)){
    arith::CmpIOp cmpop = dyn_cast <arith::CmpIOp> (op);
    arith::CmpIPredicate cmppred = cmpop.getPredicate();
    cmptype = stringifyCmpIPredicate(cmppred);
    if(cmptype == "eq") return "EQ";
    else if(cmptype == "ne") return "NE";
    else if(cmptype == "ult") return "ULT";
    else if(cmptype == "ule") return "ULE";
    else if(cmptype == "ugt") return "UGT";
    else if(cmptype == "uge") return "UGE";
        //hjy
    // add signed integer compare
    else if(cmptype == "slt") return "SLT";
    else if(cmptype == "sle") return "SLE";
    else if(cmptype == "sgt") return "SGT";
    else if(cmptype == "sge") return "SGE";
    else assert(0 && "Unsupported compare type.");
  }
  else if(isa<arith::CmpFOp>(op)){
    arith::CmpFOp cmpop = dyn_cast <arith::CmpFOp> (op);
    arith::CmpFPredicate cmppred = cmpop.getPredicate();
    cmptype = stringifyCmpFPredicate(cmppred);
    if(cmptype == "ueq") return "FEQ";
    else if(cmptype == "une") return "FNE";
    else if(cmptype == "ugt") /*return "FUGT32";*/return "FOGT";
    else if(cmptype == "uge") /*return "FUGE32";*/return "FOGE";
    else if(cmptype == "ult") /*return "FULT32";*/return "FOLT";
    else if(cmptype == "ule") /*return "FULE32";*/return "FOLE";
    /*CGRA only support ordered float computing in current version.*/

    else if(cmptype == "oeq") return "FEQ";
    else if(cmptype == "one") return "FNE";
    else if(cmptype == "ogt") return "FOGT";
    else if(cmptype == "oge") return "FOGE";
    else if(cmptype == "olt") return "FOLT";
    else if(cmptype == "ole") return "FOLE";

    else if(cmptype == "uno") /*return "FULE32";*/return "FUNO";

    else assert(0 && "Unsupported compare type.");   
  }
  else 
    assert(0 && "Not a compare operation.");
  
  return cmptype;
}
/// Sometimes the PEs only support "less than(lt)" and "less equal(le)"
/// so we convert greater to less
bool ConvertGreaterToLess(LLVMCDFGNode* node){
  if(node->getTypeName() == "UGT") node->setTypeName("ULT");
  else if(node->getTypeName() == "UGE") node->setTypeName("ULE");
  else if(node->getTypeName() == "FUGT") node->setTypeName("FULT");
  else if(node->getTypeName() == "FUGE") node->setTypeName("FULE");
  else if(node->getTypeName() == "FOGT") node->setTypeName("FOLT");
  else if(node->getTypeName() == "FOGE") node->setTypeName("FOLE");
    //hjy
  // add signed integer compare
  else if(node->getTypeName() == "SGT") node->setTypeName("SLT");
  else if(node->getTypeName() == "SGE") node->setTypeName("SLE");
  else return true;

  /// exchange operand idx
  if(node->inputEdges().size() != 2){
    return false;
  }

  auto input0 = node->getInputPort(0);
  auto input1 = node->getInputPort(1);
  node->setInputIdx(input0, 1);
  node->setInputPort(input0, 1);
  node->setInputIdx(input1, 0);
  node->setInputPort(input1, 0);

  return true;
}

AffineYieldOp getOldestAncestorYieldOp(AffineForOp forop){
  AffineYieldOp yieldop =  dyn_cast<AffineYieldOp>(forop.getBody()->getTerminator());
  assert(yieldop.getOperands().size() == 1);
  mlir::Operation* operandop = yieldop.getOperand(0).getDefiningOp();
  if(isa<AffineForOp>(operandop)){
    return getOldestAncestorYieldOp(dyn_cast<AffineForOp>(operandop));
  }
  else {
    return yieldop;
  }
  
}

bool isInteger(const std::string& str) {
    // Empty string or just a negative/positive sign is not a valid integer
    if (str.empty() || ((str[0] == '-' || str[0] == '+') && str.size() == 1)) {
        return false;
    }

    // Check each character to see if it's a digit (allowing an optional leading sign)
    for (size_t i = (str[0] == '-' || str[0] == '+') ? 1 : 0; i < str.size(); ++i) {
        if (!std::isdigit(str[i])) {
            return false;
        }
    }
    return true;
}

#define Define_Polymorphism_Of_StringIntArith(funcname) \
  std::string funcname (const int& LHS, const int& RHS){  \
    return funcname(std::to_string(LHS), std::to_string(RHS)); } \
  std::string funcname (const std::string& LHS, const int& RHS){  \
    return funcname(LHS, std::to_string(RHS)); } \
  std::string funcname (const int& LHS, const std::string& RHS){  \
    return funcname(std::to_string(LHS), RHS); }

////
/// Multiply two parameters from affine for op
///   three situations:
///   1 int * int = int
///   2 arg * int = arg
///   3 arg * arg = arg
std::string MulAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) * std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 1)
      return RHS;
    else if(std::stoi(RHS) == 1)
      return LHS;
    else
      return LHS + "*" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(MulAsStr)

////
/// Add two parameters from affine for op
///   three situations:
///   1 int + int = int
///   2 arg + int = arg
///   3 arg + arg = arg
std::string AddAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) + std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 0)
      return RHS;
    else if(std::stoi(RHS) == 0)
      return LHS;
    else
      return LHS + "+" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(AddAsStr)

////
/// Add two parameters from affine for op
///   four situations:
///   1 int - int = int
///   2 arg - int = arg
///   3 int - arg = arg
///   4 arg - arg = arg
std::string SubAsStr (const std::string& LHS, const std::string& RHS){
  if(isInteger(LHS) && isInteger(RHS)){
    return std::to_string(std::stoi(LHS) - std::stoi(RHS));
  }
  else{
    if(std::stoi(LHS) == 0)
      return "-" + RHS;
    else if(std::stoi(RHS) == 0)
      return LHS;
    else
      return LHS + "-" + RHS;
  }
}
Define_Polymorphism_Of_StringIntArith(SubAsStr)




//// Get the outter level of one operation in one kernel
std::string getOuterLoopTotalTripcountUntilKernel(mlir::Operation* op){
  // std::string total_tripcount_str = "1";
  int tripcount = 1;
  if(isa<affine::AffineForOp>(op->getParentOp())){
    tripcount = getConstantTripCount(dyn_cast<affine::AffineForOp>(op->getParentOp())).value_or(0);
    assert(tripcount != 0);

    // if(isInteger(total_tripcount_str)) 
    //   total_tripcount = std::stoi(total_tripcount_str);

    return MulAsStr(std::to_string(tripcount), getOuterLoopTotalTripcountUntilKernel(op->getParentOp()));
  }
  else if(isa<ADORA::KernelOp>(op->getParentOp())){
    return std::to_string(1);
  }
  else{
    assert(false && "ADORA Kernel is wrong.");
  }
}

//// Get the trip count of forop, return as a string, for example:
///     affine.for %arg7 = 0 to 28 return 28
///     affine.for %arg1 = 0 to %arg2 return "arg2"
///     for nonconstant for op, we can only handle following 2 types:
///       affine.for %arg1 = 0 to %arg2
///       affine.for %arg1 = 0 to 2000 - %arg2
static std::string getTripCountAsStr(affine::AffineForOp forop){
  OpBuilder b(forop);
  int tripcount = getConstantTripCount(forop).value_or(0); 
  if(tripcount == 0){
    /// Non-const iteration space
    AffineMap map;
    SmallVector<mlir::Value> operands;
    getTripCountMapAndOperands(forop, &map, &operands);
    LLVM_DEBUG(
      llvm::errs() << "[MION] Non-const iteration space: ";
      map.dump();
      for(auto operand : operands){
        llvm::errs() << operand << " ";
      }
      llvm::errs() << "\n";
    );
    assert(map.getNumDims() + map.getNumSymbols() == 1 && "Unsupported trip count!");
    assert(map.getNumResults() == 1 && operands.size() == 1 && "Unsupported trip count!");
    AffineExpr expr = map.getResult(0);
    LLVM_DEBUG(expr.dump(););
    switch (expr.getKind())
      {
      case AffineExprKind::DimId : 
      case AffineExprKind::SymbolId : {
        mlir::Value arg_v = operands[0]; ///operands.size() == 1
        // assert(!IsInKernel(arg_v.getOperation()));
        
        Location loc = _kernel_toDFG->getLoc(); ////// Fix this
        // loc.dump();
        // LLVM_DEBUG(_kernel_toDFG->dump(););
        b.setInsertionPointToStart(_kernel_toDFG->getOperation()->getBlock());
        arith::ConstantOp cst = b.create<arith::ConstantOp>(loc, b.getIntegerAttr(b.getIndexType(), 0));
        arith::AddIOp newadd = b.create<arith::AddIOp>(loc, arg_v, cst);
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););

        newadd.getOperation()->moveBefore(_kernel_toDFG->getOperation());
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););
        
        cst.getOperation()->moveBefore(newadd.getOperation());
        LLVM_DEBUG(newadd.getOperation()->getBlock()->dump(););

        //// denote the name of the generated operation
        StringAttr strattr = StringAttr::get(newadd.getOperation()->getContext(),
                                              "VARCFG_" + std::to_string(_variable_config_cnt));
        newadd.getOperation()->setAttr("VAR_CONFIG", strattr);
        // _variable_config_cnt++;
        LLVM_DEBUG(forop.dump(););
        
        return "VARCFG_" + std::to_string(_variable_config_cnt++);
      }

      // case AffineExprKind::Add : ///TODO: FIX THIS

      default :{
        assert(false && "Unsupported trip count!");
        return "";
      }
    }
  }
  else{
    return std::to_string(tripcount);
  }
}


////
// get linear access from load or store op
template <typename LoadOrStoreOp>
  mlir::SmallVector<std::pair<std::string, std::string>> \
    GetLinearAccess(LoadOrStoreOp lsop, std::map<mlir::Operation*, int> For_loop_level){
  Operation::operand_range loadIndices = lsop.getIndices();
  ::mlir::AffineMap map = lsop.getAffineMapAttr().getValue();
  MemRefType memRefType = lsop.getMemref().getType().template cast<MemRefType>();
  // map.dump();
  SmallDenseMap<AffineForOp, SmallVector<int64_t>> ForToRanks;
  // SmallVector<AffineForOp> forVec;
  for(unsigned d = 0; d < loadIndices.size(); d++){
    if(!loadIndices[d].isa<BlockArgument>() && isa<arith::ConstantOp>(loadIndices[d].getDefiningOp())){
      /// Constant index bias doesn't contribute to linear access.
      continue;
    }
    // llvm::errs() << "[test] loadIndice[i]: " ; loadIndices[d].dump() ; 
    AffineForOp forop = dyn_cast<AffineForOp>(loadIndices[d].getParentBlock()->getParentOp());
    assert(isa<AffineForOp>(forop) && "AffineLoadOp or StoreOp 's parent op should be AffineForOp!");
    // AffineForOp parentFor = dyn_cast<AffineForOp>(*forop);
    // llvm::errs() << "[test] forop: " ; forop.dump();

    /// For every dim of the affine map,  add the corresponding Multiplicator to ForToRanks
    for(unsigned r = 0; r < map.getResults().size(); r++ ){
      AffineExpr expr = map.getResult(r);
      // expr.dump();
      if(expr.isFunctionOfDim(d)){
        // find the corresponding rank
        ForToRanks[forop].push_back(ADORA::MultiplicatorOfDim(expr, d));
        // llvm::errs() << d <<": " << ADORA::MultiplicatorOfDim(expr, d) << ",";
      } 
      else {
        ForToRanks[forop].push_back(0);
      }
    }
    // assert(findElement(forVec, forop)==-1 && "For op should only be in the indices for one time.");
    // forVec.push_back(forop);
  }
  // forVec = SortForVec_InToOutLevels(forVec);

  mlir::SmallVector<std::pair<std::string, std::string>> LinearAccess;
  for(unsigned level = 0; level < For_loop_level.size(); level ++){
    /// For a new recursion of this level,
    ///   elements_each_step = RM * STEP * RANK_SHAPE
    affine::AffineForOp forop;
    for(auto loop_level : For_loop_level){
      if(loop_level.second == level){
        assert(isa<affine::AffineForOp>(loop_level.first) && "We can only handle affine for now.");
        forop = dyn_cast<affine::AffineForOp>(loop_level.first);
        break;
      }
    }
    SmallVector<int64_t> RankMultiplicators = ForToRanks[forop];
    int64_t ElementBytes = memRefType.getElementTypeBitWidth()/8;
    // std::string tripcount = getConstantTripCount(forop).value_or(0); 
    std::string tripcount = getTripCountAsStr(forop);
    // LLVM_DEBUG(_kernel_toDFG->dump(););
    // total_count_str = MulAsStr(total_count_str, tripcount);
    // LLVM_DEBUG(_kernel_toDFG->dump(););

    int64_t elements_each_step = 0;
    bool RM_flag = false;
    for(unsigned r = 0; r < RankMultiplicators.size(); r++){
      if(RankMultiplicators[r] == 0){
        continue;
      }
      else{
        assert(RM_flag == false && "This for loop should only be corresponding to one rank.");
        elements_each_step = forop.getStep().getSExtValue() * RankMultiplicators[r];
        ArrayRef<int64_t>  Shape = memRefType.getShape();
        for (unsigned i = r + 1; i < Shape.size(); i++){
          elements_each_step *= Shape[i];
        }
        RM_flag = true;
      }
    }

    /// For the last old recursion ,
    //   end_position = lb0 + (tripcount0-1) * step0 * rm0 * rank0 + lb1 + (tripcount1-1) * step1 * rm1 * rank1 + ...
    std::string end_position = "0";
    for(unsigned innerlevel = 0; innerlevel < level; innerlevel++){
      affine::AffineForOp innerforop;
      for(auto loop_level : For_loop_level){
        if(loop_level.second == innerlevel){
          assert(isa<affine::AffineForOp>(loop_level.first) && "We can only handle affine for now.");
          innerforop = dyn_cast<affine::AffineForOp>(loop_level.first);
          break;
        }
      }
      // int64_t innertripcount = getConstantTripCount(innerforop).value_or(0); 
      std::string innertripcount = getTripCountAsStr(innerforop);
      SmallVector<int64_t> innerRMs = ForToRanks[innerforop];
      
      // get lower bound of this dim
      assert(innerforop.getLowerBoundMap().getResults().size() == 1);
      AffineExpr lbExpr = innerforop.getLowerBoundMap().getResult(0);
      assert(lbExpr.getKind() == AffineExprKind::Constant);
      int64_t lb = lbExpr.dyn_cast<AffineConstantExpr>().getValue();
      // llvm::errs()<< "lbmap: " << lbmap << "\n"; 

      bool innerRM_flag = false;
      int64_t elements_each_step_inner = 0;
      for(unsigned r = 0; r < innerRMs.size(); r++){
        if(innerRMs[r] == 0){
          continue;
        }
        else{
          assert(innerRM_flag == false && "This for loop should only be corresponding to one rank.");
          elements_each_step_inner = innerforop.getStep().getSExtValue() * innerRMs[r];
          ArrayRef<int64_t>  Shape = memRefType.getShape();
          for (unsigned i = r + 1; i < Shape.size(); i++){
            // llvm::errs()<<"Shape[i]: " << Shape[i] << "\n";
            elements_each_step_inner *= Shape[i];
          }
          innerRM_flag = true;
        }
      }
      //// end_position += lb + elements_each_step_inner * innertripcount;
      end_position = AddAsStr(end_position, 
                      AddAsStr(lb, 
                        MulAsStr(elements_each_step_inner, 
                          SubAsStr(innertripcount, 1)))); 
      // llvm::errs()<<"end_position: " << end_position 
      //           << ", elements_each_step_inner:" << elements_each_step_inner
      //           << ", lb:" << lb
      //           <<"\n";
    }

    /// addr for new recursion
    std::string addrstep = MulAsStr(SubAsStr(elements_each_step, end_position), ElementBytes);
    LLVM_DEBUG(llvm::errs()<<"addrstep: " << addrstep
                << ", ElementBytes:" << ElementBytes
                << ", elements_each_step:" << elements_each_step
                << ", end_position:" << end_position
                <<"\n";);

    LinearAccess.push_back(std::pair(addrstep, tripcount));
  }
  
  LLVM_DEBUG(llvm::errs()<<"\n" << LinearAccessToStr(LinearAccess)<<"\n";);
  return LinearAccess;
}



////
// Get accumulation information from a yield-for-yield-for.... chain in a recursive method.
// vector count_interval_repeat:
//  $interval: accumate every $interval cycles
//  $count: PE be set to initial value every $count times of accumulation
//  $repeat: above operation will be repeat for $repeat times
SmallVector<std::string, 3> GetACCInfoFromYieldNode(LLVMCDFGNode* yieldnode, SmallVector<std::string, 3>& count_interval_repeat/*, int YieldIndex = 0*/){
  LLVM_DEBUG(_kernel_toDFG->dump(););
  
  assert(count_interval_repeat.size() == 3);
  assert(yieldnode->getTypeName() == "yield");
  // assert(yieldnode->inputNodes().size() == 1);
  assert(yieldnode->outputNodes().size() == 1);

  std::string total_count_str = count_interval_repeat[0];
  std::string total_interval_str = count_interval_repeat[1];
  std::string total_repeat_str = count_interval_repeat[2];
  std::string tripcount_str;

  //// get count
  LLVMCDFGNode* SuccNode = yieldnode->outputNodes()[0];
  assert(SuccNode->getTypeName() == "for");
  AffineForOp forop = dyn_cast<AffineForOp>(SuccNode->operation());
  assert(IsIterationSpaceSupported(forop));
  std::string tripcount = getTripCountAsStr(forop);
  LLVM_DEBUG(_kernel_toDFG->dump(););
  total_count_str = MulAsStr(total_count_str, tripcount);
  LLVM_DEBUG(_kernel_toDFG->dump(););

  // LLVMCDFGNode* AnceNode = yieldnode->inputNodes()[0]; 
  // if(AnceNode->getTypeName() == "for"){
  //   forop = dyn_cast<AffineForOp>(AnceNode->operation());
  //   int tripcount = getConstantTripCount(forop).value_or(0); 
  // }
  // else{
  //   /// This loop level limits the count of acc.
  //   int tripcount = getConstantTripCount(forop).value_or(0); 
  //   if(tripcount == 0){
  //     /// Non-const iteration space

  //   }
  // }

  count_interval_repeat[0] = total_count_str;
  count_interval_repeat[1] = total_interval_str;

  // assert(SuccNode->outputNodes().size() == 1);
  LLVMCDFGNode* NextYieldNode = nullptr;
  for(LLVMCDFGNode* nextnode : SuccNode->outputNodes()){
    // nextnode->operation()->dump();
    if(!nextnode->isInputBackEdge(SuccNode)){
      /// backedge is a loop-carried variable
      NextYieldNode = nextnode;
      break;
    }
  }
  if(NextYieldNode!=nullptr && NextYieldNode->getTypeName() == "yield"){
    /// Get the yield index of this yielded value in the outer for level.
    // AffineYieldOp outeryieldop =  dyn_cast<AffineYieldOp>(outerFor.getBody()->getTerminator());
    // int OuterIndex = GetYieldIndexFromValue(dyn_cast<affine::YieldOp>(NextYieldNode->operation()), forop.getResults()[index]);
    return GetACCInfoFromYieldNode(NextYieldNode, count_interval_repeat/*, OuterIndex*/);
  }
  else{
    /// get repeat
    std::string temp = getOuterLoopTotalTripcountUntilKernel(forop.getOperation());
    total_repeat_str = MulAsStr(total_repeat_str, temp);
    count_interval_repeat[2] = total_repeat_str; 
    /// recursion should stop.
    return count_interval_repeat;
  }
}

/// Delete the yield-for-yield-for... chain in CDFG
bool DeleteYield(LLVMCDFG* CDFG, LLVMCDFGNode* yieldnode){
  assert(yieldnode->getTypeName() == "yield");
  assert(yieldnode->outputNodes().size() == 1);
  LLVMCDFGNode* SuccNode = yieldnode->outputNodes()[0];
  assert(SuccNode->getTypeName() == "for");

  for(LLVMCDFGNode* AnceNode : yieldnode->inputNodes()){
    std::map<LLVMCDFGNode*, std::vector<int>> OutputNodeToIdx;
    mlir::Operation* AnceOp = AnceNode->operation();
    
    assert(AnceOp->getResults().size() == 1);
    int YieldIndex = getYieldIndexOfValue(dyn_cast<affine::AffineYieldOp>(yieldnode->operation()), AnceOp->getResult(0));
    
    /// get output nodes
    affine::AffineForOp forop = dyn_cast<affine::AffineForOp>(SuccNode->operation());
    for(LLVMCDFGNode* outputnode : SuccNode->outputNodes()){ /// output from for
      mlir::Operation* OutputOp = outputnode->operation();
      if( !outputnode->isInputBackEdge(SuccNode) 
        && ValueIsInOperands(forop.getResult(YieldIndex), OutputOp)){
        for(int operandidx = 0; operandidx < OutputOp->getOperands().size(); operandidx++){
          /// only corresponding input port index should be connected
          if(OutputOp->getOperand(operandidx) == forop.getResult(YieldIndex)){
            OutputNodeToIdx[outputnode].push_back(operandidx);
          }
        }
        /// backedge is a loop-carried variable
        // AfterForNode = nextnode;
      }
    }
    
    /// connect AnceNode and output node
    for(auto _pair: OutputNodeToIdx){
      LLVMCDFGNode* outputnode = _pair.first;
      std::vector<int> indices = _pair.second;
      for(int idx : indices){
        outputnode->addInputNode(AnceNode,  idx, /*isBackEdge=*/false);
        AnceNode->addOutputNode(outputnode, /*isBackEdge=*/false);
        CDFG->addEdge(AnceNode, outputnode);  
      }
    }
  }



  // AfterForNode->operation()->dump();
  // AffineForOp forop = dyn_cast<AffineForOp>(SuccNode->operation());
  // if(AfterForNode->getTypeName() == "yield"){
  // LLVMCDFGNode* NextForNode = AfterForNode->outputNodes()[0];
  // assert(NextForNode->getTypeName() == "for");
  // NextForNode->operation()->dump();

  CDFG->delNode(yieldnode);
  CDFG->delNode(SuccNode);

  return true;
  // }
  // else{
  //   return false;
  // }
}


///////
// A data bit_cast function. The data will be organized as a vector<unsigned char> which stores the hex byte.
// String is not suitable because string stores signed char I think.
//////
template <typename DataT>
  std::vector<unsigned char> DataBitCastToHex(DataT data){
  std::stringstream result;
    // std::string a;
  unsigned char *hex = (unsigned char*) (&data);
  unsigned length = sizeof(DataT);
  std::vector<unsigned char> c;
  // llvm::errs() << "\ndata:" << data  << "\n";
  for(int i = 0; i < length; i++){
    c.push_back(uint32_t(hex[i]));
  }
  // std::string str;
  // int str_len = str.size();
  // for(int i = length - 1; i >= 0 ; i--){
  //   result<< std::hex << std::uppercase << std::setfill('0') << std::setw(2) << uint32_t(c[i]);
  //   std::cout << std::hex << std::setw(2)  << uint32_t(c[i]) << std::endl;
  // }
  // llvm::errs() << "\nstr:" << result.str()  << "\n";

  // DataT* data_back = (DataT*)(unsigned char*) &(c[0]);
  // llvm::errs() << "\ndata_back:" << *data_back  << "\n";
  return c;
}




///////
// Get Initial mlir Value from a for op
//////
static mlir::Value getInitialValueFromFor(affine::AffineForOp forop, int index = 0){
  // assert(forop.getNumRegionIterArgs() == 1);
  if(forop.getNumRegionIterArgs() == 1){
    assert(index == 0);
  }

  if(forop.getInits()[index].isa<BlockArgument>()){
    affine::AffineForOp outerFor = dyn_cast<affine::AffineForOp>(forop.getInits()[index].getParentBlock()->getParentOp());
    assert(outerFor && "Outer operation is not affine for.");

    /// Get the yield index of this yielded value in the outer for level.
    AffineYieldOp outeryieldop =  dyn_cast<AffineYieldOp>(outerFor.getBody()->getTerminator());
    int OuterIndex = GetYieldIndexFromValue(outeryieldop, forop.getResults()[index]);
    assert(OuterIndex != -1);

    return getInitialValueFromFor(dyn_cast<affine::AffineForOp>(forop.getOperation()->getParentOp()), OuterIndex);
  }
  else{
    return forop.getInits()[index];
  }
}
static mlir::Value getInitialValueFromYieldIndex(affine::AffineYieldOp yield, int index){
  return getInitialValueFromFor(dyn_cast<affine::AffineForOp>(yield.getOperation()->getParentOp()), index);
}


void setConstantNode(LLVMCDFGNode* node){
  mlir::Operation* op = node->operation();
  arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(op);
  std::string value_str;
  // llvm::errs() << constop.getValue();
  // mlir::Type constTy = constop.getValue().getType();
  mlir::Attribute constattr = op->getAttr(constop.getValueAttrName());
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    if(floatattr.getType().isF64()){
      // unsigned char *hex = (unsigned char*) (&value);
      // int hex_int = (int)*hex;
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value);
      // llvm::errs() << "value:"<< value << ", hex:"  << ConstCal_hex ;
      node->setConstValHex(ConstCal_hex);
      // llvm::errs() << "value:"<< value << ", hex:" << *hex << ", int:" << hex_int;
      node->setDataBits(64);
    } 
    else if(floatattr.getType().isF32()){
      float value_float = (float) value;
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_float);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
    else if (floatattr.getType().isBF16()) {
      llvm::APFloat apf = floatattr.getValue();

      bool losesInfo;
      apf.convert(llvm::APFloat::BFloat(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);

      uint16_t raw = apf.bitcastToAPInt().getZExtValue();

      std::vector<unsigned char> ConstCal_hex;
      ConstCal_hex.push_back(static_cast<unsigned char>(raw & 0xFF));
      ConstCal_hex.push_back(static_cast<unsigned char>((raw >> 8) & 0xFF));

      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
    else if (floatattr.getType().isF16()) {
      llvm::APFloat apf = floatattr.getValue();
      bool losesInfo;
      apf.convert(llvm::APFloat::IEEEhalf(),
                  llvm::APFloat::rmNearestTiesToEven, &losesInfo);

      uint16_t raw = apf.bitcastToAPInt().getZExtValue();

      std::vector<unsigned char> ConstCal_hex;
      ConstCal_hex.push_back(static_cast<unsigned char>(raw & 0xFF));
      ConstCal_hex.push_back(static_cast<unsigned char>((raw >> 8) & 0xFF));

      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    if(intattr.getType().isInteger(16)){    
      int value = intattr.getInt();   
      int16_t value_16 = (int16_t) value;       
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_16);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
    else if(intattr.getType().isInteger(32)){
      int value = intattr.getInt();   
      int32_t value_32 = (int32_t) value;     
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_32);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
    else if(intattr.getType().isInteger(64)){ 
      int value = intattr.getInt();           
      int64_t value_64 = (int64_t) value;        
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_64);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(64);
    }
    else if(intattr.getType().isUnsignedInteger(16)){    
      unsigned int value = intattr.getUInt();          
      int16_t value_16 = (int16_t) value;       
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_16);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(16);
    }
    else if(intattr.getType().isUnsignedInteger(32)){
      unsigned int value = intattr.getUInt();   
      int32_t value_32 = (int32_t) value;     
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_32);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(32);
    }
    else if(intattr.getType().isUnsignedInteger(64)){ 
      unsigned int value = intattr.getUInt();                
      int64_t value_64 = (int64_t) value;        
      std::vector<unsigned char> ConstCal_hex = DataBitCastToHex(value_64);
      node->setConstValHex(ConstCal_hex);
      node->setDataBits(64);
    }
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    std::vector<unsigned char> ConstCal_hex;
    ConstCal_hex.push_back((char)value);
    node->setConstValHex(ConstCal_hex);
    node->setDataBits(1);
  }
}

uint32_t ConstantOpToHex(arith::ConstantOp constop){
  mlir::Attribute constattr = constop.getValueAttr();
  unsigned char *hex;
  if(isa<FloatAttr>(constattr)){
    FloatAttr floatattr = dyn_cast<FloatAttr>(constattr);
    double value = floatattr.getValueAsDouble();
    float value_float = (float)value;
    hex = (unsigned char*) (&value_float);
    return (uint32_t)*hex;
  } 
  else if(isa<IntegerAttr>(constattr))
  {
    IntegerAttr intattr = dyn_cast<IntegerAttr>(constattr);
    int value = intattr.getInt(); 
    uint32_t value_32 = (uint32_t) value;
    hex = (unsigned char*) (&value_32);  
    return (uint32_t)*hex;
  }
  else if(isa<BoolAttr>(constattr))
  {
    BoolAttr boolattr = dyn_cast<BoolAttr>(constattr);
    bool value = boolattr.getValue();
    uint32_t value_32 = (uint32_t) value;
    hex = (unsigned char*) (&value_32);  
    return (uint32_t)*hex; 
  }
}

//////////////////////////////
// Handle self-cycles in CDFG. Extract Acc operators and ISEL operators.
//////////////////////////////
static void HandleSelfCycle(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  SmallVector<LLVMCDFGNode*> YieldsToBeDelete;
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    if(node->getTypeName() == "yield"){
      /// Get the operand of yield op.
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(node->operation());
      if(yieldop.getOperands().size() != 0){
        // yieldop.dump();
        mlir::Operation* forop = node->operation()->getParentOp();
        // forop->dump();

        LLVMCDFGNode* fornode = CDFG->node(forop);
        // fornode->addOutputNode(node, /*isBackEdge=*/false);
        // node->addInputNode(fornode,  /*operand_idx=*/ 1, /*isBackEdge=*/false);

        fornode->addInputNode(node,  /*edgeidx=*/0, /*isBackEdge=*/true);
        node->addOutputNode(fornode, /*isBackEdge=*/true);
        CDFG->addEdge(node, fornode); //To fix: Edge Type
        // for(auto &elem : CDFG->edges()){
        //   auto edge = elem.second;
        //   auto srcName = edge->src()->getName();
        //   auto dstName = edge->dst()->getName();
        //   llvm::errs() << srcName << " -> " << dstName << "\n";
        // }
        // CDFG->delNode();
        YieldsToBeDelete.push_back(node);
      } 
      else{
        CDFG->delNode(node);
      }
    }
  }
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_1_CDFG.dot");}
  
  nodes = CDFG->nodes();
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    if(node->getTypeName() == "yield"){
      /// Get the operand of yield op.
      AffineYieldOp yieldop = dyn_cast<AffineYieldOp>(node->operation());
      AffineForOp forop = dyn_cast<AffineForOp>(yieldop.getOperation()->getParentOp());
      assert(yieldop.getOperands().size() == forop.getOperands().size());

      for(int OperandIdx = 0; OperandIdx < yieldop.getOperands().size(); OperandIdx++){
        LLVMCDFGNode* ComputeNode;
        // mlir::Value init_mlir_value;
        std::string accTypeName;
        uint32_t init_value;
        // if(isa<BlockArgument>(yieldop.getOperand(OperandIdx))){
        //   /// Create a ISEL node
        //   init_mlir_value = getInitialValueFromYieldIndex(yieldop, OperandIdx); 
        //   // ComputeNode = CDFG->addNode("ISEL");
        //   accTypeName = "ISEL";
        // }
        // else{
          mlir::Operation* ComputeOp = yieldop.getOperand(OperandIdx).getDefiningOp();
          if(verbose) llvm::errs() << "[debug]ComputeOp: ";
          if(verbose) ComputeOp->dump();
          ComputeNode = CDFG->node(ComputeOp);
          mlir::Value init_mlir_value = getInitialValueFromYieldIndex(yieldop, OperandIdx);   
          if(verbose) llvm::errs() << "[debug]init_mlir_value: " << init_mlir_value << "\n";
          
          /// set acc type
          if(ComputeNode->getTypeName() == "ADD" 
            && checkAccumulationChain<arith::AddIOp>(forop, OperandIdx)){
            /// ACC
            accTypeName = "ACC";
          }
          else if(ComputeNode->getTypeName() == "FADD"
            && checkAccumulationChain<arith::AddFOp>(forop, OperandIdx)){
            /// FACC32
            accTypeName = "FACC";
          }
          // else if(ComputeNode->getTypeName() == "MUL"
          //   && checkAccumulationChain<arith::MulIOp>(forop, OperandIdx)){
          //   /// MACC
          //   accTypeName = "MACC";
          // }
          // else if(ComputeNode->getTypeName() == "FMUL"
          //   && checkAccumulationChain<arith::MulFOp>(forop, OperandIdx)){
          //  /// FMACC32
          //   accTypeName = "FMACC";
          // }
          else if(ComputeNode->getTypeName() == "SEL"
            && checkAccumulationChain<arith::SelectOp>(forop, OperandIdx)){
            /// SEL can be extracted as accumulation mode as well.
            accTypeName = "ISEL";
          }
          else if(!checkAccumulationChain<ADORA::IselOp>(forop, OperandIdx)
          // && ComputeNode->getTypeName() == "ISEL"
          ){
            accTypeName = "ISEL";
            //// connect init node ------> ISEL <- - - - - - loop carry node 
            ///////// Get ISEL node
            SmallVector<mlir::Operation*> uses = getAllUsesInBlock(forop.getRegionIterArgs()[OperandIdx], forop.getBody());
            assert(uses.size()==1);
            if(!isa<ADORA::IselOp>(uses[0])){
              continue;
            }
            // assert(isa<ADORA::IselOp>(uses[0]));
            ADORA::IselOp iselop = dyn_cast<ADORA::IselOp>(uses[0]);
            auto IselNode = CDFG->node(iselop.getOperation());

            ///////// Connect init node 
            // auto InitNode = CDFG->node(init_mlir_value.getDefiningOp());
            // bool isBackEdge = false;
            // InitNode->addOutputNode(IselNode, isBackEdge);
            // IselNode->addInputNode(InitNode, /*edgeidx=*/1, isBackEdge);
            // CDFG->addEdge(InitNode, IselNode); //To fix: Edge Type

            ///////// connect loop carry node 
            bool isBackEdge = true;
            ComputeNode->addOutputNode(IselNode, isBackEdge);
            IselNode->addInputNode(ComputeNode, /*edgeidx=*/0, isBackEdge);
            LLVMCDFGEdge* backedge = CDFG->addEdge(ComputeNode, IselNode); //To fix: Edge Type
            backedge->setIterDist(1);

            // DependInfo DI;
            // DI.type = 
            // continue;
            ComputeNode = IselNode;
          }
          else{
            continue;
          }
        // }

        //// init value
        if(isa<arith::ConstantOp>(init_mlir_value.getDefiningOp())){
          arith::ConstantOp constop = dyn_cast<arith::ConstantOp>(init_mlir_value.getDefiningOp());
          init_value = ConstantOpToHex(constop);
        }
        else if(isa<affine::AffineLoadOp>(init_mlir_value.getDefiningOp())){
          init_value = 0x00000000;
        } else{
          /// TODO: What to do if it is not constant op
          assert(0);
        }

        SmallVector<std::string, 3> count_interval_repeat = {"1", "1", "1"};///count/interval/repeat
        count_interval_repeat = GetACCInfoFromYieldNode(node, count_interval_repeat);
        ComputeNode->setTypeName(accTypeName);
        ComputeNode->setAcc();
        ComputeNode->setACCinit(std::to_string(init_value));
        ComputeNode->setACCcount(count_interval_repeat[0]);
        ComputeNode->setACCinterval(count_interval_repeat[1]);
        ComputeNode->setACCrepeat(count_interval_repeat[2]);    

        //// Change operand idx of acc op
        SetACCOperandIdx(ComputeNode);
      }
    }
  }
  
  /// Thirdly, delete yield-for nodes
  // unsigned k = 55;
  for(auto ynode : YieldsToBeDelete){
    DeleteYield(CDFG, ynode);
  }
}


static bool HandleCompareNode(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  for(auto &elem : nodes){
    // int node_id = elem.first;
    LLVMCDFGNode* node = elem.second;
    mlir::Operation* op = node->operation();
    // if(verbose) {op->dump();}
    if(   op->getName().getStringRef() == "arith.cmpi"
        ||op->getName().getStringRef() == "arith.cmpf"){
      node->setTypeName(GetCMPTypeStr(op));
      if(!ConvertGreaterToLess(node)){
        return false;
      }
    }
  }
  return true;
}


/// @brief Handles vector extract nodes in the CDFG.
/// 
/// This function iterates through all nodes in the CDFG and looks for operations 
/// of type "vector.extract". If such an operation is found, it checks if its 
/// inputs are nodes of type "INTLV". If they are, it connects the output nodes 
/// to the input nodes and deletes the extract node from the CDFG.
/// 
/// @param CDFG A pointer to the LLVMCDFG representing the current control data flow graph.
/// @param verbose A boolean indicating whether to print detailed information (default is true).
void HandleVectorExtractNode(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  for(auto &elem : nodes){
    LLVMCDFGNode* node = elem.second;
    mlir::Operation* op = node->operation();
    if(op->getName().getStringRef() == "vector.extract"){
      mlir::vector::ExtractOp extractop = dyn_cast<mlir::vector::ExtractOp>(op);
      mlir::Operation* vecop = extractop.getVector().getDefiningOp();

      if(isa<ADORA::InterleaverOp>(vecop)){
        auto outnodes = node->outputNodes();
        auto innodes = node->inputNodes();
        for(LLVMCDFGNode* innode : innodes){
          if(innode->getTypeName().substr(0,5) == "INTLV"){
            for(LLVMCDFGNode* outnode : outnodes){
              std::vector<NodeInfo> infos = outnode->getinputInfoMap()[innode];
              for(NodeInfo info: infos){
                int edgeidx = info.idx;
                outnode->addInputNode(innode, edgeidx, /*isBackEdge=*/false);
                innode->addOutputNode(outnode, /*isBackEdge=*/false);
                CDFG->addEdge(innode, outnode); //To fix: Edge Type  
              }
              // int edgeidx = node->getInputIdx(innode);
              // outnode->addInputNode(innode, edgeidx, /*isBackEdge=*/false);
              // innode->addOutputNode(outnode, /*isBackEdge=*/false);
              // CDFG->addEdge(innode, outnode);   
            }
          }
          else if(innode->getTypeName() == "for"){
            continue;
          }
          else {
            assert(false && "Vector extract op could only support input as Interleaver op.");
          }
        }
        CDFG->delNode(node);  
      }
      else{
        assert(false && "Vector extract op could only support input as interleaver op.");
      }
    }
  }
  return;
}


/// @brief Fixes the linear access pattern of vector store nodes in the CDFG.
/// Based on the number of interleaverleaver inputs, it fixes the linear access pattern 
/// associated with the interleaver operation and updates the linear access string of that node.
/// 
/// @param CDFG A pointer to the LLVMCDFG representing the current control data flow graph.
/// @param verbose A boolean indicating whether to print detailed information (default is true).
void FixLinearAccessOfVectorNode(LLVMCDFG* CDFG, bool verbose = true){
  auto nodes = CDFG->nodes();
  for(auto &elem : nodes){
    LLVMCDFGNode* node = elem.second;
    mlir::Operation* op = node->operation();
    if(op->getName().getStringRef() == "affine.vector_store"){
      mlir::affine::AffineVectorStoreOp vecstoreop = dyn_cast<mlir::affine::AffineVectorStoreOp>(op);
      mlir::Operation* vecop = vecstoreop.getValue().getDefiningOp();
      int ElementBytes = vecstoreop.getMemRefType().getElementTypeBitWidth()/8;

      if(isa<ADORA::InterleaverOp>(vecop) || 
        isa<arith::AddIOp>(vecop) || isa<arith::AddFOp>(vecop)){
        int vecnum = 1;
        ArrayRef<int64_t> shape = dyn_cast<mlir::VectorType>(vecop->getResult(0).getType()).getShape();;
        int dimLargerThanOne = -1;
        for(int _ = 0; _ < shape.size(); _++){
          int _dim = shape[_];
          vecnum *= _dim;
          if(_dim > 1) {
            assert(dimLargerThanOne == -1); /// ensure only one dim is larger than 1
            dimLargerThanOne = _;
          }
        }
        assert(dimLargerThanOne != -1);
        // if(isa<ADORA::InterleaverOp>(vecop)){
        //   /// get the input num of interleaver
        //   int interleaverNum = dyn_cast<ADORA::InterleaverOp>(vecop).getInterleaveNumber();
        //   vecnum = interleaverNum;
        // }
        // else if(isa<arith::AddIOp>(vecop) || isa<arith::AddFOp>(vecop)){
        //   shape = dyn_cast<mlir::VectorType>(vecop->getResult(0).getType()).getShape();
        //   vecnum = 1;
        //   for(auto _dim : shape){
        //     vecnum *= _dim;
        //   }
        // }
        // else{
        //   assert(false && "Unsupported input of vector_store.");
        // }

        /// fix linear access of extractop
        assert(node->isLSaffine() && node->getTypeName() == "Output");
        std::string linearAccess = node->getLinearAccess();

        std::stringstream ss(node->getLinearAccess());
        std::string step, count;

        SmallVector<std::pair<int64_t, int64_t>> newLinearAccess;
        ArrayRef<int64_t> memRefShape =  vecstoreop.getMemRefType().getShape();
        int innermostStep = ElementBytes;
        for(int dim = memRefShape.size() - 1; dim > dimLargerThanOne; dim--){
          innermostStep *= memRefShape[dim];
        }        
        newLinearAccess.push_back(std::pair(innermostStep, vecnum));
        // newLinearAccess.push_back(std::pair( -1 * ElementBytes * interleaverNum, 1));

        int level = 0;
        while (std::getline(ss, step, ',')) {
          std::getline(ss, count, ',');
          if(level == 0){
            assert(std::stoi(step) % innermostStep == 0);
            assert(std::stoi(count) % vecnum == 0);
            int newstep = std::stoi(step) / innermostStep - newLinearAccess[0].first * (newLinearAccess[0].second - 1);
            int newcount = std::stoi(count) / vecnum;
            newLinearAccess.push_back(std::pair(newstep, newcount));   
          }
          else if(level == 1){
            int newstep = std::stoi(step) 
                          - newLinearAccess[1].first * (newLinearAccess[1].second - 1)
                          - newLinearAccess[0].first * (newLinearAccess[0].second - 1) * (newLinearAccess[1].second);
            int newcount = std::stoi(count);
            newLinearAccess.push_back(std::pair(newstep, newcount));   
          }
          else {
            newLinearAccess.push_back(std::pair(std::stoi(step), std::stoi(count)));
          }
          // if(level == 1){
          //   int newstep = std::stoi(step) - ElementBytes * vecnum + ElementBytes;
          //   newLinearAccess.push_back(std::pair(newstep, std::stoi(count)));            
          // }
          // else if(level != 0) {
          //   newLinearAccess.push_back(std::pair(std::stoi(step), std::stoi(count)));
          // }

          level++;
        }

        assert (!newLinearAccess.empty());
        
        node->setLinearAccess(LinearAccessToStr(newLinearAccess));
      }
      else{
        assert(false && "vectorstore op could only support input as interleaver op right now.");
      }
    }
    else if(op->getName().getStringRef() == "affine.vector_load"){
      mlir::affine::AffineVectorLoadOp vecloadop = dyn_cast<mlir::affine::AffineVectorLoadOp>(op);
      // for(LLVMCDFGNode* outputnode : node->outputNodes()){
      Operation* userop = node->outputNodes()[0]->operation();
      // }
      int ElementBytes = vecloadop.getMemRefType().getElementTypeBitWidth()/8;

      if(isa<ADORA::DeinterleaverOp>(userop) || 
        isa<arith::AddIOp>(userop) || isa<arith::AddFOp>(userop)){
        int vecnum = 1;
        ArrayRef<int64_t> shape = dyn_cast<mlir::VectorType>(op->getResult(0).getType()).getShape();
        // if(isa<ADORA::DeinterleaverOp>(userop)){
        //   /// get the input num of interleaver
        //   int vecnum = dyn_cast<ADORA::DeinterleaverOp>(userop).getDeinterleaveNumber();
        // }
        // else if(isa<arith::AddIOp>(userop) || isa<arith::AddFOp>(userop)){
        int dimLargerThanOne = -1;
        for(int _ = 0; _ < shape.size(); _++){
          int _dim = shape[_];
          vecnum *= _dim;
          if(_dim > 1) {
            assert(dimLargerThanOne == -1); /// ensure only one dim is larger than 1
            dimLargerThanOne = _;
          }
        }
        assert(dimLargerThanOne != -1);
        // }
        // else{
        //   assert(false && "Unsupported input of vector_store.");
        // }

        /// fix linear access of extractop
        assert(node->isLSaffine() && node->getTypeName() == "Input");
        std::string linearAccess = node->getLinearAccess();

        std::stringstream ss(node->getLinearAccess());
        std::string step, count;

        SmallVector<std::pair<int64_t, int64_t>> newLinearAccess;
        ArrayRef<int64_t> memRefShape =  vecloadop.getMemRefType().getShape();
        int innermostStep = ElementBytes;
        for(int dim = memRefShape.size() - 1; dim > dimLargerThanOne; dim--){
          innermostStep *= memRefShape[dim];
        }
        newLinearAccess.push_back(std::pair(innermostStep, vecnum));
        // newLinearAccess.push_back(std::pair( -1 * ElementBytes * interleaverNum, 1));

        int level = 0;
        while (std::getline(ss, step, ',')) {
          std::getline(ss, count, ',');
          if(level == 0){
            assert(std::stoi(step) % innermostStep == 0);
            assert(std::stoi(count) % vecnum == 0);
            int newstep = std::stoi(step) / innermostStep - newLinearAccess[0].first * (newLinearAccess[0].second - 1);
            int newcount = std::stoi(count) / vecnum;
            newLinearAccess.push_back(std::pair(newstep, newcount));   
          }
          else if(level == 1){
            int newstep = std::stoi(step) 
                          - newLinearAccess[1].first * (newLinearAccess[1].second - 1)
                          - newLinearAccess[0].first * (newLinearAccess[0].second - 1) * (newLinearAccess[1].second);
            int newcount = std::stoi(count);
            newLinearAccess.push_back(std::pair(newstep, newcount));   
          }
          else {
            newLinearAccess.push_back(std::pair(std::stoi(step), std::stoi(count)));
          }

          level++;
        }

        assert (!newLinearAccess.empty());
        
        node->setLinearAccess(LinearAccessToStr(newLinearAccess));
      }
      else{
        assert(false && "vector_load op could only support input as interleaver op right now.");
      }
    }
  }
  return;
}


static void fuseMulAccToMAC(LLVMCDFG* CDFG, const std::string& mul_name, const std::string& acc_name, const std::string& mac_name){
  auto nodes = CDFG->nodes();
  for(auto &elem : nodes){
    LLVMCDFGNode* accnode = elem.second;
    if(accnode->getTypeName() == acc_name){
      assert(accnode->inputNodes().size() == 1);
      LLVMCDFGNode* mulnode = accnode->inputNodes()[0];
      if(mulnode->getTypeName() == mul_name){
        // fusable
        accnode->setTypeName(mac_name);

        LLVMCDFGNode* mulLHS = mulnode->getInputPort(0);
        LLVMCDFGNode* mulRHS = mulnode->getInputPort(1);

        accnode->addInputNode(mulLHS, 0, /*isBackEdge=*/false);
        mulLHS->addOutputNode(accnode, /*isBackEdge=*/false);
        CDFG->addEdge(mulLHS, accnode);   
        accnode->addInputNode(mulRHS, 1, /*isBackEdge=*/false);
        mulRHS->addOutputNode(accnode, /*isBackEdge=*/false);
        CDFG->addEdge(mulRHS, accnode);   

        CDFG->delNode(mulnode);
      }
    }

  }
}

////////////////////////
/// Handle floatpoint bitwidth
////////////////////////
static void SpecifyFPNodePrecision(LLVMCDFG* CDFG, bool verbose){
  auto nodes = CDFG->nodes();
  for (auto &elem : nodes) {
    LLVMCDFGNode* node = elem.second;
    mlir::Operation* op = node->operation();
    if (!op) continue;
    if (llvm::isa<mlir::arith::AddFOp>(op) ||
        llvm::isa<mlir::arith::SubFOp>(op) ||
        llvm::isa<mlir::arith::MulFOp>(op) ||
        llvm::isa<mlir::arith::DivFOp>(op) ||
        llvm::isa<mlir::arith::CmpFOp>(op)) 
    {
      mlir::Type resultTy;
      if (llvm::isa<mlir::arith::CmpFOp>(op))
        resultTy = op->getOperand(0).getType();
      else if (op->getNumResults() > 0)
        resultTy = op->getResult(0).getType();
      else
        continue;
      
      mlir::FloatType floatTy;
      if (resultTy.isa<mlir::VectorType>()
        && dyn_cast<mlir::VectorType>(resultTy).getElementType().isa<mlir::FloatType>()){
        floatTy = dyn_cast<mlir::VectorType>(resultTy).getElementType().cast<mlir::FloatType>();
      }
      else if (resultTy.isa<mlir::FloatType>()){
        floatTy = resultTy.cast<mlir::FloatType>();
      }
      else {        
        continue;
      }

      unsigned width = floatTy.getWidth();
      std::string precision;

      std::string oldName, newName;
      oldName = node->getTypeName();

      if (floatTy.isF16()){
        precision = "16";
        newName = oldName + precision;
      }
      else if (floatTy.isBF16()){
        precision = "16";
        newName = "B" + oldName + precision;
      }
      else if (floatTy.isF32()){
        precision = "32";
        newName = oldName + precision;
      }
      else if (floatTy.isF64()){
        precision = "64";
        newName = oldName + precision;
      }
      
      node->setTypeName(newName);
    }
  }
}


/// @brief fuse operators: MAC(int), FMAC32
/// 
/// @param CDFG A pointer to the LLVMCDFG representing the current control data flow graph.
/// @param verbose A boolean indicating whether to print detailed information (default is true).
static void FuseOperators(LLVMCDFG* CDFG, bool verbose){
  if(CDFG->getFusableOperatorTypes().size() == 0)
    return;
  
  if(CDFG->getFusableOperatorTypes().count("MAC") != 0){
    fuseMulAccToMAC(CDFG, /*mul_name*/"MUL", /*acc_name*/"ACC", /*mac_name*/"MAC");
  }
 
  if(CDFG->getFusableOperatorTypes().count("FMAC32") != 0){
    fuseMulAccToMAC(CDFG, /*mul_name*/"FMUL32", /*acc_name*/"FACC32", /*mac_name*/"FMAC32");
  }
}

bool generateCDFGfromKernelAfterOptimization(LLVMCDFG* CDFG, ADORA::KernelOp kernel, bool verbose){
  if(verbose) {kernel.dump();}
  _kernel_toDFG = &kernel;

  int level = 0, level_total;
  std::map<mlir::Operation*, int> For_loop_level;
  std::map<mlir::Block*, int> loop_block_level;
  SmallVector<mlir::Operation*> OpsOutsideFor;
  kernel->walk([&](mlir::Operation* op)
  {
    if(isa<affine::AffineForOp>(op)){
      int Innermost = 1;
      op->walk([&](affine::AffineForOp temp_forop)
      { 
        temp_forop.dump();
        if(For_loop_level.count(temp_forop) == 0 && temp_forop != dyn_cast<affine::AffineForOp>(op)){ // Don't count the scf::For itself
          Innermost = 0;
          llvm::errs() << "Not innermost"  << "\n";    
        } 
      });
      if (Innermost)
      {
        For_loop_level[op] = level;
        level++;
      }
      if(isa<ADORA::KernelOp>(op->getParentOp()))
        OpsOutsideFor.push_back(op);
      // for_region.viewGraph();
    }
    // scf::ForOp forop;
    else if(isa<scf::ForOp>(op)){
      int Innermost = 1;
      op->walk([&](scf::ForOp temp_forop)
      { 
        if(For_loop_level.count(temp_forop) == 0 && temp_forop != dyn_cast<scf::ForOp>(op)){ // Don't count the scf::For itself
          Innermost = 0;
        // llvm::errs() << "Not innermost"  <<std::endl;    
        } 
      });
      if (Innermost)
      {
        For_loop_level[op] = level;
        level++;
      }
      // for_region.viewGraph();
    }
    else if((isa<affine::AffineLoadOp>(op) 
        || isa<affine::AffineStoreOp>(op)
        || isa<arith::AddFOp>(op)
        || isa<arith::AddIOp>(op)
        || isa<arith::SubFOp>(op)
        || isa<arith::SubIOp>(op)) 
        && isa<ADORA::KernelOp>(op->getParentOp())){
      OpsOutsideFor.push_back(op);
    }
  });
  level_total = level;
  // scf::ForOp scf_for;
  mlir::Operation* for_op;


  /*** Add Nodes ***/
  mlir::SmallVector<mlir::Operation*> AddedOps;
  for (level = 0; level < level_total; level++)
  {
    /// Find the scf_for body in current level
    for (auto op : For_loop_level)
    {
      if (op.second == level){
        for_op = op.first;
      }
    }
    /// this is a affine for
    if(isa<affine::AffineForOp>(for_op)){
      affine::AffineForOp affinefor = dyn_cast<affine::AffineForOp>(for_op);
      if(verbose) {errs() << "level:" << level << "\n"; affinefor.dump();}

      // mlir::Region *for_region = affinefor.getBody()->getParent();
      /// index and iter_args in scf_for
      // int block_cnt = 0;
      // for (auto itr=for_region->begin(); itr!=for_region->end(); itr++, block_cnt++)
      // {
      //   mlir::Block* for_block = &(*itr);
      //   errs() << "Block:" << for_block <<"\n";
      //   for_block->dump();
      //   // loop_block_level[for_block] = level;
      //   // CDFG->addNode(for_block, level); 
      // }
      // affinefor.walk([&](mlir::Operation *op){
      //   errs() << "Node:"; op->dump();
      // });
      // assert(block_cnt == 1 && "Region in affine.for should only contain 1 block !");
      affinefor.walk([&](mlir::Operation *op)
      { 
        if(verbose) {errs() << "Node:"; op->dump();}
        if(findElement(AddedOps, op) != -1)
          return WalkResult::advance();
        else
          AddedOps.push_back(op);

        if(op->getName().getStringRef() == "affine.for"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // node->setisSCFForOp(true);
          // return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "affine.yield"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // TODO: settle this
          // return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "affine.apply"){
          // LLVMCDFGNode* node = CDFG->addNode(op); 
          // node->setLoopLevel(level);
          // // TODO: settle this
          return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "ADORA.interleaver"){
          ADORA::InterleaverOp interleaverop = dyn_cast<ADORA::InterleaverOp>(op);
          int interleavernum = interleaverop.getInterleaveNumber();
          std::string interleavertypename = "INTLV" + std::to_string(interleavernum);
          LLVMCDFGNode* node = CDFG->addNode(op, /*typeName=*/interleavertypename); 
          node->setLoopLevel(level);

          //// set acc for interleaver op
          SmallVector<std::string, 3> count_interval_repeat = {"1", "1", "1"};///count/interval/repeat
          node->setAcc();
          node->setACCinit("0");
          node->setACCcount(count_interval_repeat[0]);
          node->setACCinterval(count_interval_repeat[1]);
          node->setACCrepeat(count_interval_repeat[2]);    
          // // TODO: settle this
          return WalkResult::advance();
        } 
        else if(op->getName().getStringRef() == "ADORA.deinterleaver"){
          ADORA::DeinterleaverOp deinterleaverop = dyn_cast<ADORA::DeinterleaverOp>(op);
          int deinterleavernum = deinterleaverop.getDeinterleaveNumber();
          std::string deinterleavertypename = "DEINTLV" + std::to_string(deinterleavernum);
          LLVMCDFGNode* node = CDFG->addNode(op, /*typeName=*/deinterleavertypename); 
          node->setLoopLevel(level);

          //// set acc for deinterleaver op
          SmallVector<std::string, 3> count_interval_repeat = {"1", "1", "1"};///count/interval/repeat
          node->setAcc();
          node->setACCinit("0");
          node->setACCcount(count_interval_repeat[0]);
          node->setACCinterval(count_interval_repeat[1]);
          node->setACCrepeat(count_interval_repeat[2]);    
          // // TODO: settle this
          return WalkResult::advance();
        } 
        else if (op->getName().getStringRef() == "affine.vector_store" ){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineVectorStoreOp vecstore = dyn_cast<affine::AffineVectorStoreOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(vecstore, For_loop_level));
          // std::string initAddr_str = std::to_string(GetInitAddr(vecstore, For_loop_level));
          std::string initAddr_str = "0";
          int memrefsize = GetMemrefSize(vecstore);
          mlir::Operation* mrefop = vecstore.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
          }
          else{
            assert(0);
          }
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);

          if(op->hasAttr("Pingpong")) /// pingpong
            node->setPingpong(true);
        }
        else if (op->getName().getStringRef() == "affine.vector_load" ){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineVectorLoadOp vecload = dyn_cast<affine::AffineVectorLoadOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(vecload, For_loop_level));
          // std::string initAddr_str = std::to_string(GetInitAddr(vecstore, For_loop_level));
          std::string initAddr_str = "0";
          int memrefsize = GetMemrefSize(vecload);
          mlir::Operation* mrefop = vecload.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
          }
          else{
            assert(0);
          }
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);

          if(op->hasAttr("Pingpong")) /// pingpong
            node->setPingpong(true);
        }
        else if (op->getName().getStringRef() == "affine.load"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineLoadOp load = dyn_cast<affine::AffineLoadOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(load, For_loop_level));
          std::string initAddr_str = std::to_string(GetInitAddr(load, For_loop_level));

          mlir::Operation* mrefop = load.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
          }
          else
            assert(0);
          // mrefop->dump();
          // llvm::errs() << mrefop->getName();
          int memrefsize = GetMemrefSize(load);
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);

          if(op->hasAttr("Pingpong")) /// pingpong
            node->setPingpong(true);
        }
        else if (op->getName().getStringRef() == "affine.store" ){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          affine::AffineStoreOp store = dyn_cast<affine::AffineStoreOp>(op);
          std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(store, For_loop_level));
          std::string initAddr_str = std::to_string(GetInitAddr(store, For_loop_level));
          int memrefsize = GetMemrefSize(store);
          mlir::Operation* mrefop = store.getMemref().getDefiningOp();
          std::string ref_name;
          if(isa<ADORA::DataBlockLoadOp>(mrefop)){
            ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
          }
          else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
            ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
            ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
          }
          else{
            assert(0);
          }
          // op->getResult(0).addAttribute("LinearAccess", b.getStringAttr(linearaccess_str));
          node->setLinearAccess(linearaccess_str);
          node->setInitAddr(initAddr_str);
          node->setMemrefSize(memrefsize);
          node->setMemrefName(ref_name);
          node->setLSaffine(true);

          if(op->hasAttr("Pingpong")) /// pingpong
            node->setPingpong(true);
        }
        else if(op->getName().getStringRef() == "arith.constant"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          setConstantNode(node);
        }
        else if(op->getName().getStringRef() == "arith.cmpi"
              ||op->getName().getStringRef() == "arith.cmpf"){
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
          // node->setTypeName(GetCMPTypeStr(op));
          // ConvertGreaterToLess(node);
        }
        else{
          LLVMCDFGNode* node = CDFG->addNode(op); 
          node->setLoopLevel(level);
        }

        return WalkResult::advance();
      });
    }
  }
  /// Add those nodes outside for loop
  for(mlir::Operation* lsop : OpsOutsideFor){
    LLVMCDFGNode* node = CDFG->addNode(lsop); 
    node->setLoopLevel(level_total);
    if (lsop->getName().getStringRef() == "affine.load"){
      affine::AffineLoadOp load = dyn_cast<affine::AffineLoadOp>(lsop);
      std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(load, For_loop_level));
      std::string initAddr_str = std::to_string(GetInitAddr(load, For_loop_level));

      mlir::Operation* mrefop = load.getMemref().getDefiningOp();
      std::string ref_name;
      if(isa<ADORA::DataBlockLoadOp>(mrefop)){
        ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
        ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
      }
      else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
        ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
        ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
      }
      else
        assert(0);

      int memrefsize = GetMemrefSize(load);
      node->setLinearAccess(linearaccess_str);
      node->setInitAddr(initAddr_str);
      node->setMemrefSize(memrefsize);
      node->setMemrefName(ref_name);
      node->setLSaffine(true);
    }
    else if (lsop->getName().getStringRef() == "affine.store" ){
      affine::AffineStoreOp store = dyn_cast<affine::AffineStoreOp>(lsop);
      std::string linearaccess_str = LinearAccessToStr(GetLinearAccess(store, For_loop_level));
      std::string initAddr_str = std::to_string(GetInitAddr(store, For_loop_level));
      int memrefsize = GetMemrefSize(store);
      mlir::Operation* mrefop = store.getMemref().getDefiningOp();
      std::string ref_name;
      if(isa<ADORA::DataBlockLoadOp>(mrefop)){
        ADORA::DataBlockLoadOp Bload = dyn_cast<ADORA::DataBlockLoadOp>(mrefop);
        ref_name = std::string(kernel.getKernelName()) + ":" + std::string(Bload.getId());
      }
      else if(isa<ADORA::LocalMemAllocOp>(mrefop)){
        ADORA::LocalMemAllocOp BAlloc = dyn_cast<ADORA::LocalMemAllocOp>(mrefop);
        ref_name = std::string(kernel.getKernelName()) + ":" + std::string(BAlloc.getId());
      }
      else{
        assert(0);
      }
      node->setLinearAccess(linearaccess_str);
      node->setInitAddr(initAddr_str);
      node->setMemrefSize(memrefsize);
      node->setMemrefName(ref_name);
      node->setLSaffine(true);
    }
  }

  /*** Add Edges ***/
  for (auto nodepair : CDFG->nodes())
  {
    // int id = nodepair.first;
    LLVMCDFGNode *SuccNode = nodepair.second;

    /// Skip Loop index and arg node because it don't get operands 
    if(SuccNode->getTypeName() == "Loop index" 
            || SuccNode->getTypeName() == "Loop arg")
    { 
      if(verbose) { errs() << nodepair.first << ".Node:" << SuccNode->getTypeName() <<"\n";}
      continue;
    }

    mlir::Operation *op = SuccNode->operation();
    if(verbose) {errs() << nodepair.first << ".Node:";}
    if(verbose) {op->dump();}
    for (unsigned operand_idx = 0; operand_idx < op->getNumOperands(); operand_idx++)
    {
      LLVMCDFGNode *AnceNode = NULL;
      bool isBackEdge = false;
      mlir::Value _v = op->getOperand(operand_idx);

      int edgeidx;
      if (SuccNode->getTypeName() == "SEL")
        edgeidx = 2 - operand_idx;
      else if (SuccNode->getTypeName() == "load")
        // memref.load: MLIR operand 0 = memref, 1+ = address indices; CDFG address = first operand (0)
        edgeidx = (operand_idx >= 1) ? (int)(operand_idx - 1) : (int)operand_idx;
      else
        edgeidx = operand_idx;
      
      if (_v.isa<BlockArgument>()) {
        /// Operands is a loop index or loop arg
        mlir::BlockArgument arg = _v.cast<BlockArgument>();
        mlir::Block * owner = arg.getOwner();
        int blk_level = loop_block_level[owner];
        
        // _v.dump();
        // _v.getType().dump();
        // owner->dump();
        mlir::Operation* parentop;

        if(_v.getType().isIndex()){
          isBackEdge = true;
          parentop = _v.getParentBlock()->getParentOp();
          if(isa<affine::AffineForOp>(parentop)){
            if(verbose) {errs() << "  parentop:" << *parentop <<"\n";}
            
            if(SuccNode->isLinearAccess()){
              continue;      
            }       
            else{
              AnceNode = CDFG->node(parentop);
              AnceNode->addOutputNode(SuccNode, isBackEdge);
              SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
              CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
            }
          }
        }
        else{
          isBackEdge = true;
          parentop = _v.getParentBlock()->getParentOp();
          if(isa<affine::AffineForOp>(parentop)){
            AnceNode = CDFG->node(parentop);
            AnceNode->addOutputNode(SuccNode, isBackEdge);
            SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
            CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type            
          }
        }
        continue;

        // parentop = _v.getParentBlock()->getParentOp();
        // if(isa<func::FuncOp>(parentop))
        //   continue;
        
        // switch (arg.getArgNumber())
        // {
        // case 0: /// loop index
        //   AnceNode = CDFG->node_lpidx(blk_level);
        //   if(verbose) { errs() << "   Loop index,level:" << blk_level << ", " <<_v << ", owner:" << owner <<"\n"; }
        //   break;
        // case 1: /// loop-carried iteration args
        //   // AnceNode = CDFG->node_lparg(blk_level);
        //   // errs() << "  forop:" << *forop <<"\n";
        //   isBackEdge = true;
        //   parentop = _v.getParentBlock()->getParentOp();

        //   /// if parentop is not for op, just skip
        //   if(!isa<affine::AffineForOp>(parentop))
        //     continue;

        //   if(verbose) { errs() << "  parentop:" << *parentop <<"\n";}
        //   AnceNode = CDFG->node(parentop);
        //   AnceNode->addOutputNode(SuccNode, isBackEdge);
        //   SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
        //   CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
        //   if(verbose) { errs() << "   Loop arg,level:" << blk_level << ", " <<_v << ", owner:" << owner <<"\n";}

        //   break;
        // default:
        //   assert( 0 && "The node of BlockArgument has not been stored into DFG !");
        //   break;
        // }
        // continue;
      }

      else{
        mlir::Operation *ance_op = op->getOperand(operand_idx).getDefiningOp();
        if(verbose) {errs() << "   Operands:";ance_op->dump();}

        AnceNode = CDFG->node(ance_op);
        if(AnceNode == NULL){ /// AnceNode is outside loop
          if(ance_op->getName().getStringRef() == "arith.constant"){
            AnceNode = CDFG->addNode(ance_op);
            setConstantNode(AnceNode);
          }
          else if(ance_op->getName().getStringRef() == "memref.get_global"){
            /// Tofix
            // AnceNode = CDFG->addNode(ance_op);
            continue;
          }
          else if(ance_op->getName().getStringRef() == "affine.apply"){
            continue;
          }
          else if(ance_op->getName().getStringRef() == "ADORA.BlockLoad"
                ||ance_op->getName().getStringRef() == "ADORA.LocalMemAlloc"){
            if(SuccNode->isLinearAccess())
              continue;
            // hjy：added for memref.load/store outside affine.for
            else if(SuccNode->operation()->getName().getStringRef() == "memref.load"
                ||SuccNode->operation()->getName().getStringRef() == "memref.store")
              continue;
            else 
              assert(0); /// Todo: fix this jhlou
          }
          //hjy
          else if(ance_op->getName().getStringRef() == "memref.alloca"){
            
            continue;
          }
          else {
            
            LLVM_DEBUG(llvm::errs() << "[Warning] Skipping unhandled external operation: " 
                       << ance_op->getName().getStringRef() << "\n");
            continue;
          }
          //   /// Extract Accumulation Operations
          //   // assert(isa<Affine::YieldOp>(op));
          //   AffineForOp ancefor = dyn_cast<AffineForOp>(ance_op);
          //   ValueRange YieldOperands =
          //       dyn_cast<AffineYieldOp>(ancefor.getBody()->getTerminator()).getOperands();
          //   assert(YieldOperands.size() == 1);
          //   for(mlir::Value Operand : YieldOperands){
          //     llvm::errs() << Operand <<"\n";
          //     mlir::Operation* OperandOp = Operand.getDefiningOp();
          //     if(   OperandOp->getName().getStringRef() == "arith.add" 
          //         || OperandOp->getName().getStringRef() == "arith.addf" 
          //         || OperandOp->getName().getStringRef() == "arith.mul"
          //         || OperandOp->getName().getStringRef() == "arith.mulf")
          //     {
          //       LLVMCDFGNode *ACCNode = CDFG->node(OperandOp);
          //       /// Set this node to be acc
          //       AnceNode = ACCNode;
          //       // AnceNode->SetBondingAccNode()
          //     }
          //     else if( OperandOp->getName().getStringRef() == "affine.for" ){
          //       /// operand is forop
          //       AffineYieldOp OldestAnceYield = getOldestAncestorYieldOp(dyn_cast<AffineForOp>(OperandOp));
          //       llvm::errs() << OldestAnceYield <<"\n";
          //       ValueRange OldestYieldOperands =
          //           dyn_cast<AffineYieldOp>(dyn_cast<AffineForOp>(OperandOp).getBody()->getTerminator()).getOperands();
          //       assert(OldestYieldOperands.size() == 1);
          //       mlir::Operation* OldestOperandOp = OldestYieldOperands[0].getDefiningOp();
          //       llvm::errs() << *OldestOperandOp <<"\n";
          //       LLVMCDFGNode *ACCNode = CDFG->node(OldestOperandOp);
          //     }
          //     else{
          //       assert(0 && "Unsupported Accumulation Type.(Support: ACC/FACC/MULACC/FMULACC)");
          //     }
          //   }
          // }
        }
      }

      AnceNode->addOutputNode(SuccNode, isBackEdge);
      SuccNode->addInputNode(AnceNode, edgeidx, isBackEdge);
      CDFG->addEdge(AnceNode, SuccNode); //To fix: Edge Type
    }
  }
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_0_CDFG.dot");}

  ////////////////////////
  /// Extract Accumulation
  ////////////////////////
  HandleSelfCycle(CDFG, verbose);
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_2_CDFG.dot");}
  ////////////////////////
  /// End of extracting acc op
  ////////////////////////


  ////////////////////////
  /// Handle compare node
  ////////////////////////
  bool result = HandleCompareNode(CDFG, verbose);
  if(!result) return false;

  ////////////////////////
  /// Handle vector_extract node && Maybe useless
  ////////////////////////
  HandleVectorExtractNode(CDFG, verbose);

  ////////////////////////
  /// Handle affine.vector_store node
  ////////////////////////
  FixLinearAccessOfVectorNode(CDFG, verbose);

  ////////////////////////
  /// Remove redundant nodes: bitcast, for with no source and sink, truncf
  ////////////////////////
  bool removing = true;
  while(removing){
    removing = false;
    auto nodes = CDFG->nodes();
    for(auto &elem : nodes){
      // int node_id = elem.first;
      LLVMCDFGNode* node = elem.second;
      if(node->getTypeName() == "bitcast" || node->getTypeName() == "index_cast" ){
        assert(node->inputNodes().size() == 1);
        LLVMCDFGNode* AnceNode = node->getInputPort(0);
        for(int edgeid : node->outputEdges())
        {
          LLVMCDFGNode* output = CDFG->edge(edgeid)->dst();
          assert(output != NULL);
          // output's input is node (the cast we are removing), not AnceNode
          const auto &inputMap = output->inputInfoMap();
          auto it = inputMap.find(node);
          if(it != inputMap.end()){
            const std::vector<NodeInfo> &infos = it->second;
            for(const NodeInfo &info : infos){
              output->addInputNode(AnceNode, info.idx, info.isBackEdge);
              AnceNode->addOutputNode(output, info.isBackEdge);
              CDFG->addEdge(AnceNode, output); //To fix: Edge Type
            }
          } else {
            int edgeidx = output->getInputIdx(node);
            bool isbackedge = output->isInputBackEdge(node);
            output->addInputNode(AnceNode, edgeidx, isbackedge);
            AnceNode->addOutputNode(output, isbackedge);
            CDFG->addEdge(AnceNode, output); //To fix: Edge Type
          }
        }
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "for"){
        ////// for to input/output
        const std::vector<LLVMCDFGNode *>& sinknodes = node->outputNodes();
        // for(LLVMCDFGNode* sinknode : sinknodes){
        //   if( ( sinknode->getTypeName()=="Input" || sinknode->getTypeName()=="INPUT"
        //       ||sinknode->getTypeName()=="Output" || sinknode->getTypeName()=="OUTPUT" )
        //     && sinknode->isLSaffine() ) {
        //       CDFG->delEdge(CDFG->edge(node, sinknode));
        //       node->delOutputNode(sinknode);
        //       sinknode->delInputNode(node); 
        //       LLVM_DEBUG( llvm::errs()<< node->inputNodes().size()<< " " << node->outputNodes().size()<< " ");
        //   }
        // }
        // LLVM_DEBUG( llvm::errs()<< node->inputNodes().size()<< " " << node->outputNodes().size()<< " ");
        // if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_3_CDFG.dot");}
        assert(node->inputNodes().size() == 0 && node->outputNodes().size() == 0);
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "truncf"){
        assert(node->inputNodes().size() == 1 && node->outputNodes().size() == 0);
        CDFG->delNode(node);
        removing = 1;
      }
      else if(node->getTypeName() == "CONST"){
        if(node->inputNodes().size() == 0 && node->outputNodes().size() == 0){
          // Useless constant 
          CDFG->delNode(node);
          removing = 1;
        }  
        else if(node->outputNodes().size() != 0){
          if(dyn_cast<arith::ConstantOp>(node->operation()).getValue().getType().isIndex()){
            int i = 0;
            for(i = 0; i < node->outputNodes().size(); i++){
              LLVMCDFGNode* output = node->outputNodes()[i];
              if(verbose) {output->operation()->dump();}
              if((output->getTypeName()=="Input" || output->getTypeName()=="INPUT")
              &&(output->isLinearAccess()))
                continue;
              else
                break;
            }
            if(i == node->outputNodes().size()){
              /// All output node is linear access input, constant node can be removed
              CDFG->delNode(node);
              removing = 1;
            }
          }
        }
      }
    }    
  }

  ////////////////////////
  /// Handle floatpoint bitwidth
  ////////////////////////
  SpecifyFPNodePrecision(CDFG, verbose);

  ////////////////////////
  /// fuse operators: MAC, FMAC32
  ////////////////////////
  if(verbose) { CDFG->CDFGtoDOT(CDFG->name_str()+"_3_CDFG.dot");}
  FuseOperators(CDFG, verbose);

  return true;
}


LLVMCDFG* mlir::ADORA::generateCDFGfromKernel(LLVMCDFG* &CDFG, ADORA::KernelOp kernel, bool verbose){
  
  
  /// remove truncf op
  RemoveConstantTruncF(kernel);

  // hjy
  // [Add] If-Conversion (scf.if -> arith.select)
  lowerSCFIfToSelect(kernel);
  if(verbose) {
    llvm::errs() << "[ADORA] Applied If-Conversion (scf.if -> arith.select).\n";
    kernel.dump(); // 打印 IR 查看转换结果
  }
  // [Add] End

  ADORA::KernelOp kernel_clone = kernel.clone();
  // kernel_clone.dump();
  kernel.getOperation()->getBlock()->push_back(kernel_clone);
  kernel_clone->moveBefore(kernel);

  /// Hoist load store op
  /// FIX: We should judge whether to use hoist
  HoistLoadStoreInKernelOp(kernel); 
  if(verbose) kernel.dump();

  

  /// For every loop carried value, find their accumulation mode. Move initial value computing to outer most level.
  MoveLoopCarriedInitailValue(kernel);
  if(verbose) kernel.dump();

  /// For accumulation chain, move accumulation operation to the last using commutative law of addition/multiplication
  MoveAccumulationToLast(kernel);
  if(verbose) kernel.dump();

  /// insert ISEL operator for loop carried value(not acc)
  InsertIselForLoopCarry(kernel, verbose);
  if(verbose) kernel.dump();

  /// Generate
  if(generateCDFGfromKernelAfterOptimization(CDFG, kernel, verbose)){
    kernel_clone.erase();
  }
  else{
    if(verbose) llvm::errs() << "[Mion] CDFG generation failed. Try again with no opt.\n";
    if(verbose) kernel.dump();
    if(verbose) kernel_clone.dump();
    LLVMCDFG* NewCDFG = new LLVMCDFG(CDFG->name_str(), CDFG->getOpNameFilePath());
    delete CDFG;
    CDFG = NewCDFG;
    if(generateCDFGfromKernelAfterOptimization(CDFG, kernel_clone, verbose)){
      kernel.erase();
    }
    else{
      llvm::errs() << "[ERROR] CDFG generation failed again. Abort.\n";
      abort();
    }
  }
  
  return CDFG;
}
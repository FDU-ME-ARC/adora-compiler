//===- Test.h - Test dialect --------------------------------------*- C++ -*-===//
//===----------------------------------------------------------------------===//

#ifndef CGRAOPT_DIALECT_ADORA_UTILITY_H_
#define CGRAOPT_DIALECT_ADORA_UTILITY_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
//===----------------------------------------------------------------------===//
// ADORA Dialect Helpers
//===----------------------------------------------------------------------===//

namespace mlir {
namespace ADORA {
///////////////
/// AdjustMemoryFootprint.cpp
///////////////
std::optional<int64_t> getSingleMemrefAccessSpace(::mlir::affine::AffineForOp forOp);

///////////////
/// AffineLoopSimplify.cpp
///////////////
LogicalResult simplifyLoopLevelsInRegion(mlir::Region& region, bool donttouchkernel = false);
///////////////
/// SimplifyLoadStore.cpp
///////////////
enum class PositionRelationInLoop { SameLevel, LhsOuter, RhsOuter, NotInSameLoopNest}; 
PositionRelationInLoop getPositionRelationship(Operation* lhs, Operation* rhs);
template <typename LoadOrStoreT, typename OpToWalkT>
  SmallVector<LoadOrStoreT,  4> GetAllHoistOp(OpToWalkT op_tocheck);

///////////////
/// SimplifyLoadStoreUtil.cpp
///////////////
void SimplifyLoadStoreOpsInRegion(mlir::Region& region);

///////////////
/// Utilities.cpp
///////////////
// Define supported affine operations on CGRA
typedef std::set<StringRef> OpTable;

mlir::Operation* eraseKernel(::mlir::func::FuncOp& TopFunc, ADORA::KernelOp& Kernel);
LogicalResult SpecifiedAffineFortoKernel(::mlir::affine::AffineForOp& forOp);
LogicalResult SpecifiedAffineFortoKernel(::mlir::affine::AffineForOp& forOp, std::string kernel_name);
AffineExpr getConstPartofAffineExpr(AffineExpr& expr);
signed MultiplicatorOfDim(const AffineExpr& expr, const unsigned dim);
// void removeUnusedRegionArgs(Region &region);
void eliminateUnusedIndices(Operation *op);
::llvm::SmallDenseMap<mlir::Value, unsigned> getOperandInRank(Operation *op, unsigned rank);
SmallVector<mlir::Operation*>  getAllUsesInRegion(const mlir::Value beused, ::mlir::Region* region);
SmallVector<mlir::Operation*>  getAllUsesInBlock(const mlir::Value beused, ::mlir::Block* block);

mlir::Operation* GetTheSourceOperationOfBlockStore(ADORA::DataBlockStoreOp store);

void ResetIndexOfBlockAccessOpInFunc(func::FuncOp& func);
inline bool opIsContainedByKernel(mlir::Operation* op);

void LLVM_ATTRIBUTE_UNUSED 
 simplifyMapWithOperands(AffineMap &map, ArrayRef<mlir::Value> operands);

///// following 3 functions are defined to simplify AffineApplyOp
void simplifyConstantAffineApplyOpsInRegion(::mlir::Region& region);
void simplifyAddAffineApplyOpsInRegionButOutOfKernel(::mlir::Region& region);
void simplifyLoadAndStoreOpsInRegion(::mlir::Region& region);

///// following 4 functions are defined to help extract kernel function
// bool isSinkingBeneficiary(Operation *op);
// static bool extractBeneficiaryOps(Operation *op, llvm::SetVector<Value> existingDependencies,
//       llvm::SetVector<Operation *> &beneficiaryOps, llvm::SmallPtrSetImpl<Value> &availableValues);
LogicalResult sinkOperationsIntoKernelOp(ADORA::KernelOp kernelOp);
ADORA::KernelOp getSingleKernelFromFunc(func::FuncOp func);
ADORA::KernelOp getKernelFromCopiedModule(ModuleOp ModuleOp, ADORA::KernelOp kernel);

func::FuncOp GenKernelFunc(ADORA::KernelOp KernelOp, llvm::SetVector<Value> &operands);
func::FuncOp ConvertMatmulToFunc(mlir::linalg::MatmulOp op, llvm::SetVector<mlir::Value> &operands, std::string FnName);
bool IsIterationSpaceSupported(mlir::affine::AffineForOp &forOp);

bool LoadStoreSameMemAddr(::mlir::affine::AffineLoadOp loadop, ::mlir::affine::AffineStoreOp storeop);
SmallVector<int> getOperandDimensionsInMap(const int dim, const ::mlir::AffineMap map);
unsigned  getInstanceNumFromADG  (const std::string& CGRAadg, const std::string& instype_to_count);
int64_t getIntegerAttrFromADG(const std::string &CGRAadg, const std::string &key);

ADORA::IselOp ReplaceValueWithNewIselOp(OpBuilder b, Location loc, mlir::Value value);
ADORA::IselOp ReplaceLoopCarryValueWithNewIselOp(affine::AffineForOp& forop, int IterRegionOperandIdx);

std::optional<mlir::affine::AffineForOp> MoveLoadStorePairOut(::mlir::affine::AffineLoadOp loadop, ::mlir::affine::AffineStoreOp storeop);

ADORA::KernelOp findTheOnlyKernelInNestedLoop(affine::AffineForOp forOp);

template <typename T>
inline SmallVector<T> SetMergeForVector(const llvm::SmallVector<T>& v1, const llvm::SmallVector<T>& v2){
  SmallVector<T> v;
  for(auto e : v1){
    if(findElement(v, e) == -1){
      v.push_back(e);
    }
  }
  for(auto e : v2){
    if(findElement(v, e) == -1){
      v.push_back(e);
    }
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Async-token accessors (PR2)
//
// Uniform helpers to inspect the async form of the three async-capable ADORA
// ops: DataBlockLoadOp / DataBlockStoreOp / KernelOp. Callers that do not
// know the concrete op class use these to read asyncDependencies / asyncToken
// without branching on op type themselves.
//===----------------------------------------------------------------------===//

/// Return the !ADORA.token produced by `op`, or a null Value when `op` is
/// either not async-capable or currently in its synchronous form (i.e.,
/// produceToken=false at construction time).
inline ::mlir::Value getAsyncTokenOrNull(::mlir::Operation *op) {
  if (auto l = ::mlir::dyn_cast_or_null<DataBlockLoadOp>(op))
    return l.getAsyncToken();
  if (auto s = ::mlir::dyn_cast_or_null<DataBlockStoreOp>(op))
    return s.getAsyncToken();
  if (auto k = ::mlir::dyn_cast_or_null<KernelOp>(op))
    return k.getAsyncToken();
  return {};
}

/// Return the asyncDependencies operand range of `op`. For non-async-capable
/// ops, returns an empty OperandRange so callers can iterate unconditionally.
inline ::mlir::OperandRange getAsyncDeps(::mlir::Operation *op) {
  if (auto l = ::mlir::dyn_cast<DataBlockLoadOp>(op))
    return l.getAsyncDependencies();
  if (auto s = ::mlir::dyn_cast<DataBlockStoreOp>(op))
    return s.getAsyncDependencies();
  if (auto k = ::mlir::dyn_cast<KernelOp>(op))
    return k.getAsyncDependencies();
  return op->getOperands().take_front(0);
}

/// True iff `op` is one of the three async-capable ADORA ops.
inline bool isAsyncCapable(::mlir::Operation *op) {
  return ::mlir::isa_and_nonnull<DataBlockLoadOp, DataBlockStoreOp, KernelOp>(
      op);
}

} // namespace ADORA
} // namespace mlir

#endif //CGRAOPT_DIALECT_ADORA_IR_Test_H_

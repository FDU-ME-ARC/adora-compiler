//===- KernelOp.cpp - KernelOp of the ADORA dialect -------------------------===//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
// #include "mlir/Dialect/Affine/IR/AffineValueMap.h"
// #include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "llvm/ADT/STLExtras.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "ADORA/Dialect/ADORA/IR/ADORAOps.h.inc"


using namespace mlir;
using namespace mlir::ADORA;
using namespace mlir::func;
using namespace mlir::affine;

//===----------------------------------------------------------------------===//
// KernelOp
//===----------------------------------------------------------------------===//
void KernelOp::build(OpBuilder &builder, OperationState &result) {

  // Add the data operands.

  // This is a good area to add static operands.

  // Create a kernel body region with kNumConfigRegionAttributes + N arguments,
  // where the first kNumConfigRegionAttributes arguments have `index` type and
  // the rest have the same types as the data operands.
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  kernelRegion->push_back(body);
}

void KernelOp::build(OpBuilder &builder, OperationState &result, std::string KernelName) {
  auto KernelNameAttr = builder.getStringAttr(KernelName);
  result.addAttribute(getKernelNameAttrStr(), KernelNameAttr);
  build(builder, result);
}

void KernelOp::build(OpBuilder &builder, OperationState &result,
                     std::string KernelName, ValueRange asyncDependencies,
                     bool produceToken) {
  result.addOperands(asyncDependencies);
  if (produceToken)
    result.addTypes(TokenType::get(builder.getContext()));
  if (!KernelName.empty())
    result.addAttribute(getKernelNameAttrStr(),
                        builder.getStringAttr(KernelName));
  Region *kernelRegion = result.addRegion();
  Block *body = new Block();
  for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
    body->addArgument(builder.getIndexType(), result.location);
  kernelRegion->push_back(body);
}

/// Rebuild-friendly async builder.
///
/// Constructs a new KernelOp that carries `asyncDependencies` and optionally
/// produces an `asyncToken`, while transferring the body region from
/// `takeFromRegion` (typically the body of the old sync KernelOp we are
/// replacing) via Region::takeBody. This preserves the original kernel body
/// verbatim and avoids cloning costs.
///
/// @param KernelName        name attribute of the new kernel (may be empty)
/// @param asyncDependencies SSA !ADORA.token values this kernel waits on
/// @param produceToken      whether to emit an asyncToken result
/// @param takeFromRegion    region whose body is moved into the new kernel;
///                          if null, a fresh empty body is created.
/// Side-effects: `*takeFromRegion` is emptied after the call.
void KernelOp::build(OpBuilder &builder, OperationState &result,
                     std::string KernelName, ValueRange asyncDependencies,
                     bool produceToken, Region *takeFromRegion) {
  // 1. Wire operands/result-types identically to the non-rebuild overload.
  result.addOperands(asyncDependencies);
  if (produceToken)
    result.addTypes(TokenType::get(builder.getContext()));
  if (!KernelName.empty())
    result.addAttribute(getKernelNameAttrStr(),
                        builder.getStringAttr(KernelName));

  // 2. Either steal the caller-provided region or synthesize an empty one.
  Region *kernelRegion = result.addRegion();
  if (takeFromRegion && !takeFromRegion->empty()) {
    // NOTE: takeBody transfers all blocks over; the source region is emptied.
    kernelRegion->takeBody(*takeFromRegion);
  } else {
    Block *body = new Block();
    for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
      body->addArgument(builder.getIndexType(), result.location);
    kernelRegion->push_back(body);
  }
}

// void KernelOp::getCanonicalizationPatterns(RewritePatternSet &results,
//                                           MLIRContext *context) {
// }
LogicalResult KernelOp::verify() {
  //  Include this code if Kernel launch takes KNumConfigOperands leading
  //  operands for grid/block sizes and transforms them into
  //  kNumConfigRegionAttributes region arguments for block/thread identifiers
  //  and grid/block sizes.
  // if (!op.body().empty()) {
  //   if (op.body().getNumArguments() !=
  //       LaunchOp::kNumConfigOperands + op.getNumOperands())
  //     return op.emitOpError("unexpected number of region arguments");
  // }

  /// To Fix
  for (Block &block : getBody()) {
    if (block.empty())
      continue;
    if (block.back().getNumSuccessors() != 0)
      continue;
  }
  
  Region& knRegion = getBody();
  /// A kernel should only contain 1 region
  /// and this region should only contain 1 block
  if(knRegion.getBlocks().size() != 1)
    return emitOpError(
        "Kernel Region should only get 1 Block.");
  Block& knBlock = knRegion.front();
  /// this kernel Block should only contain 2 Op(a forOp and a terminatorOp)
  /// To fix: maybe not a forOp
  // if(knBlock.getOperations().size() != 2 )
  //   return emitOpError(
  //       "kernel Block should only get 2 Ops.");

  /// Async token invariant: consuming tokens requires producing one.
  if (!getAsyncDependencies().empty() && !getAsyncToken())
    return emitOpError(
        "has asyncDependencies but does not produce an asyncToken");
  return success();
}

void KernelOp::print(OpAsmPrinter &printer) {
  if (getAsyncToken()) {
    printer << " async";
    if (!getAsyncDependencies().empty()) {
      printer << " [";
      llvm::interleaveComma(getAsyncDependencies(), printer.getStream(),
                            [&](Value v) { printer.printOperand(v); });
      printer << "]";
    }
  }
  printer << ' ';
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/true,/*printBlockTerminators=*/true);
  printer.printOptionalAttrDict((*this)->getAttrs());
}

// Parses a Launch operation.
// operation ::= `gpu.launch` region attr-dict?
// ssa-reassignment ::= `(` ssa-id `=` ssa-use (`,` ssa-id `=` ssa-use)* `)`
ParseResult KernelOp::parse(OpAsmParser &parser, OperationState &result) {

  // Optional "async" keyword: if present, produce an asyncToken result and
  // optionally parse a [%t0, %t1, ...] dependency list.
  bool isAsync = succeeded(parser.parseOptionalKeyword("async"));
  SmallVector<OpAsmParser::UnresolvedOperand, 4> asyncDeps;
  if (isAsync && succeeded(parser.parseOptionalLSquare())) {
    if (parser.parseOperandList(asyncDeps) || parser.parseRSquare())
      return failure();
  }

  // Region arguments to be created.
  SmallVector<OpAsmParser::UnresolvedOperand, 16> regionArgs(
      KernelOp::kNumConfigRegionAttributes);
  MutableArrayRef<OpAsmParser::UnresolvedOperand> regionArgsRef(regionArgs);

  // Introduce the body region and parse it. The region has
  // kNumConfigRegionAttributes arguments that correspond to
  // block/thread identifiers and grid/block sizes, all of the `index` type.
  Type index = parser.getBuilder().getIndexType();
  SmallVector<Type, KernelOp::kNumConfigRegionAttributes> dataTypes(
      KernelOp::kNumConfigRegionAttributes, index);

  SmallVector<OpAsmParser::Argument> regionArguments;
  for (auto ssaValueAndType : llvm::zip(regionArgs, dataTypes)) {
    OpAsmParser::Argument arg;
    arg.ssaName = std::get<0>(ssaValueAndType);
    arg.type = std::get<1>(ssaValueAndType);
    regionArguments.push_back(arg);
  }

  Region *body = result.addRegion();
  if (parser.parseRegion(*body, regionArguments) ||
      parser.parseOptionalAttrDict(result.attributes)) {
    return failure();
  }

  // Resolve async dependencies and add asyncToken result.
  if (isAsync || !asyncDeps.empty()) {
    Type tokenTy = TokenType::get(result.getContext());
    if (parser.resolveOperands(asyncDeps, tokenTy, result.operands))
      return failure();
    result.addTypes(tokenTy);
  }
  return success();
}

//=======================================
//=======================================

LogicalResult mlir::ADORA::specifyOneOperationToADORAKernel(Operation *op) {
  assert(op && "specifyOneOperationToADORAKernel: null op");
  if (op->getParentOfType<ADORA::KernelOp>())
    return LogicalResult::failure();

  OpBuilder builder(op);
  Location loc = op->getLoc();

  auto kernel = builder.create<ADORA::KernelOp>(loc);

  Block &entry = kernel.getBody().front();
  builder.setInsertionPointToEnd(&entry);
  auto term = builder.create<ADORA::TerminatorOp>(loc);

  op->remove();
  entry.getOperations().insert(Block::iterator(term.getOperation()), op);

  return LogicalResult::success();
}

LogicalResult mlir::ADORA::specifyOneOperationToADORAKernel(Operation *op, std::string kernel_name) {
  if(specifyOneOperationToADORAKernel(op).succeeded()){
    if(kernel_name != "")
      dyn_cast<ADORA::KernelOp>(op->getParentOp()).setKernelName(kernel_name);
    return LogicalResult::success();
  }

  return LogicalResult::failure();
}


#define GET_OP_CLASSES
#include "ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.cpp.inc"
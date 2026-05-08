//===- AsyncRuntimeToLLVM.cpp - Lower ADORA async event ops to llvm.call --===//
//
// PR3 commit C.  Translates the four event ops inserted by
// adora-lower-async-tokens into llvm.call instructions that target the
// adora_async_rt.h runtime ABI:
//
//   ADORA.event.create  -> %ptr = llvm.call @adoraEventCreate()
//   ADORA.event.destroy -> llvm.call @adoraEventDestroy(%ptr)
//   ADORA.signal        -> llvm.call @adoraEventRecord(%ptr, %stream_i64)
//   ADORA.wait          -> llvm.call @adoraEventWait(%ptr, %stream_i64)
//
// !ADORA.token is replaced by llvm.ptr (opaque pointer).
// The four extern declarations are inserted once at module scope.
//
//===----------------------------------------------------------------------===//

#include "LowerPassDetail.h"
#include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

using namespace mlir;
using namespace mlir::ADORA;

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the opaque pointer type used to represent AdoraEvent at the LLVM IR
/// level.  Matches the `typedef void *AdoraEvent` in adora_async_rt.h.
static LLVM::LLVMPointerType eventPtrTy(MLIRContext *ctx) {
  return LLVM::LLVMPointerType::get(ctx);
}

/// Return the LLVM void type.
static LLVM::LLVMVoidType voidTy(MLIRContext *ctx) {
  return LLVM::LLVMVoidType::get(ctx);
}

/// Ensure that a zero-argument or one/two-argument extern LLVM function
/// declaration exists in the module, creating it if absent.  Returns the
/// FlatSymbolRefAttr for use in llvm.call.
static FlatSymbolRefAttr getOrInsertFn(
    OpBuilder &b, ModuleOp module, StringRef name,
    Type resultTy,
    ArrayRef<Type> argTys) {
  if (module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return FlatSymbolRefAttr::get(b.getContext(), name);

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToStart(module.getBody());
  auto fnTy = LLVM::LLVMFunctionType::get(resultTy, argTys,
                                           /*isVarArg=*/false);
  b.create<LLVM::LLVMFuncOp>(module.getLoc(), name, fnTy,
                               LLVM::Linkage::External);
  return FlatSymbolRefAttr::get(b.getContext(), name);
}

// ---------------------------------------------------------------------------
// Pattern: ADORA.event.create  ->  %ev = llvm.call @adoraEventCreate()
// ---------------------------------------------------------------------------
struct EventCreateLowering : public OpConversionPattern<EventCreateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(EventCreateOp op,
                                OpAdaptor /*adaptor*/,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder b(op);
    auto ctx = op->getContext();
    auto fnRef = getOrInsertFn(b, module, "adoraEventCreate",
                               eventPtrTy(ctx), {});
    auto call = rewriter.create<LLVM::CallOp>(
        op.getLoc(), TypeRange{eventPtrTy(ctx)},
        fnRef, ValueRange{});
    rewriter.replaceOp(op, call.getResult());
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pattern: ADORA.event.destroy %ev  ->  llvm.call @adoraEventDestroy(%ev)
// ---------------------------------------------------------------------------
struct EventDestroyLowering : public OpConversionPattern<EventDestroyOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(EventDestroyOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder b(op);
    auto ctx = op->getContext();
    auto fnRef = getOrInsertFn(b, module, "adoraEventDestroy",
                               voidTy(ctx), {eventPtrTy(ctx)});
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                   fnRef, ValueRange{adaptor.getToken()});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pattern: ADORA.signal %ev on stream N  ->  llvm.call @adoraEventRecord(%ev, N)
// ---------------------------------------------------------------------------
struct SignalLowering : public OpConversionPattern<SignalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(SignalOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder b(op);
    auto ctx = op->getContext();
    auto i64 = IntegerType::get(ctx, 64);
    auto fnRef = getOrInsertFn(b, module, "adoraEventRecord",
                               voidTy(ctx), {eventPtrTy(ctx), i64});
    // Materialise the stream id as an i64 constant.
    Value streamId = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(op.getStream())));
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                   fnRef,
                                   ValueRange{adaptor.getToken(), streamId});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pattern: ADORA.wait %ev on stream N  ->  llvm.call @adoraEventWait(%ev, N)
// ---------------------------------------------------------------------------
struct WaitLowering : public OpConversionPattern<WaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(WaitOp op,
                                OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    auto module = op->getParentOfType<ModuleOp>();
    OpBuilder b(op);
    auto ctx = op->getContext();
    auto i64 = IntegerType::get(ctx, 64);
    auto fnRef = getOrInsertFn(b, module, "adoraEventWait",
                               voidTy(ctx), {eventPtrTy(ctx), i64});
    Value streamId = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), i64,
        rewriter.getI64IntegerAttr(static_cast<int64_t>(op.getStream())));
    rewriter.create<LLVM::CallOp>(op.getLoc(), TypeRange{},
                                   fnRef,
                                   ValueRange{adaptor.getToken(), streamId});
    rewriter.eraseOp(op);
    return success();
  }
};

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ADORAAsyncRuntimeToLLVMPass
    : public ADORAAsyncRuntimeToLLVMBase<ADORAAsyncRuntimeToLLVMPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // Type converter: !ADORA.token  →  llvm.ptr
    LLVMTypeConverter typeConverter(ctx);
    typeConverter.addConversion([&](TokenType) -> Type {
      return LLVM::LLVMPointerType::get(ctx);
    });

    RewritePatternSet patterns(ctx);
    patterns.add<EventCreateLowering,
                 EventDestroyLowering,
                 SignalLowering,
                 WaitLowering>(typeConverter, ctx);

    ConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    // All four event ops must be lowered.
    target.addIllegalOp<EventCreateOp, EventDestroyOp,
                        SignalOp, WaitOp>();
    // Everything else stays.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::ADORA::createADORAAsyncRuntimeToLLVMPass() {
  return std::make_unique<ADORAAsyncRuntimeToLLVMPass>();
}

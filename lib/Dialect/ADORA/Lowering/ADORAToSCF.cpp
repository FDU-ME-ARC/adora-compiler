//===- FuncToLLVM.cpp - Func to LLVM dialect conversion -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert ADORA dialect
// into builtin scf dialect.
//
//===----------------------------------------------------------------------===//


#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FormatVariadic.h"

#include "./LowerPassDetail.h"
#include "ADORA/Dialect/ADORA/Lowering/LowerPasses.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"

using namespace llvm; // for llvm.errs()
using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;

#define PASS_NAME "ADORA-convert-loadstore-to-scf"

namespace mlir {
namespace ADORA {

/// @brief Helper: convert outer affine.for to scf.for (inner loops stay affine).
/// @param func function to transform
/// @param outerDepth max outer loop depth to convert (default 1)
/// @return success/failure
LogicalResult affineForOuterToSCF(func::FuncOp func, unsigned outerDepth) {
  // Collect top-level affine.for ops first to avoid iterator invalidation
  SmallVector<AffineForOp> toConvert;
  func.walk([&](AffineForOp forop) {
    // Check depth by counting parent affine.for ops
    unsigned depth = 0;
    Operation *parent = forop->getParentOp();
    while (parent) {
      if (isa<AffineForOp>(parent))
        depth++;
      parent = parent->getParentOp();
    }
    if (depth < outerDepth) {
      toConvert.push_back(forop);
    }
  });

  // Now convert each collected affine.for to scf.for
  for (auto forop : toConvert) {
    OpBuilder builder(forop);
    Location loc = forop.getLoc();

    // Step 1: Expand lb, ub, step
    auto expand = [&](AffineMap map, ValueRange operands) -> Value {
      auto expanded = expandAffineMap(builder, loc, map, operands);
      if (!expanded || expanded->size() != 1) {
        return Value();
      }
      return (*expanded)[0];
    };

    Value lb = expand(forop.getLowerBoundMap(), forop.getLowerBoundOperands());
    Value ub = expand(forop.getUpperBoundMap(), forop.getUpperBoundOperands());
    Value step = builder.create<arith::ConstantIndexOp>(loc, forop.getStep().getSExtValue());
    if (!lb || !ub) {
      continue; // skip, too complex
    }

    // Step 2: Create scf.for
    scf::ForOp scfFor;
    if (forop.getNumResults() == 0) {
      scfFor = builder.create<scf::ForOp>(loc, lb, ub, step);
    } else {
      // With iter args (not yet handled; skip for now)
      continue;
    }

    // Step 3: Move body
    Block *srcBody = forop.getBody();
    // Save terminator info first (affine.yield)
    Operation *terminator = srcBody->getTerminator();
    // Move body ops first
    scfFor.getBody()->clear();
    scfFor.getBody()->getOperations().splice(
        scfFor.getBody()->begin(),
        srcBody->getOperations(),
        srcBody->begin(),
        std::prev(srcBody->end())); // stop before terminator!
    // Now add scf.yield (empty, since affine.yield is empty)
    {
      OpBuilder b(forop.getContext());
      b.setInsertionPointToEnd(scfFor.getBody());
      b.create<scf::YieldOp>(loc);
    }
    // Now erase old terminator
    terminator->erase();

    // Step 4: Replace induction variable uses
    srcBody->getArgument(0).replaceAllUsesWith(scfFor.getInductionVar());

    // Step 5: Replace the original affine.for with the new scf.for
    forop.replaceAllUsesWith(scfFor.getResults());
    forop.erase();
  }
  return success();
}

} // namespace ADORA
} // namespace mlir

namespace {

/// Lower affine.load to memref.load by expanding affine map.
struct AffineLoadOpLowering : public OpRewritePattern<AffineLoadOp> {
  using OpRewritePattern<AffineLoadOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineLoadOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = expandAffineMap(rewriter, op.getLoc(),
                                   op.getAffineMap(), op.getMapOperands());
    if (!indices)
      return failure();
    rewriter.replaceOpWithNewOp<memref::LoadOp>(op, op.getMemRef(), *indices);
    return success();
  }
};

/// Lower affine.store to memref.store by expanding affine map.
struct AffineStoreOpLowering : public OpRewritePattern<AffineStoreOp> {
  using OpRewritePattern<AffineStoreOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(AffineStoreOp op,
                                PatternRewriter &rewriter) const override {
    auto indices = expandAffineMap(rewriter, op.getLoc(),
                                   op.getAffineMap(), op.getMapOperands());
    if (!indices)
      return failure();
    rewriter.replaceOpWithNewOp<memref::StoreOp>(op, op.getValue(),
                                                 op.getMemRef(), *indices);
    return success();
  }
};

} // namespace

namespace {
/// A pass converting Func operations into the LLVM IR dialect.
struct ConvertADORAToSCFPass
    : public ConvertADORAToSCFBase<ConvertADORAToSCFPass> {
  ConvertADORAToSCFPass() = default;

  SmallVector<::llvm::StringRef, 8> KernelNameVec;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    RewritePatternSet patterns(&getContext());
    patterns.add<
        AffineLoadOpLowering, AffineStoreOpLowering
    >(patterns.getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, scf::SCFDialect, memref::MemRefDialect>();
    // Legalize ADORA ops and any remaining affine ops
    target.addLegalDialect<ADORA::ADORADialect>();
    target.addLegalOp<AffineForOp, AffineLoadOp, AffineStoreOp>();

    if (failed(applyPartialConversion(m, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::ADORA::createConvertADORAToSCFPass() {
  return std::make_unique<ConvertADORAToSCFPass>();
}

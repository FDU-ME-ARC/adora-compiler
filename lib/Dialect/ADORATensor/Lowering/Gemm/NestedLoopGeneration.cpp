//===------------------ NestedLoopGeneration.cpp - ADORATensor Lower process util functions ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;
using namespace ::mlir;

namespace mlir
{
  namespace ADORA
  {

    mlir::Value getConstantOpAccordingToDataType(OpBuilder &builder, Location loc, Type datatype, float value)
    {
      arith::ConstantOp newconst;
      if (datatype.isBF16() || datatype.isF32())
      {
        float constvalue = value;
        FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
        newconst = builder.create<arith::ConstantOp>(loc, datatype, constAttr);
      }
      else if (datatype.isF64())
      {
        double constvalue = (double)value;
        FloatAttr constAttr = FloatAttr::get(datatype, constvalue);
        newconst = builder.create<arith::ConstantOp>(loc, datatype, constAttr);
      }
      else
      {
        int64_t constvalue = (int)value;
        IntegerAttr constAttr = IntegerAttr::get(datatype, constvalue);
        newconst = builder.create<arith::ConstantOp>(loc, datatype, constAttr);
      }
      return newconst.getResult();
    }

    /// @brief Create an `arith.add` op for the given operands.
    ///
    /// Selects `AddFOp` for floating-point types (bf16/f32/f64),
    /// otherwise uses `AddIOp` for integer types.
    /// Returns the created operation.
    mlir::Operation *genArithAddOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs)
    {
      assert(lhs.getType() == rhs.getType());
      mlir::Type datatype;
      if (isa<mlir::VectorType>(lhs.getType()))
      {
        datatype = dyn_cast<mlir::VectorType>(lhs.getType()).getElementType();
      }
      else
      {
        datatype = lhs.getType();
      }

      mlir::Operation *add;
      if (datatype.isBF16() || datatype.isF32() || datatype.isF64())
      {
        add = builder.create<arith::AddFOp>(loc, lhs, rhs);
      }
      else
      {
        add = builder.create<arith::AddIOp>(loc, lhs, rhs);
      }
      return add;
    }

    mlir::Operation *genArithMulOpAccordingToDataType(OpBuilder &builder, Location loc, mlir::Value lhs, mlir::Value rhs)
    {
      assert(lhs.getType() == rhs.getType());
      mlir::Type datatype;
      if (isa<mlir::VectorType>(lhs.getType()))
      {
        datatype = dyn_cast<mlir::VectorType>(lhs.getType()).getElementType();
      }
      else
      {
        datatype = lhs.getType();
      }

      mlir::Operation *mul;
      if (datatype.isBF16() || datatype.isF32() || datatype.isF64())
      {
        mul = builder.create<arith::MulFOp>(loc, lhs, rhs);
      }
      else
      {
        mul = builder.create<arith::MulIOp>(loc, lhs, rhs);
      }
      return mul;
    }

    mlir::Operation *initOutWithC2DLike(
        OpBuilder &b, Location loc,
        Value out, Value C,
        ArrayRef<int64_t> outMN /*{M,N}*/)
    {

      auto outTy = out.getType().dyn_cast<MemRefType>();
      auto cTy = C.getType().dyn_cast<MemRefType>();
      assert(outTy && cTy && "out/C must be memref");
      assert(outMN.size() == 2);

      // Fast path: exactly same memref type => memref.copy
      if (outTy == cTy)
      {
        return b.create<memref::CopyOp>(loc, C, out).getOperation();
      }

      int64_t M = outMN[0];
      int64_t N = outMN[1];

      // affine.for m = 0..M
      auto mFor = b.create<affine::AffineForOp>(loc, 0, M, 1);
      OpBuilder mb(mFor.getBody(), mFor.getBody()->begin());
      Value mIv = mFor.getInductionVar();

      // affine.for n = 0..N
      auto nFor = mb.create<affine::AffineForOp>(loc, 0, N, 1);
      OpBuilder nb(nFor.getBody(), nFor.getBody()->begin());
      Value nIv = nFor.getInductionVar();

      // Helpers.
      auto c0 = nb.create<arith::ConstantIndexOp>(loc, 0);

      int64_t rC = cTy.getRank();
      Value cVal;

      if (rC == 0)
      {
        cVal = nb.create<memref::LoadOp>(loc, C, ValueRange{}); // affine.load doesn't support rank-0 well; keep memref.load
      }
      else if (rC == 1)
      {
        // [N] or [1]
        int64_t d0 = cTy.getShape()[0];
        Value i0 = (d0 == 1) ? c0 : nIv;
        cVal = nb.create<affine::AffineLoadOp>(loc, C, ValueRange{i0});
      }
      else if (rC == 2)
      {
        // [M,N] or [1,N] or [M,1] or [1,1]
        int64_t d0 = cTy.getShape()[0];
        int64_t d1 = cTy.getShape()[1];
        Value i0 = (d0 == 1) ? c0 : mIv;
        Value i1 = (d1 == 1) ? c0 : nIv;
        cVal = nb.create<affine::AffineLoadOp>(loc, C, ValueRange{i0, i1});
      }
      else if (rC == 3)
      {
        // common: [1,M,N] / [1,1,N]
        int64_t d1 = cTy.getShape()[1];
        int64_t d2 = cTy.getShape()[2];
        Value i0 = c0;
        Value i1 = (d1 == 1) ? c0 : mIv;
        Value i2 = (d2 == 1) ? c0 : nIv;
        cVal = nb.create<affine::AffineLoadOp>(loc, C, ValueRange{i0, i1, i2});
      }
      else
      {
        assert(false && "Unsupported C rank for initOutWithC2DLikeAffine");
      }

      // Store into out (support out rank 2 or 3 (batch==1))
      int64_t rO = outTy.getRank();
      if (rO == 2)
      {
        nb.create<affine::AffineStoreOp>(loc, cVal, out, ValueRange{mIv, nIv});
      }
      else if (rO == 3)
      {
        Value b0 = c0;
        nb.create<affine::AffineStoreOp>(loc, cVal, out, ValueRange{b0, mIv, nIv});
      }
      else
      {
        assert(false && "Unsupported out rank for init");
      }

      // Important: affine.for body must end with affine.yield
      // nb.create<affine::AffineYieldOp>(loc);
      mb.setInsertionPointAfter(nFor);
      // mb.create<affine::AffineYieldOp>(loc);
      return nFor.getOperation();
    }

    /// @brief Generate a nested affine.for loop on device(data transfer is already done).
    ///  This function is shared by IS and WS
    ///
    /// Creates a loop nest of depth `level` with upper bounds
    /// specified by `Upperbounds`, and invokes the provided
    /// `BodyBuilder` at the innermost level.
    /// The function returns the outermost `AffineForOp`.
    ///
    /// @param builder     MLIR OpBuilder for loop creation
    /// @param loc         Source location
    /// @param level       Nesting depth (must > 0)
    /// @param Upperbounds Upper bounds per loop level
    /// @param BodyBuilder Callback to build loop body
    affine::AffineForOp GenerateOnDeviceNestedLoop(
        OpBuilder &builder, Location loc,
        int level, SmallVector<int> Upperbounds,
        StationaryBodyBuilderFn BodyBuilder)
    {
      assert(level > 0 && "Loop nest level must be > 0");
      assert((int)Upperbounds.size() == level && "Upperbounds size must == loop level");

      SmallVector<Value> allIvs;
      AffineForOp outer;
      OpBuilder currentBuilder = builder;

      // Outer loops: created via safe iteration with the default yield.
      for (int i = 0; i < level - 1; ++i)
      {
        auto loop = currentBuilder.create<AffineForOp>(loc, 0, Upperbounds[i], 1);
        if (i == 0)
          outer = loop;
        allIvs.push_back(loop.getInductionVar());
        currentBuilder = OpBuilder::atBlockTerminator(loop.getBody());
      }

      // The innermost loop is constructed using a lambda closure.
      auto innermostLoop = currentBuilder.create<AffineForOp>(
          loc, 0, Upperbounds[level - 1], 1, ValueRange(),
          [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs)
          {
            allIvs.push_back(iv);
            if (BodyBuilder)
            {
              BodyBuilder(b, l, allIvs);
            }
          });

      if (level == 1)
        outer = innermostLoop;

      return outer;
    }

    /// @brief Generate nested loop which is the outer loops out of a systolic tile
    ///  Shared by three stationary dataflow.
    /// @param A            MemRef value representing the activation/input tensor.
    /// @param B            MemRef value representing the weight tensor (stationary).
    /// @param C            MemRef value representing the output/accumulation tensor.
    /// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
    /// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
    affine::AffineForOp OffDeviceNestedLoop(
        OpBuilder &builder, Location loc,
        int level,
        SmallVector<int> Upperbounds,
        SmallVector<int> Steps,
        StationaryBodyBuilderFn InnerMostBodyBuilder)
    {
      assert(level > 0 && "Loop nest level must be > 0");
      assert((int)Upperbounds.size() == level && "Upperbounds size must == loop level");
      assert((int)Steps.size() == level && "Steps size must == loop level");

      SmallVector<Value> allIvs;
      AffineForOp outer;
      OpBuilder currentBuilder = builder;

      // create outermost loop
      for (int i = 0; i < level - 1; ++i)
      {
        auto loop = currentBuilder.create<AffineForOp>(loc, 0, Upperbounds[i], Steps[i]);
        if (i == 0)
          outer = loop;
        allIvs.push_back(loop.getInductionVar());
        currentBuilder = OpBuilder::atBlockTerminator(loop.getBody());
      }

      // Innermost loop: implemented via a lambda closure that avoids duplicate yields.
      auto innermostLoop = currentBuilder.create<AffineForOp>(
          loc, 0, Upperbounds[level - 1], Steps[level - 1], ValueRange(),
          [&](OpBuilder &b, Location l, Value iv, ValueRange iterArgs)
          {
            allIvs.push_back(iv);
            if (InnerMostBodyBuilder)
            {
              InnerMostBodyBuilder(b, l, allIvs);
            }
          });

      if (level == 1)
        outer = innermostLoop;

      return outer;
    }

  }
}
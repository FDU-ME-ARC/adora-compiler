//===------------------ ISGemm.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/Utility/Utility.h"

#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

// #include "tensorop/TensorOp.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;
using namespace ::mlir;

namespace mlir
{
  namespace ADORA
  {

    /// @brief Loop body builder for the innermost tiled GEMM computation
    ///        under Input-Stationary (IS) dataflow.
    ///
    /// This function returns a body-builder lambda of type `ISBodyBuilderFn`,
    /// which is meant to be plugged into an `affine.for` nest.
    /// Within the loop body, affine load/store operations are generated to:
    ///   - Load an activation element from A, indexed by (i, k),
    ///   - Load a weight element from B, indexed by (k, j),
    ///   - Load the current partial sum from C, indexed by (i, j),
    ///   - Perform multiply-accumulate (A * B + C),
    ///   - Store the updated result back into C.
    ///
    /// The induction variables (`ivs`) are expected to come in the order {k, i, j},
    /// corresponding to the outer loop nest over (K, M, N).
    /// For each loop iteration, the builder further expands a `tile_row_size × tile_col_size`
    /// micro-kernel of MAC operations inside the loop body.
    ///
    /// @param A            MemRef value representing the activation/input tensor(stationary).
    /// @param B            MemRef value representing the weight tensor.
    /// @param C            MemRef value representing the output/accumulation tensor.
    /// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
    /// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
    ///
    /// Example pseudocode for one tile iteration:
    /// ```text
    /// for k in K
    ///   for i in M
    ///     for j in N
    ///       for n in tile_row_size
    ///         for t in tile_col_size
    ///           C[i+n, j] += A[i+t, k+n] * B[k+n, j]
    StationaryBodyBuilderFn BodyOfTiledWithInputStationary(
        // OpBuilder builder,
        mlir::ValueRange A,
        mlir::ValueRange B,
        mlir::ValueRange C_in,
        mlir::ValueRange C_out,
        int tile_row_size,
        int tile_col_size,
        int innermostTripCount)
    {
      mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A[0].getType()).getElementType();
      return [=](OpBuilder &builder, Location loc, ValueRange ivs)
      {
        // ivs = {k i j}
        ///////////////////////
        //// create deinterleaver for A
        //////////////////////
        SmallVector<mlir::Value> A_stationaries;
        for (int row = 0; row < tile_row_size; row++)
        {
          int col = 0;
          if (tile_col_size >= 4)
          {
            for (; 4 * col + 4 <= tile_col_size; col++)
            {
              /// generate deinterleaver for B
              SmallVector<AffineExpr, 2> Exprs;
              Exprs.push_back(builder.getAffineDimExpr(0) + row);
              Exprs.push_back(builder.getAffineConstantExpr(4 * col));

              SmallVector<int64_t, 4> shape;
              shape.push_back(4);

              AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
              VectorType newVec = VectorType::get(shape, dtype);

              AffineVectorLoadOp vecLoadA = builder.create<affine::AffineVectorLoadOp>(
                  loc, newVec, A[row * ((tile_col_size + 3) / 4) + col], ivs[0], memIVmap);
              setPingpongAttr(vecLoadA);

              ADORA::DeinterleaverOp deinterleaver = builder.create<ADORA::DeinterleaverOp>(loc, vecLoadA.getResult());

              for (int idx = 0; idx < 4; idx++)
              {
                A_stationaries.push_back(deinterleaver.getResult(idx));
              }
            }
          }
          // last several stationaries
          // for(; col <= tile_col_size%4; col++){
          /// generate deinterleaver for A
          if (tile_col_size % 4 != 0 && tile_col_size % 4 > 1)
          {
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineDimExpr(0));            // last dim's affine expr
            Exprs.push_back(builder.getAffineConstantExpr(4 * col)); // last dim's affine expr

            SmallVector<int64_t, 4> shape;
            shape.push_back(tile_col_size % 4);

            AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
            VectorType newVec = VectorType::get(shape, dtype);

            AffineVectorLoadOp vecLoadA = builder.create<affine::AffineVectorLoadOp>(
                loc, newVec, A[row * ((tile_col_size + 3) / 4) + col], ivs[0], memIVmap);
            setPingpongAttr(vecLoadA);

            ADORA::DeinterleaverOp deinterleaver = builder.create<ADORA::DeinterleaverOp>(loc, vecLoadA.getResult());

            for (int idx = 0; idx < tile_col_size % 4; idx++)
            {
              A_stationaries.push_back(deinterleaver.getResult(idx));
            }
          }
          else if (tile_col_size % 4 == 1)
          {
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineDimExpr(0) + row);      // last dim's affine expr
            Exprs.push_back(builder.getAffineConstantExpr(4 * col)); // last dim's affine expr

            SmallVector<int64_t, 4> shape;
            shape.push_back(tile_col_size % 4);

            AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
            // VectorType newVec = VectorType::get(shape, dtype);

            AffineLoadOp LoadA = builder.create<affine::AffineLoadOp>(
                loc, A[row * ((tile_col_size + 3) / 4) + col], memIVmap, ivs[0]);
            setPingpongAttr(LoadA);

            A_stationaries.push_back(LoadA);
          }
        }

        ///////////////////////
        //// create innermost for loop: last level of temporal map
        ///////////////////////
        affine::AffineForOp inner = builder.create<affine::AffineForOp>(
            // loc, 0, Upperbounds[i], 1, /*iterArgs =*/ ValueRange({}), InnerMostBodyBuilder);
            loc, (int64_t)0, (int64_t)innermostTripCount, (int64_t)1, /*iterArgs =*/ValueRange());
        mlir::Block *innermostBody = inner.getBody();
        builder.setInsertionPointToStart(innermostBody);

        // assert(ivs.size() == 2 && "Expect 2 loop induction variables (i,j,k)");
        assert(B.size() == tile_col_size);
        assert(C_in.size() == tile_row_size && C_in.size() == C_out.size());

        ///////////////////
        /// Generate body of loop
        ///////////////////
        Value i_it = ivs[0];
        Value j_it = inner.getInductionVar();

        SmallVector<SmallVector<Value>> sum_results;
        ///////////////////
        /// Generate B
        ///////////////////
        /**
         * Example: K M N (k i j)
            affine.for %arg4 = 0 to 2 {  /// M
              affine.for %arg5 = 0 to 36 { /// N
                %0 = affine.load %A[%arg4, 0] : memref<?x36xi32> // A
                %1 = affine.load %arg1[0, %arg5] : memref<?x36xi32>
                %2 = arith.muli %0, %1 : i32
                %3 = affine.load %arg2[0, %arg5] : memref<?x36xi32>
                %4 = arith.addi %3, %2 : i32
                affine.store %4, %arg2[0, %arg5] : memref<?x36xi32>
                ...
              }
            }
         */
        for (int i = 0; i < tile_row_size; ++i)
        {
          sum_results.push_back(SmallVector<Value>()); /// initial i row's sum
          for (int k = 0; k < tile_col_size; ++k)
          {
            /// get the rhs of add
            mlir::Value mul_lhs_a = A_stationaries[i * tile_col_size + k];

            /// affine map of B :
            /// %1 = affine.load %arg1[0, %arg5] : memref<?x36xi32>
            /// B[k, j]
            AffineExpr rowExpr_B = builder.getAffineConstantExpr(0);
            AffineExpr colExpr_B = builder.getAffineDimExpr(0);
            AffineMap map_B = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                             {rowExpr_B, colExpr_B}, builder.getContext());
            AffineLoadOp LoadB = builder.create<affine::AffineLoadOp>(
                loc, B[k], map_B, ValueRange{j_it});
            setPingpongAttr(LoadB);

            //// get data type of mul and add
            Value mul, add;
            mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A[0].getType()).getElementType();
            mul = genArithMulOpAccordingToDataType(builder, loc, mul_lhs_a, LoadB)->getResult(0);
            if (k != 0)
            {
              Value add_rhs = sum_results[i][k - 1];
              add = genArithAddOpAccordingToDataType(builder, loc, mul, add_rhs)->getResult(0);
            }
            else
            {
              add = mul;
            }

            sum_results[i].push_back(add);
          }
        }

        ///////////////////
        /// Generate store back of C
        ///////////////////
        for (int i = 0; i < tile_row_size; ++i)
        {
          /// %3 = affine.load %arg2[0, %arg5] : memref<?x36xi32>
          /// C[i, j]
          AffineExpr rowExpr_C = builder.getAffineDimExpr(0);
          AffineExpr colExpr_C = builder.getAffineDimExpr(1);
          AffineMap map_C = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                           {rowExpr_C, colExpr_C}, builder.getContext());
          AffineLoadOp LoadC = builder.create<affine::AffineLoadOp>(
              loc, C_in[i], map_C, ValueRange{i_it, j_it});

          mlir::Value add = genArithAddOpAccordingToDataType(builder, loc, sum_results[i][tile_col_size - 1], LoadC)->getResult(0);
          AffineStoreOp StoreC = builder.create<affine::AffineStoreOp>(
              loc, add, C_out[i], map_C, ValueRange{i_it, j_it});

          setPingpongAttr(LoadC);
          setPingpongAttr(StoreC);
        }

        builder.setInsertionPointAfter(inner);
        builder.create<affine::AffineYieldOp>(loc);
      };
    }

    /// @brief Loop body builder for the innermost tiled GEMM computation
    ///        under Weight-Stationary (WS) dataflow.
    ///
    /// This function returns a body-builder lambda of type `StationaryBodyBuilderFn`,
    /// which is meant to be plugged into an `affine.for` nest.
    /// Within the loop body, affine load/store operations are generated to:
    ///   - Load an activation element from A, indexed by (i, k),
    ///   - Load a weight element from B, indexed by (k, j),
    ///   - Load the current partial sum from C, indexed by (i, j),
    ///   - Perform multiply-accumulate (A * B + C),
    ///   - Store the updated result back into C.
    ///
    /// The induction variables (`ivs`) are expected to come in the order {j, k, i},
    /// corresponding to the outer loop nest over (N, K, M).
    /// For each loop iteration, the builder further expands a `tile_row_size × tile_col_size`
    /// micro-kernel of MAC operations inside the loop body.
    ///
    /// @param A            MemRef value representing the activation/input tensor.
    /// @param B            MemRef value representing the weight tensor (stationary).
    /// @param C            MemRef value representing the output/accumulation tensor.
    /// @param tile_row_size Number of rows in the tile (micro-kernel row dimension).
    /// @param tile_col_size Number of columns in the tile (micro-kernel column dimension).
    ///
    /// Example pseudocode for one tile iteration:
    /// ```text
    ///
    /// for kk in K
    ///   for ii in M
    ///     //// followings are temporal map
    ///     for iii in temporal_count_dim_i
    ///       (for jj in temporal_count_dim_n (might == N))
    ///       for j in temporal_count_dim_n
    ///         //// followings are spatial map
    ///         for i in tile_row_size
    ///           for k in tile_col_size
    ///             C[i, j] += A[i, k] * B[k, j]
    StationaryBodyBuilderFn TileofInputStationary(
        // OpBuilder builder,
        mlir::Value A,
        mlir::Value B,
        mlir::Value C,
        int temporal_count_dim_m,
        int temporal_count_dim_n,
        int tile_row_size,
        int tile_col_size)
    {
      mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A.getType()).getElementType();
      return [=](OpBuilder &builder, Location loc, ValueRange ivs)
      {
        /**
         * Example: K M [MM NN N]
         * affine.for %arg3 = 0 to 9 { /// K
            affine.for %arg4 = 0 to 9 {  /// M
              affine.for %arg5 = 0 to 4 {  /// N
                /// A sub matrix

            }
          }
         */
        mlir::Value vk = ivs[0];
        mlir::Value vi = ivs[1];
        mlir::Value vj = ivs[2];
        ///////////////////////////////////////////////
        ////// generate explicit data transfer: host to device
        ///////////////////////////////////////////////
        unsigned BlockLoadStoreOpId = 0;
        SmallVector<ADORA::DataBlockStoreOp> stores;
        SmallVector<mlir::Value> A_in, B_in, C_in, C_out;

        /////////////////////
        /// transfer A
        /////////////////////
        for (int row = 0; row < tile_row_size; row++)
        {
          int col = 0;
          if (tile_col_size >= 4)
          {
            for (; col < tile_col_size / 4; col++)
            {
              SmallVector<AffineExpr, 2> Exprs;
              Exprs.push_back(builder.getAffineDimExpr(0) + row);     // last dim's affine expr
              Exprs.push_back(builder.getAffineDimExpr(1) + col * 4); // last dim's affine expr

              SmallVector<int64_t, 4> shape;
              shape.push_back(temporal_count_dim_m);
              shape.push_back(4);

              AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
              MemRefType newMemRef = MemRefType::get(shape, dtype);
              // llvm::errs() << "[debug] newMemRef: ";newMemRef.dump();

              ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, A, memIVmap, ValueRange({vi, vk}), newMemRef);

              BlockLoad.setKernelName("GEMMIS");
              BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
              setPingpongAttr(BlockLoad);

              /// has stride
              if (temporal_count_dim_m != 1)
              {
                BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
              }

              A_in.push_back(BlockLoad);
            }
          }

          //// the remaining col%4 or col < 4
          if (tile_col_size % 4 != 0)
          {
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineDimExpr(0) + row);     // last dim's affine expr
            Exprs.push_back(builder.getAffineDimExpr(1) + col * 4); // last dim's affine expr

            SmallVector<int64_t, 4> shape;
            shape.push_back(temporal_count_dim_m);
            shape.push_back(tile_col_size % 4);

            AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
            MemRefType newMemRef = MemRefType::get(shape, dtype);

            ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, A, memIVmap, ValueRange({vi, vk}), newMemRef);

            BlockLoad.setKernelName("GEMMIS");
            BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
            setPingpongAttr(BlockLoad);

            /// has stride
            if (temporal_count_dim_m != 1)
            {
              BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
            }

            A_in.push_back(BlockLoad);
          }
        }
        /////////////////////
        /// End of transfer A
        /////////////////////

        /////////////////////
        /// transfer B
        /////////////////////
        for (int col = 0; col < tile_col_size; col++)
        {
          SmallVector<AffineExpr, 2> Exprs;
          Exprs.push_back(builder.getAffineDimExpr(0) + col); // last dim's affine expr
          Exprs.push_back(builder.getAffineDimExpr(1));       // last dim's affine expr

          SmallVector<int64_t, 4> shape;
          shape.push_back(1);
          shape.push_back(temporal_count_dim_n);

          AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
          MemRefType newMemRef = MemRefType::get(shape, dtype);

          ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, B, memIVmap, ValueRange({vk, vj}), newMemRef);

          BlockLoad.setKernelName("GEMMIS");
          BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
          setPingpongAttr(BlockLoad);

          B_in.push_back(BlockLoad);

          /// has NO stride
        }
        /////////////////////
        /// End of transfer B
        /////////////////////

        /////////////////////
        /// transfer C
        /////////////////////
        for (int row = 0; row < tile_row_size; row++)
        {
          SmallVector<AffineExpr, 2> Exprs;
          Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
          Exprs.push_back(builder.getAffineDimExpr(1));       // last dim's affine expr

          SmallVector<int64_t, 4> shape;
          shape.push_back(temporal_count_dim_m);
          shape.push_back(temporal_count_dim_n);

          AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
          MemRefType newMemRef = MemRefType::get(shape, dtype);

          ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, C, memIVmap, ValueRange({vi, vj}), newMemRef);

          BlockLoad.setKernelName("GEMMIS");
          BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
          setPingpongAttr(BlockLoad);

          C_in.push_back(BlockLoad);

          /// has stride
          if (temporal_count_dim_m != 1)
          {
            BlockLoad.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
          }

          ///////////
          /// generate the store back
          ///////////
          ADORA::LocalMemAllocOp alloc = builder.create<ADORA::LocalMemAllocOp>(loc, newMemRef);
          alloc.setKernelName("GEMMIS");
          alloc.setId(std::to_string(BlockLoadStoreOpId));
          C_out.push_back(alloc);
          setPingpongAttr(alloc);

          ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>(loc, alloc, C, memIVmap, ValueRange({vi, vj}));

          BlockStore.setKernelName("GEMMIS");
          BlockStore.setId(std::to_string(BlockLoadStoreOpId++));
          setPingpongAttr(BlockStore);

          /// has stride
          if (temporal_count_dim_m != 1)
          {
            BlockStore.setStrides(ArrayRef<int64_t>({tile_row_size, 1}));
          }

          stores.push_back(BlockStore);
        }
        /////////////////////
        /// End of transfer C
        /////////////////////

        /////////////////////
        /// start generate inner loop
        /////////////////////
        AffineForOp loop = GenerateOnDeviceNestedLoop(
            builder, loc,
            /*level*/ 2,
            /*upper bounds*/ {temporal_count_dim_m, temporal_count_dim_n}, //// K -> M
            /*BodyBuilder*/ BodyOfTiledWithInputStationary(A_in, B_in, C_in, C_out, tile_row_size, tile_col_size, temporal_count_dim_n));

        SpecifiedAffineFortoKernel(loop, "GEMMIS");

        affine::AffineYieldOp yield = builder.create<affine::AffineYieldOp>(loc);

        for (auto store : stores)
        {
          store.getOperation()->moveBefore(yield);
        }
      };
    }

    /////
    //// IS Order : K (k) -> M (i) -> N (j)
    ////  K - reduction dim
    ////  M - row of A, row of C
    ////  N - col of B, col of C
    //// when tile size = 4, (M_temporal_tile, N_temporal_tile, M_spatial_tile, K_spatial_tile)
    //// when tile size = 3, (N_temporal_tile, M_spatial_tile, K_spatial_tile), M_temporal_tile == 1
    //// when tile size = 2, (M_spatial_tile, K_spatial_tile), M_temporal_tile == 1, N_temporal_tile = N
    AffineForOp TiledInputStationaryGemm(
        OpBuilder opbuilder,
        ADORATensor::GemmOp op,
        ArrayRef<int64_t> tilesize //(K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
    )
    {
      assert(tilesize.size() == 2 || tilesize.size() == 3 || tilesize.size() == 4);

      llvm::SmallVector<int64_t, 2> ShapeA = get2DShape(op.getA());
      llvm::SmallVector<int64_t, 2> ShapeB = get2DShape(op.getB());
      llvm::SmallVector<int64_t, 2> ShapeC = get2DShape(op.getC());

      //////////////////////////////////////
      /// Get tiled matmul micro-kernel parameter, which is also the tile of B matrix
      //////////////////////////////////////
      int64_t tilerow, tilecol, M_temporal_tile, N_temporal_tile;
      tilerow = tilesize[tilesize.size() - 2]; /// the tile of micro kernel row, also the tile of A's row
      tilecol = tilesize[tilesize.size() - 1]; /// the tile of micro kernel col, also the tile of A's col

      if (tilesize.size() == 3)
      {
        M_temporal_tile = 1;
        N_temporal_tile = tilesize[0];
      }
      else if (tilesize.size() == 4)
      {
        M_temporal_tile = tilesize[0];
        N_temporal_tile = tilesize[1];
      }
      else
      {
        M_temporal_tile = 1;
        N_temporal_tile = ShapeB[1];
      }

      // make sure matmul is legal
      assert(ShapeA.size() == 2 && ShapeB.size() == 2 && ShapeC.size() == 2);
      assert(ShapeA[0] == ShapeC[0] && ShapeA[1] == ShapeB[0] && ShapeB[1] == ShapeC[1]);
      assert(ShapeA[0] % (tilerow * M_temporal_tile) == 0 && ShapeA[1] % tilecol == 0 && ShapeB[1] % N_temporal_tile == 0);

      if (ShapeB[1] != N_temporal_tile)
      {
        assert(M_temporal_tile == 1);
      }

      int64_t M_step, N_step, K_step;

      // K_aftertile = ShapeB[0] / tilecol;
      // M_aftertile = ShapeA[0] / (M_temporal_tile * tilerow);
      // N_aftertile = ShapeB[1] / N_temporal_tile;
      K_step = tilecol;
      M_step = (M_temporal_tile * tilerow);
      N_step = N_temporal_tile;

      //==========================================================
      // Allocate output buffer (type comes from GEMM output, NOT C)
      //==========================================================
      Location loc = op.getLoc();

      auto outTy = op.getO().getType().dyn_cast<MemRefType>();
      assert(outTy && "Expected GemmOp to return memref as output in lowering.");

      opbuilder.setInsertionPointAfter(op.getOperation());
      Value out = opbuilder.create<memref::AllocOp>(loc, outTy);

      //==========================================================
      // Initialize out with C (copy if same shape, else broadcast init)
      //==========================================================
      mlir::Operation *InitializationOp = initOutWithC2DLike(opbuilder, loc, out, op.getC(), ArrayRef<int64_t>({ShapeA[0], ShapeB[1]}));

      //////////////////////////////////////
      /// Generate systolic gemm
      //////////////////////////////////////
      AffineForOp loop;
      loop = OffDeviceNestedLoop(
          opbuilder, op.getLoc(),
          /*level*/ 3,
          /*upper bounds*/ {ShapeB[0], ShapeA[0], ShapeB[1]}, ////  K -> M -> N
          /*steps*/ {K_step, M_step, N_step},                 ////  K -> M -> N
          /*InnerMostBodyBuilder*/ TileofInputStationary(op.getA(), op.getB(), op.getC(), M_temporal_tile, N_temporal_tile, tilerow, tilecol));

      // op.getOperation()->getBlock()->push_back(loop);
      // loop.getOperation()->moveAfter(op);
      loop.getOperation()->moveAfter(InitializationOp);
      // }
      // else{
      //   loop = GenerateTiledNestedLoopWithoutLoopCarry(
      //     builder, op.getLoc(),
      //     /*level*/4,
      //     /*upper bounds*/{N_aftertile, K_aftertile, M_aftertile, tiletemporal}, //// M -> N -> K -> MM
      //     /*InnerMostBodyBuilder*/InnermostBodyOfTiledWithWeightStationary(
      //       op.getA(), op.getB(), op.getC(), tilerow, tilecol
      //     )
      //   );
      // }
      loop.dump();
      SimplifyLoadStoreOpsInRegion(loop.getRegion());

      loop.walk([&](Operation *op)
                { op->setAttr("ADORAGemm", UnitAttr::get(loop.getContext())); });

      op.getO().replaceAllUsesWith(out);

      loop.dump();

      return loop;
    }

  }
}
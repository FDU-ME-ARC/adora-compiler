//===------------------ OSGemm.cpp - ADORATensor Lower process ----------------------===//
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
    /// The induction variables (`ivs`) are expected to come in the order {i, j, k},
    /// corresponding to the outer loop nest over (M, N, K).
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
    /// for i in M
    ///   for i in M
    ///     for j in N
    ///       for n in tile_row_size
    ///         for t in tile_col_size
    ///           C[i+n, j] += A[i+t, k+n] * B[k+n, j]
    StationaryBodyBuilderFn BodyOfTiledWithOutputStationary(
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
        // ivs = {i j k}

        mlir::Value zero = getConstantOpAccordingToDataType(builder, loc, dtype, 0);
        SmallVector<mlir::Value> zeros;
        for (int _ = 0; _ < tile_row_size * tile_col_size; _++)
        {
          zeros.push_back(zero);
        }
        // getRegionIterArgs
        ///////////////////////
        //// create innermost for loop: last level of temporal map
        ///////////////////////
        affine::AffineForOp inner = builder.create<affine::AffineForOp>(
            // loc, 0, Upperbounds[i], 1, /*iterArgs =*/ ValueRange({}), InnerMostBodyBuilder);
            loc, (int64_t)0, (int64_t)innermostTripCount, /*Step*/ (int64_t)1, /*iterArgs =*/ValueRange(zeros));
        mlir::Block *innermostBody = inner.getBody();
        builder.setInsertionPointToStart(innermostBody);

        // assert(ivs.size() == 2 && "Expect 2 loop induction variables (i,j,k)");
        assert(A.size() == tile_row_size && B.size() == tile_col_size);
        assert(C_in.size() == C_out.size());

        ///////////////////
        /// Generate body of loop
        ///////////////////
        Value j_it = ivs[0];
        Value k_it = inner.getInductionVar();

        SmallVector<Value> acc_results;
        /**
         * Example: M N K (i j k)
            affine.for %arg4 = 0 to 2 {  /// N
              affine.for %arg5 = 0 to 36 { /// K
                %0 = affine.load %A[0, %arg5] : memref<1x36xi32> // A
                %1 = affine.load %arg1[%arg5, %arg4] : memref<36x2xi32> //B
                %2 = arith.muli %0, %1 : i32
                %3 = affine.load %arg2[0, 0] : memref<?x36xi32>
                %4 = arith.addi %3, %2 : i32
                affine.store %4, %arg2[0, 0] : memref<?x36xi32>
                ...
              }
            }
         */
        for (int j = 0; j < tile_col_size; ++j)
        {
          for (int i = 0; i < tile_row_size; ++i)
          {
            /// affine map of A :
            /// %0 = affine.load %arg0[0, %arg4] : memref<1x36xi32>
            /// A[i, k]
            AffineExpr rowExpr_A = builder.getAffineConstantExpr(0);
            AffineExpr colExpr_A = builder.getAffineDimExpr(0);
            AffineMap map_A = AffineMap::get(/*dimCount=*/1, /*symbolCount=*/0,
                                             {rowExpr_A, colExpr_A}, builder.getContext());
            AffineLoadOp LoadA = builder.create<affine::AffineLoadOp>(
                loc, A[i], map_A, ValueRange{k_it});
            setPingpongAttr(LoadA);

            /// affine map of B :
            /// %1 = affine.load %arg1[0, %arg5] : memref<?x36xi32>
            /// B[k, j]
            AffineExpr rowExpr_B = builder.getAffineDimExpr(0);
            AffineExpr colExpr_B = builder.getAffineDimExpr(1);
            AffineMap map_B = AffineMap::get(/*dimCount=*/2, /*symbolCount=*/0,
                                             {rowExpr_B, colExpr_B}, builder.getContext());
            AffineLoadOp LoadB = builder.create<affine::AffineLoadOp>(
                loc, B[j], map_B, ValueRange{k_it, j_it});
            setPingpongAttr(LoadB);

            //// get data type of mul and add
            Value mul, add;
            mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A[0].getType()).getElementType();
            mul = genArithMulOpAccordingToDataType(builder, loc, LoadA, LoadB)->getResult(0);

            Value add_rhs = inner.getRegionIterArgs()[j * tile_row_size + i];
            add = genArithAddOpAccordingToDataType(builder, loc, mul, add_rhs)->getResult(0);

            acc_results.push_back(add);
          }
        }
        builder.create<affine::AffineYieldOp>(loc, acc_results);

        builder.setInsertionPointAfter(inner);

        ///////////////////
        /// Generate interleaver for C
        ///////////////////
        SmallVector<mlir::Value> C_stationaries;
        for (int col = 0; col < tile_col_size; col++)
        {
          int row = 0;
          if (tile_row_size >= 4)
          {
            for (; 4 * row + 4 <= tile_row_size; row++)
            {
              /// generate interleaver for C
              SmallVector<int64_t, 4> shape;
              shape.push_back(4);
              shape.push_back(1);

              SmallVector<mlir::Value> ToInterleaver;
              for (int i = 0; i < 4; i++)
              {
                ToInterleaver.push_back(inner.getResult(col * tile_row_size + row * 4 + i));
              }
              ADORA::InterleaverOp interleaver = builder.create<ADORA::InterleaverOp>(loc, ToInterleaver, shape);

              /// generate vector input for C
              SmallVector<AffineExpr, 2> Exprs;
              Exprs.push_back(builder.getAffineConstantExpr(4 * row));
              Exprs.push_back(builder.getAffineDimExpr(0));

              AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
              VectorType newVec = VectorType::get(shape, dtype);

              AffineVectorLoadOp vecLoadC = builder.create<affine::AffineVectorLoadOp>(
                  loc, newVec, C_in[col * ((tile_row_size + 3) / 4) + row], j_it, memIVmap);
              // setPingpongAttr(vecLoadC);

              Value vecadd = genArithAddOpAccordingToDataType(builder, loc, interleaver, vecLoadC)->getResult(0);

              AffineVectorStoreOp vecStoreC = builder.create<affine::AffineVectorStoreOp>(
                  loc, vecadd, C_out[col * ((tile_row_size + 3) / 4) + row], memIVmap, j_it);
              // setPingpongAttr(vecStoreC);
            }
          }
          // last several stationaries
          if (tile_row_size % 4 != 0 && tile_row_size % 4 > 1)
          {
            /// generate interleaver for C
            SmallVector<int64_t, 4> shape;
            shape.push_back(tile_row_size % 4);
            shape.push_back(1);

            SmallVector<mlir::Value> ToInterleaver;
            for (int i = 0; i < tile_row_size % 4; i++)
            {
              ToInterleaver.push_back(inner.getResult(col * tile_row_size + row * 4 + i));
            }
            ADORA::InterleaverOp interleaver = builder.create<ADORA::InterleaverOp>(loc, ToInterleaver, shape);

            /// generate vector input for C
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineConstantExpr(4 * row));
            Exprs.push_back(builder.getAffineDimExpr(0));

            AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
            VectorType newVec = VectorType::get(shape, dtype);

            AffineVectorLoadOp vecLoadC = builder.create<affine::AffineVectorLoadOp>(
                loc, newVec, C_in[col * ((tile_row_size + 3) / 4) + row], j_it, memIVmap);
            // setPingpongAttr(vecLoadC);

            Value vecadd = genArithAddOpAccordingToDataType(builder, loc, interleaver, vecLoadC)->getResult(0);

            AffineVectorStoreOp vecStoreC = builder.create<affine::AffineVectorStoreOp>(
                loc, vecadd, C_out[col * ((tile_row_size + 3) / 4) + row], memIVmap, j_it);
            // setPingpongAttr(vecStoreC);
          }
          else if (tile_row_size % 4 == 1)
          {
            /// generate vector input for C
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineConstantExpr(4 * row)); // last dim's affine expr
            Exprs.push_back(builder.getAffineDimExpr(0));            // last dim's affine expr

            AffineMap memIVmap = AffineMap::get(1, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs

            AffineLoadOp LoadC = builder.create<affine::AffineLoadOp>(
                loc, C_in[col * ((tile_row_size + 3) / 4) + row], memIVmap, j_it);
            // setPingpongAttr(LoadC);

            Value add = genArithAddOpAccordingToDataType(builder, loc, inner.getResult(col * tile_row_size + row * 4), LoadC)->getResult(0);

            AffineStoreOp StoreC = builder.create<affine::AffineStoreOp>(
                loc, add, C_out[col * ((tile_row_size + 3) / 4) + row], memIVmap, j_it);
            // setPingpongAttr(StoreC);
          }
        }

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
    /// for ii in M
    ///   for jj in N
    ///     //// followings are temporal map
    ///     for jjj in temporal_count_dim_n
    ///       (for kk in temporal_count_dim_k (might == K))
    ///       for k in temporal_count_dim_k
    ///         //// followings are spatial map
    ///         for i in tile_row_size
    ///           for j in tile_col_size
    ///             C[i, j] += A[i, k] * B[k, j]
    StationaryBodyBuilderFn TileofOutputStationary(
        // OpBuilder builder,
        mlir::Value A,
        mlir::Value B,
        mlir::Value C,
        int temporal_count_dim_n,
        int temporal_count_dim_k,
        int tile_row_size,
        int tile_col_size)
    {
      mlir::Type dtype = dyn_cast<::mlir::MemRefType>(A.getType()).getElementType();
      return [=](OpBuilder &builder, Location loc, ValueRange ivs)
      {
        /**
         * Example: K M [MM NN N]
         * affine.for %arg3 = 0 to 9 { /// M
            affine.for %arg4 = 0 to 9 {  /// N
              affine.for %arg5 = 0 to 4 {  /// K
                /// A sub matrix

            }
          }
         */
        mlir::Value vi = ivs[0];
        mlir::Value vj = ivs[1];
        mlir::Value vk = ivs[2];
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
          SmallVector<AffineExpr, 2> Exprs;
          Exprs.push_back(builder.getAffineDimExpr(0) + row); // last dim's affine expr
          Exprs.push_back(builder.getAffineDimExpr(1));       // last dim's affine expr

          SmallVector<int64_t, 4> shape;
          shape.push_back(1);
          shape.push_back(temporal_count_dim_k);

          AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
          MemRefType newMemRef = MemRefType::get(shape, dtype);

          ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, A, memIVmap, ValueRange({vi, vk}), newMemRef);

          BlockLoad.setKernelName("GEMMOS");
          BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
          setPingpongAttr(BlockLoad);

          A_in.push_back(BlockLoad);
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
          Exprs.push_back(builder.getAffineDimExpr(0));       // last dim's affine expr
          Exprs.push_back(builder.getAffineDimExpr(1) + col); // last dim's affine expr

          SmallVector<int64_t, 4> shape;
          shape.push_back(temporal_count_dim_k);
          shape.push_back(temporal_count_dim_n);

          AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
          MemRefType newMemRef = MemRefType::get(shape, dtype);

          ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, B, memIVmap, ValueRange({vk, vj}), newMemRef);

          BlockLoad.setKernelName("GEMMOS");
          BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
          setPingpongAttr(BlockLoad);

          if (temporal_count_dim_n != 1)
          {
            BlockLoad.setStrides(ArrayRef<int64_t>({1, tile_col_size}));
          }

          B_in.push_back(BlockLoad);
        }
        /////////////////////
        /// End of transfer B
        /////////////////////

        /////////////////////
        /// transfer C
        /////////////////////
        for (int col = 0; col < tile_col_size; col++)
        {
          int row = 0;
          if (tile_row_size >= 4)
          {
            for (; row < tile_row_size / 4; row++)
            {
              SmallVector<AffineExpr, 2> Exprs;
              Exprs.push_back(builder.getAffineDimExpr(0) + row * 4); // last dim's affine expr
              Exprs.push_back(builder.getAffineDimExpr(1) + col);     // last dim's affine expr

              SmallVector<int64_t, 4> shape;
              shape.push_back(4);
              shape.push_back(temporal_count_dim_n);

              AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
              MemRefType newMemRef = MemRefType::get(shape, dtype);
              // llvm::errs() << "[debug] newMemRef: ";newMemRef.dump();

              ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, C, memIVmap, ValueRange({vi, vj}), newMemRef);

              BlockLoad.setKernelName("GEMMOS");
              BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
              // setPingpongAttr(BlockLoad);

              /// has stride
              if (temporal_count_dim_n != 1)
              {
                BlockLoad.setStrides(ArrayRef<int64_t>({1, tile_col_size}));
              }

              C_in.push_back(BlockLoad);

              ///////////
              /// generate the store back
              ///////////
              ADORA::LocalMemAllocOp alloc = builder.create<ADORA::LocalMemAllocOp>(loc, newMemRef);
              alloc.setKernelName("GEMMOS");
              alloc.setId(std::to_string(BlockLoadStoreOpId));
              C_out.push_back(alloc);
              // setPingpongAttr(alloc);

              ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>(loc, alloc, C, memIVmap, ValueRange({vi, vj}));

              BlockStore.setKernelName("GEMMOS");
              BlockStore.setId(std::to_string(BlockLoadStoreOpId++));
              // setPingpongAttr(BlockStore);

              /// has stride
              if (temporal_count_dim_n != 1)
              {
                BlockStore.setStrides(ArrayRef<int64_t>({1, tile_col_size}));
              }

              stores.push_back(BlockStore);
            }
          }

          //// the remaining col%4 or col < 4
          if (tile_row_size % 4 != 0)
          {
            SmallVector<AffineExpr, 2> Exprs;
            Exprs.push_back(builder.getAffineDimExpr(0) + row * 4); // last dim's affine expr
            Exprs.push_back(builder.getAffineDimExpr(1) + col);     // last dim's affine expr

            SmallVector<int64_t, 4> shape;
            shape.push_back(tile_row_size % 4);
            shape.push_back(temporal_count_dim_n);

            AffineMap memIVmap = AffineMap::get(2, /*symbolCount=*/0, Exprs, builder.getContext()); /// stores corresponding AffineMap of above memIVs
            MemRefType newMemRef = MemRefType::get(shape, dtype);

            ADORA::DataBlockLoadOp BlockLoad = builder.create<ADORA::DataBlockLoadOp>(loc, C, memIVmap, ValueRange({vi, vj}), newMemRef);

            BlockLoad.setKernelName("GEMMOS");
            BlockLoad.setId(std::to_string(BlockLoadStoreOpId++));
            setPingpongAttr(BlockLoad);

            /// has stride
            if (temporal_count_dim_n != 1)
            {
              BlockLoad.setStrides(ArrayRef<int64_t>({1, tile_col_size}));
            }

            C_in.push_back(BlockLoad);

            ///////////
            /// generate the store back
            ///////////
            ADORA::LocalMemAllocOp alloc = builder.create<ADORA::LocalMemAllocOp>(loc, newMemRef);
            alloc.setKernelName("GEMMOS");
            alloc.setId(std::to_string(BlockLoadStoreOpId));
            C_out.push_back(alloc);
            // setPingpongAttr(alloc);

            ADORA::DataBlockStoreOp BlockStore = builder.create<ADORA::DataBlockStoreOp>(loc, alloc, C, memIVmap, ValueRange({vi, vj}));

            BlockStore.setKernelName("GEMMOS");
            BlockStore.setId(std::to_string(BlockLoadStoreOpId++));
            // setPingpongAttr(BlockStore);

            /// has stride
            if (temporal_count_dim_n != 1)
            {
              BlockStore.setStrides(ArrayRef<int64_t>({1, tile_col_size}));
            }

            stores.push_back(BlockStore);
          }
        }
        /////////////////////
        /// End of transfer C
        /////////////////////

        /////////////////////
        /// start generate inner loop
        /////////////////////
        AffineForOp loop = GenerateOnDeviceNestedLoop(
            builder, loc,
            /*level*/ 1,
            /*upper bounds*/ {temporal_count_dim_n}, //// K -> M
            /*BodyBuilder*/ BodyOfTiledWithOutputStationary(A_in, B_in, C_in, C_out, tile_row_size, tile_col_size, temporal_count_dim_k));

        SpecifiedAffineFortoKernel(loop, "GEMMOS");

        affine::AffineYieldOp yield = builder.create<affine::AffineYieldOp>(loc);

        for (auto store : stores)
        {
          store.getOperation()->moveBefore(yield);
        }
      };
    }

    /////
    //// OS Order : M (i) -> N (j) -> K (k)

    ////  M - row of A, row of C
    ////  N - col of B, col of C
    ////  K - reduction dim

    //// when tile size = 4, (N_temporal_tile, K_temporal_tile, M_spatial_tile, N_spatial_tile)
    //// when tile size = 3, (K_temporal_tile, M_spatial_tile, N_spatial_tile), N_temporal_tile == 1
    //// when tile size = 2, (M_spatial_tile, N_spatial_tile), N_temporal_tile == 1, K_temporal_tile = N
    AffineForOp TiledOutputStationaryGemm(
        OpBuilder opbuilder,
        ADORATensor::GemmOp op,
        ArrayRef<int64_t> tilesize //(K_temporal_tile, M_temporal_tile, K_spatial_tile, N_spatial_tile)
    )
    {
      assert(tilesize.size() == 2 || tilesize.size() == 3 || tilesize.size() == 4);

      llvm::SmallVector<int64_t, 2> ShapeA = get2DShape(op.getA());
      llvm::SmallVector<int64_t, 2> ShapeB = get2DShape(op.getB());
      llvm::SmallVector<int64_t, 2> ShapeO = get2DShape(op.getO());

      //////////////////////////////////////
      /// Get tiled matmul micro-kernel parameter, which is also the tile of B matrix
      //////////////////////////////////////
      int64_t tilerow, tilecol, N_temporal_tile, K_temporal_tile;
      tilerow = tilesize[tilesize.size() - 2]; /// the tile of micro kernel row, also the tile of C's row
      tilecol = tilesize[tilesize.size() - 1]; /// the tile of micro kernel col, also the tile of C's col

      if (tilesize.size() == 3)
      {
        N_temporal_tile = 1;
        K_temporal_tile = tilesize[0];
      }
      else if (tilesize.size() == 4)
      {
        N_temporal_tile = tilesize[0];
        K_temporal_tile = tilesize[1];
      }
      else
      {
        N_temporal_tile = 1;
        K_temporal_tile = ShapeB[0];
      }

      // make sure matmul is legal
      assert(ShapeA.size() == 2 && ShapeB.size() == 2 && ShapeO.size() == 2);
      assert(ShapeA[0] == ShapeO[0] && ShapeA[1] == ShapeB[0] && ShapeB[1] == ShapeO[1]);
      assert(ShapeO[1] % (tilecol * N_temporal_tile) == 0 && ShapeA[0] % tilerow == 0 && ShapeB[0] % K_temporal_tile == 0);

      if (ShapeB[0] != K_temporal_tile)
      {
        assert(N_temporal_tile == 1);
      }

      int64_t M_step, N_step, K_step;

      // K_aftertile = ShapeB[0] / tilecol;
      // M_aftertile = ShapeA[0] / (M_temporal_tile * tilerow);
      // N_aftertile = ShapeB[1] / N_temporal_tile;
      M_step = tilerow;
      N_step = (N_temporal_tile * tilecol);
      K_step = K_temporal_tile;

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

      op.getOperation()->getBlock()->dump();
      //////////////////////////////////////
      /// Generate systolic gemm
      //////////////////////////////////////
      AffineForOp loop;
      loop = OffDeviceNestedLoop(
          opbuilder, op.getLoc(),
          /*level*/ 3,
          /*upper bounds*/ {ShapeA[0], ShapeB[1], ShapeB[0]}, //// M -> N -> K
          /*steps*/ {M_step, N_step, K_step},                 //// M -> N -> K
          /*InnerMostBodyBuilder*/ TileofOutputStationary(op.getA(), op.getB(), out, N_temporal_tile, K_temporal_tile, tilerow, tilecol));

      // op.getOperation()->getBlock()->push_back(loop);
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
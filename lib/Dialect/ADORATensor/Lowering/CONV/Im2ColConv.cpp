//===------------------ Im2ColConv.cpp - ADORATensor Lowering ------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;
using namespace ::mlir;

namespace mlir
{
    namespace ADORA
    {
        namespace ADORATensor
        {

            // Re-use ConvMetadata helper from GenericDirectConv.cpp or duplicate here
            // Assuming it's available in LowerConv.h headers or similar context

            static StationaryBodyBuilderFn TileofIm2ColConv(
                ConvOp op,
                const ConvMetadata &meta,
                ArrayRef<int64_t> tileSizes // [T_M, T_N, T_K] (GEMM dimensions)
            )
            {
                return [=](OpBuilder &builder, Location loc, ValueRange ivs)
                {
                    // Outer Loops iterate over Gemm Dimensions: M, N, K
                    // M = N_batch * P * Q
                    // N = K_out
                    // K = C * R * S
                    Value iv_m = ivs[0];
                    Value iv_n = ivs[1];
                    Value iv_k = ivs[2];

                    int64_t T_M = tileSizes[0];
                    int64_t T_N = tileSizes[1];
                    int64_t T_K = tileSizes[2];

                    unsigned OpId = 0;

                    // === 1. Virtual Im2Col Loading for Input (A) ===
                    // Input X is [N, C, H, W]. We want to load a tile [T_M, T_K].
                    // AffineMap needed: (m, k) -> (n, c, h, w)
                    // This mapping involves modulo/division which standard AffineMap supports via floordiv/mod.
                    // m -> (n, p, q)
                    // k -> (c, r, s)
                    // h = p*stride + r*dilation

                    // Constants
                    int64_t P = meta.bounds[DimP];
                    int64_t Q = meta.bounds[DimQ];
                    int64_t C = meta.bounds[DimC];
                    int64_t R = meta.bounds[DimR];
                    int64_t S = meta.bounds[DimS];
                    int64_t stride_h = meta.strides[0];
                    int64_t stride_w = meta.strides[1];
                    int64_t dil_h = meta.dilations[0];
                    int64_t dil_w = meta.dilations[1];

                    SmallVector<AffineExpr, 4> im2colExprs;
                    auto d0 = builder.getAffineDimExpr(0); // global_m = iv_m + local_m
                    auto d1 = builder.getAffineDimExpr(1); // global_k = iv_k + local_k

                    // Decompose m -> n, p, q
                    auto q_expr = d0 % Q;
                    auto p_expr = (d0.floorDiv(Q)) % P;
                    auto n_expr = d0.floorDiv(P * Q);

                    // Decompose k -> c, r, s
                    auto s_expr = d1 % S;
                    auto r_expr = (d1.floorDiv(S)) % R;
                    auto c_expr = d1.floorDiv(R * S);

                    // Calculate h, w
                    auto h_expr = p_expr * stride_h + r_expr * dil_h;
                    auto w_expr = q_expr * stride_w + s_expr * dil_w;

                    im2colExprs.push_back(n_expr); // Dim 0 of X
                    im2colExprs.push_back(c_expr); // Dim 1 of X
                    im2colExprs.push_back(h_expr); // Dim 2 of X
                    im2colExprs.push_back(w_expr); // Dim 3 of X

                    // Map: (global_m, global_k) -> indices for X
                    AffineMap mapIm2Col = AffineMap::get(2, 0, im2colExprs, builder.getContext());

                    // We need to pass (iv_m, iv_k) as symbols or base offsets?
                    // DataBlockLoad typically takes ivs. We construct map relative to loaded tile.
                    // Map for DataBlockLoad: [local_m, local_k] + symbols [base_m, base_k] -> X indices
                    // To fit DataBlockLoad signature which takes AffineMap and IVs:
                    // We compose the offset addition into the map.

                    // Adjusted Map for DataBlockLoad:
                    // (local_m, local_k)[base_m, base_k] -> ...
                    auto sym_m = builder.getAffineSymbolExpr(0);
                    auto sym_k = builder.getAffineSymbolExpr(1);
                    auto global_m = d0 + sym_m;
                    auto global_k = d1 + sym_k;

                    // Rebuild exprs with global indices
                    auto q_g = global_m % Q;
                    auto p_g = (global_m.floorDiv(Q)) % P;
                    auto n_g = global_m.floorDiv(P * Q);
                    auto s_g = global_k % S;
                    auto r_g = (global_k.floorDiv(S)) % R;
                    auto c_g = global_k.floorDiv(R * S);
                    auto h_g = p_g * stride_h + r_g * dil_h;
                    auto w_g = q_g * stride_w + s_g * dil_w;

                    AffineMap mapLoadA = AffineMap::get(2, 2, {n_g, c_g, h_g, w_g}, builder.getContext());
                    MemRefType tileTypeA = MemRefType::get({T_M, T_K}, meta.elementType);

                    auto loadA = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getX(), mapLoadA, ValueRange{iv_m, iv_k}, tileTypeA);
                    loadA.setKernelName("ConvIm2Col");
                    loadA.setId(std::to_string(OpId++));
                    setPingpongAttr(loadA);

                    // === 2. Load Weight (B) as Matrix ===
                    // Weight W is [K_out, C, R, S] -> Flatten to [K_out, K_gemm] = [N, K]
                    // Or standard GEMM B is [K, N]?
                    // ADORA GEMM usually A[M,K], B[K,N].
                    // W is [N, C, R, S]. To match B[K, N], we need to transpose or adjust map.
                    // Let's treat W as B[N, K] logically (transposed GEMM) or just map indices.
                    // Let's assume we want standard B[K, N].
                    // W indices: (n, c, r, s). global_n = d1+sym_n, global_k = d0+sym_k (swapped for B[K,N])

                    // Mapping (k, n) -> (n, c, r, s)
                    // k -> (c, r, s) as above. n -> n.
                    auto d0_k = builder.getAffineDimExpr(0); // local_k
                    auto d1_n = builder.getAffineDimExpr(1); // local_n
                    auto g_k = d0_k + builder.getAffineSymbolExpr(0);
                    auto g_n = d1_n + builder.getAffineSymbolExpr(1);

                    auto s_w = g_k % S;
                    auto r_w = (g_k.floorDiv(S)) % R;
                    auto c_w = g_k.floorDiv(R * S);

                    AffineMap mapLoadB = AffineMap::get(2, 2, {g_n, c_w, r_w, s_w}, builder.getContext());
                    MemRefType tileTypeB = MemRefType::get({T_K, T_N}, meta.elementType);

                    auto loadB = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getW(), mapLoadB, ValueRange{iv_k, iv_n}, tileTypeB);
                    loadB.setKernelName("ConvIm2Col");
                    loadB.setId(std::to_string(OpId++));
                    setPingpongAttr(loadB);

                    // === 3. Allocate Output (C) ===
                    // Output Y is [N, K_out, P, Q] -> [N, P, Q, K_out] logical?
                    // Flattened M = N*P*Q.
                    // Map (m, n) -> (n_batch, n_channel, p, q)
                    auto d0_m = builder.getAffineDimExpr(0);
                    auto d1_n_out = builder.getAffineDimExpr(1);
                    auto g_m_out = d0_m + builder.getAffineSymbolExpr(0);
                    auto g_n_out = d1_n_out + builder.getAffineSymbolExpr(1); // output channel

                    auto q_out = g_m_out % Q;
                    auto p_out = (g_m_out.floorDiv(Q)) % P;
                    auto n_batch_out = g_m_out.floorDiv(P * Q);

                    AffineMap mapStoreC = AffineMap::get(2, 2, {n_batch_out, g_n_out, p_out, q_out}, builder.getContext());
                    MemRefType tileTypeC = MemRefType::get({T_M, T_N}, meta.elementType);

                    auto allocC = builder.create<ADORA::LocalMemAllocOp>(loc, tileTypeC);
                    allocC.setKernelName("ConvIm2Col");
                    allocC.setId(std::to_string(OpId));

                    // === 4. Compute (Reuse GEMM Logic) ===
                    // We now have A[T_M, T_K], B[T_K, T_N], C[T_M, T_N].
                    // This is exactly what TileofOutputStationary needs.
                    // But TileofOutputStationary expects vectors of loaded values for ping-pong or specific tiling.
                    // It calls GenerateOnDeviceNestedLoop internally? No, TileofOutputStationary IS the body builder for Outer Loops.
                    // But here we ARE inside the body builder of Outer Loops.
                    // We need to call the "On Device" loop generation.

                    // We can manually generate the inner loops here, similar to GenericDirectConv.
                    // M, N, K order.
                    SmallVector<int64_t, 3> innerBounds = {T_M, T_N, T_K};

                    auto computeBody = [&](OpBuilder &b, Location l, ValueRange iivs)
                    {
                        Value m = iivs[0], n = iivs[1], k = iivs[2];
                        Value valA = b.create<memref::LoadOp>(l, loadA, ValueRange{m, k});
                        Value valB = b.create<memref::LoadOp>(l, loadB, ValueRange{k, n});
                        Value valC = b.create<memref::LoadOp>(l, allocC, ValueRange{m, n});

                        Value mul = genArithMulOpAccordingToDataType(b, l, valA, valB)->getResult(0);
                        Value res = genArithAddOpAccordingToDataType(b, l, valC, mul)->getResult(0);

                        b.create<memref::StoreOp>(l, res, allocC, ValueRange{m, n});
                    };

                    AffineForOp devLoop = GenerateOnDeviceNestedLoop(builder, loc, 3, innerBounds, computeBody);
                    SpecifiedAffineFortoKernel(devLoop, "ConvIm2Col");

                    // === 5. Store Result ===
                    auto storeC = builder.create<ADORA::DataBlockStoreOp>(
                        loc, allocC, op.getY(), mapStoreC, ValueRange{iv_m, iv_n});
                    storeC.setKernelName("ConvIm2Col");
                    storeC.setId(std::to_string(OpId++));
                    setPingpongAttr(storeC);
                };
            }

            mlir::affine::AffineForOp LowerIm2ColConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                Location loc = op.getLoc();
                ConvMetadata meta = getConvMetadata(op);

                // GEMM Dimensions
                int64_t M = meta.bounds[DimN] * meta.bounds[DimP] * meta.bounds[DimQ];
                int64_t N = meta.bounds[DimK]; // Output Channels
                int64_t K = meta.bounds[DimC] * meta.bounds[DimR] * meta.bounds[DimS];

                // Tile Sizes (Default or Config)
                int64_t T_M = 32, T_N = 32, T_K = 32;
                if (config.tileSizes.size() >= 3)
                {
                    T_M = config.tileSizes[0];
                    T_N = config.tileSizes[1];
                    T_K = config.tileSizes[2];
                }

                SmallVector<int64_t> bounds = {M, N, K};
                SmallVector<int64_t> steps = {T_M, T_N, T_K};

                AffineForOp topLoop = OffDeviceNestedLoop(
                    b, loc, 3, bounds, steps,
                    TileofIm2ColConv(op, meta, {T_M, T_N, T_K}));

                SimplifyLoadStoreOpsInRegion(topLoop.getRegion());
                topLoop.walk([&](Operation *inst)
                             { inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); });

                return topLoop;
            }

        } // namespace ADORATensor
    } // namespace ADORA
} // namespace mlir
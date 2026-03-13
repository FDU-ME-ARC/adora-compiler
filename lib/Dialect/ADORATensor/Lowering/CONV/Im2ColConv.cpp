//===------------------ Im2ColConv.cpp - ADORATensor Lowering ------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

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

            static std::pair<AffineForOp, Value> lowerGemmLike(OpBuilder &b, Location loc,
                                                               Value A, Value B, Value C,
                                                               SystolicConfig config);

            static void generateIm2Col(OpBuilder &b, Location loc, Value input, Value colBuffer, const ConvMetadata &meta)
            {
                SmallVector<int64_t> lbs(6, 0);
                SmallVector<int64_t> ubs = {meta.bounds[DimN], meta.bounds[DimC], meta.bounds[DimR], meta.bounds[DimS], meta.bounds[DimP], meta.bounds[DimQ]};
                SmallVector<int64_t> steps(6, 1);

                affine::buildAffineLoopNest(b, loc, lbs, ubs, steps,
                                            [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                            {
                                                Value n = ivs[0], c = ivs[1], r = ivs[2], s = ivs[3], p = ivs[4], q = ivs[5];

                                                AffineExpr p_expr = builder.getAffineDimExpr(0);
                                                AffineExpr r_expr = builder.getAffineDimExpr(1);
                                                AffineExpr h_in_expr = p_expr * meta.strides[0] + r_expr * meta.dilations[0];
                                                Value h_in_coord = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(2, 0, h_in_expr), ValueRange{p, r});

                                                AffineExpr q_expr = builder.getAffineDimExpr(0);
                                                AffineExpr s_expr = builder.getAffineDimExpr(1);
                                                AffineExpr w_in_expr = q_expr * meta.strides[1] + s_expr * meta.dilations[1];
                                                Value w_in_coord = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(2, 0, w_in_expr), ValueRange{q, s});

                                                Value loaded_val = builder.create<affine::AffineLoadOp>(bodyLoc, input, ValueRange{n, c, h_in_coord, w_in_coord});

                                                AffineExpr n_expr_m = builder.getAffineDimExpr(0);
                                                AffineExpr p_expr_m = builder.getAffineDimExpr(1);
                                                AffineExpr q_expr_m = builder.getAffineDimExpr(2);
                                                AffineExpr m_gemm_expr = n_expr_m * (meta.bounds[DimP] * meta.bounds[DimQ]) + p_expr_m * meta.bounds[DimQ] + q_expr_m;
                                                Value m_gemm = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(3, 0, m_gemm_expr), ValueRange{n, p, q});

                                                AffineExpr c_expr_k = builder.getAffineDimExpr(0);
                                                AffineExpr r_expr_k = builder.getAffineDimExpr(1);
                                                AffineExpr s_expr_k = builder.getAffineDimExpr(2);
                                                AffineExpr k_gemm_expr = c_expr_k * (meta.bounds[DimR] * meta.bounds[DimS]) + r_expr_k * meta.bounds[DimS] + s_expr_k;
                                                Value k_gemm = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(3, 0, k_gemm_expr), ValueRange{c, r, s});

                                                builder.create<affine::AffineStoreOp>(bodyLoc, loaded_val, colBuffer, ValueRange{m_gemm, k_gemm});
                                            });
            }

            mlir::affine::AffineForOp LowerIm2ColConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                Location loc = op.getLoc();
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPoint(op);

                ConvMetadata meta = getConvMetadata(op);
                auto elemType = meta.elementType;

                // ======================================================================
                // Step 0: Handle Padding
                // ======================================================================
                Value actualInput = op.getX();
                int64_t pad_h_top = meta.pads.size() > 0 ? meta.pads[0] : 0;
                int64_t pad_w_left = meta.pads.size() > 1 ? meta.pads[1] : 0;
                int64_t pad_h_bottom = meta.pads.size() > 2 ? meta.pads[2] : 0;
                int64_t pad_w_right = meta.pads.size() > 3 ? meta.pads[3] : 0;

                if (pad_h_top > 0 || pad_w_left > 0 || pad_h_bottom > 0 || pad_w_right > 0)
                {
                    auto inputType = mlir::cast<MemRefType>(op.getX().getType());
                    auto inputShape = inputType.getShape();
                    int64_t N = inputShape[0], C = inputShape[1], H_in = inputShape[2], W_in = inputShape[3];
                    int64_t H_padded = H_in + pad_h_top + pad_h_bottom;
                    int64_t W_padded = W_in + pad_w_left + pad_w_right;

                    auto paddedType = MemRefType::get({N, C, H_padded, W_padded}, elemType);
                    Value paddedInput = b.create<memref::AllocOp>(loc, paddedType);

                    Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemType));
                    SmallVector<int64_t> fillLbs(4, 0);
                    SmallVector<int64_t> fillUbs = {N, C, H_padded, W_padded};
                    SmallVector<int64_t> fillSteps(4, 1);

                    affine::buildAffineLoopNest(b, loc, fillLbs, fillUbs, fillSteps,
                                                [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                                {
                                                    builder.create<affine::AffineStoreOp>(bodyLoc, zero, paddedInput, ivs);
                                                });

                    SmallVector<int64_t> copyLbs(4, 0);
                    SmallVector<int64_t> copyUbs = {N, C, H_in, W_in};
                    SmallVector<int64_t> copySteps(4, 1);

                    affine::buildAffineLoopNest(b, loc, copyLbs, copyUbs, copySteps,
                                                [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                                {
                                                    Value n = ivs[0], c = ivs[1], h = ivs[2], w = ivs[3];
                                                    Value val = builder.create<affine::AffineLoadOp>(bodyLoc, op.getX(), ValueRange{n, c, h, w});

                                                    SmallVector<AffineExpr, 4> storeExprs;
                                                    storeExprs.push_back(builder.getAffineDimExpr(0));
                                                    storeExprs.push_back(builder.getAffineDimExpr(1));
                                                    storeExprs.push_back(builder.getAffineDimExpr(2) + pad_h_top);
                                                    storeExprs.push_back(builder.getAffineDimExpr(3) + pad_w_left);
                                                    AffineMap storeMap = AffineMap::get(4, 0, storeExprs, builder.getContext());

                                                    builder.create<affine::AffineStoreOp>(bodyLoc, val, paddedInput, storeMap, ValueRange{n, c, h, w});
                                                });
                    actualInput = paddedInput;
                }

                // ======================================================================
                // 1. Get GEMM dimensions
                // ======================================================================
                int64_t M_gemm = meta.bounds[DimN] * meta.bounds[DimP] * meta.bounds[DimQ];
                int64_t N_gemm = meta.bounds[DimK];
                int64_t K_gemm = meta.bounds[DimC] * meta.bounds[DimR] * meta.bounds[DimS];

                // 2. Allocate and generate the im2col matrix (col_buffer)
                auto colBufferType = MemRefType::get({M_gemm, K_gemm}, elemType);
                Value colBuffer = b.create<memref::AllocOp>(loc, colBufferType);
                generateIm2Col(b, loc, actualInput, colBuffer, meta);

                // ======================================================================
                // 3. Pure Affine weight transpose and layout transformation: [K, C, R, S] -> [K_gemm, N_gemm]
                // ======================================================================
                auto reshapedWeightType = MemRefType::get({K_gemm, N_gemm}, elemType);
                Value reshapedWeight = b.create<memref::AllocOp>(loc, reshapedWeightType);

                SmallVector<int64_t> wLbs(4, 0);
                SmallVector<int64_t> wUbs = {meta.bounds[DimK], meta.bounds[DimC], meta.bounds[DimR], meta.bounds[DimS]};
                SmallVector<int64_t> wSteps(4, 1);

                affine::buildAffineLoopNest(b, loc, wLbs, wUbs, wSteps,
                                            [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                            {
                                                Value k = ivs[0], c = ivs[1], r = ivs[2], s = ivs[3];
                                                Value val = builder.create<affine::AffineLoadOp>(bodyLoc, op.getW(), ValueRange{k, c, r, s});

                                                // k_gemm = c * (R * S) + r * S + s
                                                AffineExpr c_expr = builder.getAffineDimExpr(0);
                                                AffineExpr r_expr = builder.getAffineDimExpr(1);
                                                AffineExpr s_expr = builder.getAffineDimExpr(2);
                                                AffineExpr k_gemm_expr = c_expr * (meta.bounds[DimR] * meta.bounds[DimS]) + r_expr * meta.bounds[DimS] + s_expr;
                                                Value k_gemm = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(3, 0, k_gemm_expr), ValueRange{c, r, s});

                                                // Key idea: map the original outer dimension k into the N_gemm dimension
                                                builder.create<affine::AffineStoreOp>(bodyLoc, val, reshapedWeight, ValueRange{k_gemm, k});
                                            });

                // ======================================================================
                // 4. Prepare C matrix for GEMM (And Initialize Bias)
                // ======================================================================
                auto gemmOutputType = MemRefType::get({M_gemm, N_gemm}, elemType);
                Value gemmC = b.create<memref::AllocOp>(loc, gemmOutputType);

                SmallVector<int64_t> fillLbsC(2, 0);
                SmallVector<int64_t> fillUbsC = {M_gemm, N_gemm};
                SmallVector<int64_t> fillStepsC(2, 1);

                if (op.getB() && !mlir::isa<NoneType>(op.getB().getType()))
                {
                    affine::buildAffineLoopNest(b, loc, fillLbsC, fillUbsC, fillStepsC,
                                                [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                                {
                                                    Value m = ivs[0], n = ivs[1];
                                                    Value b_val = builder.create<affine::AffineLoadOp>(bodyLoc, op.getB(), ValueRange{n});
                                                    builder.create<affine::AffineStoreOp>(bodyLoc, b_val, gemmC, ValueRange{m, n});
                                                });
                }
                else
                {
                    Value zero = b.create<arith::ConstantOp>(loc, b.getZeroAttr(elemType));
                    affine::buildAffineLoopNest(b, loc, fillLbsC, fillUbsC, fillStepsC,
                                                [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                                {
                                                    builder.create<affine::AffineStoreOp>(bodyLoc, zero, gemmC, ivs);
                                                });
                }

                // ======================================================================
                // 5. Lower the GEMM part
                // ======================================================================
                auto gemmRes = lowerGemmLike(b, loc, colBuffer, reshapedWeight, gemmC, config);
                AffineForOp gemmLoops = gemmRes.first;
                Value gemmResult = gemmRes.second;

                // ======================================================================
                // 6. Pure Affine output layout transformation: [M_gemm, N_gemm] -> [N, K, P, Q]
                // ======================================================================
                auto finalOutputType = mlir::cast<MemRefType>(op.getY().getType());
                Value finalResult = b.create<memref::AllocOp>(loc, finalOutputType);

                SmallVector<int64_t> outLbs(4, 0);
                SmallVector<int64_t> outUbs = {meta.bounds[DimN], meta.bounds[DimK], meta.bounds[DimP], meta.bounds[DimQ]};
                SmallVector<int64_t> outSteps(4, 1);

                affine::buildAffineLoopNest(b, loc, outLbs, outUbs, outSteps,
                                            [&](OpBuilder &builder, Location bodyLoc, ValueRange ivs)
                                            {
                                                Value n = ivs[0], k = ivs[1], p = ivs[2], q = ivs[3];

                                                // m_gemm = n * (P * Q) + p * Q + q
                                                AffineExpr n_expr = builder.getAffineDimExpr(0);
                                                AffineExpr p_expr = builder.getAffineDimExpr(1);
                                                AffineExpr q_expr = builder.getAffineDimExpr(2);
                                                AffineExpr m_gemm_expr = n_expr * (meta.bounds[DimP] * meta.bounds[DimQ]) + p_expr * meta.bounds[DimQ] + q_expr;
                                                Value m_gemm = builder.create<affine::AffineApplyOp>(bodyLoc, AffineMap::get(3, 0, m_gemm_expr), ValueRange{n, p, q});

                                                Value val = builder.create<affine::AffineLoadOp>(bodyLoc, gemmResult, ValueRange{m_gemm, k});
                                                builder.create<affine::AffineStoreOp>(bodyLoc, val, finalResult, ValueRange{n, k, p, q});
                                            });

                // 7. Replace the original conv op
                op.replaceAllUsesWith(finalResult);
                op.erase();

                return gemmLoops;
            }

            // ======================================================================
            // Bridging to GEMM backend
            // ======================================================================
            static std::pair<AffineForOp, Value> lowerGemmLike(OpBuilder &b, Location loc,
                                                               Value A, Value B, Value C,
                                                               SystolicConfig config)
            {
                OpBuilder::InsertionGuard guard(b);
                
                // 【修复 1】：删除错误的 setInsertionPointAfter(C.getDefiningOp());
                // 现在生成器会乖乖地跟在 Bias 初始化的循环后面，保证先后顺序正确。

                // 【修复 2】：截断防御机制 (Clamping)，防止 Tile Size 越过实际张量大小
                auto shapeA = mlir::cast<MemRefType>(A.getType()).getShape();
                auto shapeB = mlir::cast<MemRefType>(B.getType()).getShape();
                int64_t actual_M = shapeA[0];
                int64_t actual_K = shapeA[1];
                int64_t actual_N = shapeB[1];

                SmallVector<int64_t> clampedTileSizes;
                for (int64_t ts : config.tileSizes) {
                    clampedTileSizes.push_back(ts);
                }

                // 智能适配：拦截 GEMM 切块配置，通过 std::min 强行将 Tile 截断到实际上限
                if (clampedTileSizes.size() == 3) {
                    // 对于 [Tile_M, Tile_N, Tile_K] 格式
                    clampedTileSizes[0] = std::max<int64_t>(1, std::min(clampedTileSizes[0], actual_M));
                    clampedTileSizes[1] = std::max<int64_t>(1, std::min(clampedTileSizes[1], actual_N));
                    clampedTileSizes[2] = std::max<int64_t>(1, std::min(clampedTileSizes[2], actual_K));
                } else if (clampedTileSizes.size() == 4) {
                    // 对于 [Temporal_M, Temporal_N, Spatial_M, Spatial_N] 格式，直接约束 Spatial 层
                    clampedTileSizes[2] = std::max<int64_t>(1, std::min(clampedTileSizes[2], actual_M));
                    clampedTileSizes[3] = std::max<int64_t>(1, std::min(clampedTileSizes[3], actual_N));
                }

                auto tempGemmOp = b.create<ADORATensor::GemmOp>(loc, C.getType(), A, B, C);

                tempGemmOp->setAttr("Algorithm", b.getStringAttr("GEMM_Standard"));
                tempGemmOp->setAttr("StationaryKind", b.getStringAttr(getDataflowStrategyStrRef(config.dataflow)));
                
                // 将修复后、防越界的 clampedTileSizes 传入 Gemm 算子
                tempGemmOp->setAttr("TileSize", b.getI64ArrayAttr(clampedTileSizes));
                tempGemmOp->setAttr("LoopOrder", b.getI64ArrayAttr(config.loopOrder));

                auto dummyCast = b.create<memref::CastOp>(loc, tempGemmOp.getO().getType(), tempGemmOp.getO());

                AffineForOp newfor;
                switch (config.dataflow)
                {
                case DataflowStrategy::WeightStationary:
                    newfor = TiledWeightStationaryGemm(b, tempGemmOp, clampedTileSizes);
                    break;
                case DataflowStrategy::InputStationary:
                    newfor = TiledInputStationaryGemm(b, tempGemmOp, clampedTileSizes);
                    break;
                case DataflowStrategy::OutputStationary:
                    newfor = TiledOutputStationaryGemm(b, tempGemmOp, clampedTileSizes);
                    break;
                default:
                    tempGemmOp.emitError("Unsupported dataflow strategy for im2col->gemm lowering");
                    return {AffineForOp(), nullptr};
                }

                // Value gemmOutBuffer = dummyCast.getSource();

                // if (gemmOutBuffer == tempGemmOp.getO())
                // {
                //     gemmOutBuffer = C;
                // }

                dummyCast.erase();
                tempGemmOp.erase();

                // return {newfor, gemmOutBuffer};
                return {newfor, C};
            }

            mlir::affine::AffineForOp LowerVirtualIm2ColConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                op.emitError("Lowering for virtual im2col is not yet implemented.");
                return AffineForOp();
            }

        } // namespace ADORATensor
    } // namespace ADORA
} // namespace mlir
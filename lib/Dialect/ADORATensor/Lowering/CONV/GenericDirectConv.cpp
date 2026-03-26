//===------------------ GenericDirectConv.cpp - ADORATensor Lowering ------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
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

            // === 1. Metadata Helper ===
            ConvMetadata getConvMetadata(ConvOp op)
            {
                ConvMetadata meta;
                auto inShape = op.getX().getType().cast<MemRefType>().getShape();
                auto wShape = op.getW().getType().cast<MemRefType>().getShape();
                auto outShape = op.getY().getType().cast<MemRefType>().getShape();

                meta.bounds.assign({outShape[0], outShape[1], outShape[2], outShape[3],
                                    inShape[1], wShape[2], wShape[3]});

                if (auto s = op.getStrides())
                {
                    for (auto val : s.value())
                        meta.strides.push_back(val.cast<IntegerAttr>().getInt());
                }
                else
                {
                    meta.strides = {1, 1};
                }

                if (auto d = op.getDilations())
                {
                    for (auto val : d.value())
                        meta.dilations.push_back(val.cast<IntegerAttr>().getInt());
                }
                else
                {
                    meta.dilations = {1, 1};
                }

                if (auto p = op.getPads())
                {
                    for (auto val : p.value())
                        meta.pads.push_back(val.cast<IntegerAttr>().getInt());
                }
                else
                {
                    meta.pads = {0, 0, 0, 0};
                }

                meta.elementType = op.getX().getType().cast<MemRefType>().getElementType();
                return meta;
            }

            // === 2. Body Builder for Generic Direct Conv ===
            StationaryBodyBuilderFn BuildTiledDirectConvBody(
                ConvOp op,
                const ConvMetadata &meta,
                ArrayRef<int64_t> tileSizes,
                ArrayRef<int> safeLoopOrder,
                Value finalResult)
            {
                return [=](OpBuilder &builder, Location loc, ValueRange ivs) mutable
                {
                    Value zero_idx = builder.create<arith::ConstantIndexOp>(loc, 0);
                    Value iv_n = zero_idx, iv_k = zero_idx, iv_p = zero_idx, iv_q = zero_idx;

                    for (size_t i = 0; i < std::min((size_t)4, ivs.size()); ++i)
                    {
                        switch (safeLoopOrder[i])
                        {
                        case DimN:
                            iv_n = ivs[i];
                            break;
                        case DimK:
                            iv_k = ivs[i];
                            break;
                        case DimP:
                            iv_p = ivs[i];
                            break;
                        case DimQ:
                            iv_q = ivs[i];
                            break;
                        }
                    }

                    int64_t T_N = tileSizes[DimN];
                    int64_t T_K = tileSizes[DimK];
                    int64_t T_P = tileSizes[DimP];
                    int64_t T_Q = tileSizes[DimQ];

                    int64_t T_H_in = (T_P - 1) * meta.strides[0] + (meta.bounds[DimR] - 1) * meta.dilations[0] + 1;
                    int64_t T_W_in = (T_Q - 1) * meta.strides[1] + (meta.bounds[DimS] - 1) * meta.dilations[1] + 1;
                    int64_t T_C = meta.bounds[DimC];

                    unsigned OpId = 0;

                    // ==========================================================
                    // 1. Data Transfer: Load Input (X) Tile
                    // ==========================================================
                    SmallVector<AffineExpr, 4> xExprs;
                    xExprs.push_back(builder.getAffineDimExpr(0));
                    xExprs.push_back(builder.getAffineConstantExpr(0));
                    xExprs.push_back(builder.getAffineDimExpr(1) * meta.strides[0] - meta.pads[0]);
                    xExprs.push_back(builder.getAffineDimExpr(2) * meta.strides[1] - meta.pads[1]);

                    AffineMap mapX = AffineMap::get(3, 0, xExprs, builder.getContext());

                    // The SRAM allocation size is now truncated
                    MemRefType tileTypeX = MemRefType::get({T_N, T_C, T_H_in, T_W_in}, meta.elementType);
                    auto loadX = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getX(), mapX, ValueRange{iv_n, iv_p, iv_q}, tileTypeX);
                    loadX.setKernelName("ConvDirect");
                    loadX.setId(std::to_string(OpId++));
                    setPingpongAttr(loadX);

                    // ==========================================================
                    // 2. Data Transfer: Load Weight (W) Tile
                    // ==========================================================
                    SmallVector<AffineExpr, 4> wExprs;
                    wExprs.push_back(builder.getAffineDimExpr(0));
                    wExprs.push_back(builder.getAffineConstantExpr(0));
                    wExprs.push_back(builder.getAffineConstantExpr(0));
                    wExprs.push_back(builder.getAffineConstantExpr(0));

                    AffineMap mapW = AffineMap::get(1, 0, wExprs, builder.getContext());

                    MemRefType tileTypeW = MemRefType::get({T_K, T_C, meta.bounds[DimR], meta.bounds[DimS]}, meta.elementType);
                    auto loadW = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getW(), mapW, ValueRange{iv_k}, tileTypeW);
                    loadW.setKernelName("ConvDirect");
                    loadW.setId(std::to_string(OpId++));
                    setPingpongAttr(loadW);

                    // ==========================================================
                    // 3. Allocate and Initialize Output (Y) Accumulator (with Bias)
                    // ==========================================================
                    MemRefType tileTypeY = MemRefType::get({T_N, T_K, T_P, T_Q}, meta.elementType);
                    auto allocY = builder.create<ADORA::LocalMemAllocOp>(loc, tileTypeY);
                    allocY.setKernelName("ConvDirect");
                    allocY.setId(std::to_string(OpId));

                    SmallVector<int64_t> fillLbs(4, 0);
                    SmallVector<int64_t> fillUbs = {T_N, T_K, T_P, T_Q};
                    SmallVector<int64_t> fillSteps(4, 1);

                    // Load Bias
                    if (op.getB() && !mlir::isa<NoneType>(op.getB().getType()))
                    {
                        affine::buildAffineLoopNest(builder, loc, fillLbs, fillUbs, fillSteps,
                                                    [&](OpBuilder &b, Location bodyLoc, ValueRange init_ivs)
                                                    {
                                                        Value k_loc = init_ivs[1]; // Local coordinate of the output channel

                                                        // Global Bias index = k_loc + tile base iv_k
                                                        SmallVector<AffineExpr, 1> bExprs;
                                                        bExprs.push_back(b.getAffineDimExpr(0) + b.getAffineSymbolExpr(0));
                                                        AffineMap mapB = AffineMap::get(1, 1, bExprs, b.getContext());

                                                        auto bVal = b.create<affine::AffineLoadOp>(bodyLoc, op.getB(), mapB, ValueRange{k_loc, iv_k});
                                                        setPingpongAttr(bVal);

                                                        auto initStore = b.create<affine::AffineStoreOp>(bodyLoc, bVal, allocY.getResult(), init_ivs);
                                                        setPingpongAttr(initStore);
                                                    });
                    }
                    else
                    {
                        // If there is no Bias, use zero instead.
                        Value zero = builder.create<arith::ConstantOp>(loc, builder.getZeroAttr(meta.elementType));
                        affine::buildAffineLoopNest(builder, loc, fillLbs, fillUbs, fillSteps,
                                                    [&](OpBuilder &b, Location bodyLoc, ValueRange init_ivs)
                                                    {
                                                        auto initStore = b.create<affine::AffineStoreOp>(bodyLoc, zero, allocY.getResult(), init_ivs);
                                                        setPingpongAttr(initStore);
                                                    });
                    }

                    // ==========================================================
                    // 4. On-Device Computation (Inner/Reduction Loops)
                    // ==========================================================
                    SmallVector<int> c_bound = {static_cast<int>(meta.bounds[DimC])};

                    auto cBodyBuilder = [&](OpBuilder &b, Location l, ValueRange c_iv)
                    {
                        Value c = c_iv.empty() ? zero_idx : c_iv[0];

                        SmallVector<int64_t> rsLbs = {0, 0};
                        SmallVector<int64_t> rsUbs = {meta.bounds[DimR], meta.bounds[DimS]};
                        SmallVector<int64_t> rsSteps = {1, 1};

                        affine::buildAffineLoopNest(b, l, rsLbs, rsUbs, rsSteps,
                                                    [&](OpBuilder &b2, Location loc2, ValueRange rs_ivs)
                                                    {
                                                        Value r = rs_ivs.size() > 0 ? rs_ivs[0] : zero_idx;
                                                        Value s = rs_ivs.size() > 1 ? rs_ivs[1] : zero_idx;

                                                        SmallVector<int64_t> tileLbs(4, 0);
                                                        SmallVector<int64_t> tileUbs = {T_N, T_K, T_P, T_Q};
                                                        SmallVector<int64_t> tileSteps(4, 1);

                                                        affine::buildAffineLoopNest(b2, loc2, tileLbs, tileUbs, tileSteps,
                                                                                    [&](OpBuilder &builder, Location microLoc, ValueRange microIVs)
                                                                                    {
                                                                                        Value n = microIVs.size() > 0 ? microIVs[0] : zero_idx;
                                                                                        Value k = microIVs.size() > 1 ? microIVs[1] : zero_idx;
                                                                                        Value p = microIVs.size() > 2 ? microIVs[2] : zero_idx;
                                                                                        Value q = microIVs.size() > 3 ? microIVs[3] : zero_idx;

                                                                                        SmallVector<AffineExpr, 4> xLoadExprs;
                                                                                        xLoadExprs.push_back(builder.getAffineDimExpr(0));                                                                     // n
                                                                                        xLoadExprs.push_back(builder.getAffineDimExpr(1));                                                                     // c
                                                                                        xLoadExprs.push_back(builder.getAffineDimExpr(2) * meta.strides[0] + builder.getAffineDimExpr(3) * meta.dilations[0]); // p, r
                                                                                        xLoadExprs.push_back(builder.getAffineDimExpr(4) * meta.strides[1] + builder.getAffineDimExpr(5) * meta.dilations[1]); // q, s
                                                                                        AffineMap mapXLoad = AffineMap::get(6, 0, xLoadExprs, builder.getContext());

                                                                                        auto loadXOp = builder.create<affine::AffineLoadOp>(microLoc, loadX, mapXLoad, ValueRange{n, c, p, r, q, s});
                                                                                        setPingpongAttr(loadXOp);
                                                                                        Value valX = loadXOp.getResult();

                                                                                        auto loadWOp = builder.create<affine::AffineLoadOp>(microLoc, loadW, ValueRange{k, c, r, s});
                                                                                        setPingpongAttr(loadWOp);
                                                                                        Value valW = loadWOp.getResult();

                                                                                        auto loadYOp = builder.create<affine::AffineLoadOp>(microLoc, allocY, ValueRange{n, k, p, q});
                                                                                        setPingpongAttr(loadYOp);
                                                                                        Value valY = loadYOp.getResult();

                                                                                        Value mul = genArithMulOpAccordingToDataType(builder, microLoc, valX, valW)->getResult(0);
                                                                                        Value res = genArithAddOpAccordingToDataType(builder, microLoc, valY, mul)->getResult(0);

                                                                                        auto storeYOp = builder.create<affine::AffineStoreOp>(microLoc, res, allocY, ValueRange{n, k, p, q});
                                                                                        setPingpongAttr(storeYOp);
                                                                                    });
                                                    });
                        b.create<affine::AffineYieldOp>(l);
                    };

                    AffineForOp computeLoop = GenerateOnDeviceNestedLoop(
                        builder, loc, 1, c_bound, cBodyBuilder);

                    (void)SpecifiedAffineFortoKernel(computeLoop, "ConvDirect");

                    // ==========================================================
                    // 5. Store Result Back
                    // ==========================================================
                    SmallVector<AffineExpr, 4> yExprs;
                    yExprs.push_back(builder.getAffineDimExpr(0)); // d0
                    yExprs.push_back(builder.getAffineDimExpr(1)); // d1
                    yExprs.push_back(builder.getAffineDimExpr(2)); // d2
                    yExprs.push_back(builder.getAffineDimExpr(3)); // d3

                    AffineMap mapY = AffineMap::get(4, 0, yExprs, builder.getContext());

                    auto storeY = builder.create<ADORA::DataBlockStoreOp>(
                        loc, allocY, finalResult, mapY, ValueRange{iv_n, iv_k, iv_p, iv_q});
                    storeY.setKernelName("ConvDirect");
                    storeY.setId(std::to_string(OpId++));
                    setPingpongAttr(storeY);

                    builder.create<affine::AffineYieldOp>(loc);
                };
            }

            // === 3. Main Entry Point ===
            mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                Location loc = op.getLoc();
                OpBuilder::InsertionGuard guard(b);
                b.setInsertionPoint(op);

                ConvMetadata meta = getConvMetadata(op);

                SmallVector<int, 4> safeLoopOrder;
                if (config.loopOrder.size() >= 4)
                {
                    for (int i = 0; i < 4; ++i)
                    {
                        safeLoopOrder.push_back(config.loopOrder[i]);
                    }
                }
                else
                {
                    safeLoopOrder = {DimN, DimK, DimP, DimQ};
                }

                // 【关键修复】：加入了针对 TileSize 的截断钳位 (Clamping) 机制
                SmallVector<int64_t, 7> tileSizes(7, 1);
                if (!config.tileSizes.empty())
                {
                    if (config.tileSizes.size() == 4)
                    {
                        for (size_t i = 0; i < 4; ++i)
                        {
                            int dim = safeLoopOrder[i];
                            // 强行约束：Tile大小绝不能超过张量在此维度的实际物理大小
                            tileSizes[dim] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[dim]));
                        }
                    }
                    else if (config.tileSizes.size() == 7)
                    {
                        for (size_t i = 0; i < 7; ++i)
                        {
                            // 强行约束：Tile大小绝不能超过张量在此维度的实际物理大小
                            tileSizes[i] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[i]));
                        }
                    }
                    else
                    {
                        op.emitError("DirectConv requires a tile_size array of length 4 or 7.");
                        return AffineForOp();
                    }
                }
                else
                {
                    // 默认缺省值也应用安全截断
                    tileSizes[DimN] = std::min<int64_t>(1, meta.bounds[DimN]);
                    tileSizes[DimK] = std::min<int64_t>(4, meta.bounds[DimK]);
                    tileSizes[DimP] = std::min<int64_t>(4, meta.bounds[DimP]);
                    tileSizes[DimQ] = std::min<int64_t>(4, meta.bounds[DimQ]);
                }

                SmallVector<int64_t> outerUpperBounds_i64;
                SmallVector<int64_t> outerSteps_i64;
                for (int i = 0; i < 4; ++i)
                {
                    int dim = safeLoopOrder[i];
                    outerUpperBounds_i64.push_back(meta.bounds[dim]);
                    outerSteps_i64.push_back(tileSizes[dim]);
                }

                SmallVector<int> outerUpperBounds_int;
                SmallVector<int> outerSteps_int;
                for (int64_t bound : outerUpperBounds_i64)
                {
                    outerUpperBounds_int.push_back(static_cast<int>(bound));
                }
                for (int64_t step : outerSteps_i64)
                {
                    outerSteps_int.push_back(static_cast<int>(step));
                }

                auto finalOutputType = mlir::cast<MemRefType>(op.getY().getType());
                Value finalResult = b.create<memref::AllocOp>(loc, finalOutputType);

                AffineForOp topLoop = OffDeviceNestedLoop(
                    b, loc,
                    /*level=*/4,
                    outerUpperBounds_int,
                    outerSteps_int,
                    /*BodyBuilder=*/BuildTiledDirectConvBody(op, meta, tileSizes, safeLoopOrder, finalResult));

                op.replaceAllUsesWith(finalResult);
                // op.erase();

                SimplifyLoadStoreOpsInRegion(topLoop.getRegion());
                topLoop.walk([&](Operation *inst)
                             { inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); });

                return topLoop;
            }

        } // namespace ADORATensor
    } // namespace ADORA
} // namespace mlir
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

#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;
using namespace ::mlir;

namespace mlir
{
namespace ADORA
{
namespace ADORATensor
{

ConvMetadata getConvMetadata(ConvOp op)
{
    ConvMetadata meta;
    auto inShape = mlir::cast<MemRefType>(op.getX().getType()).getShape();
    auto wShape = mlir::cast<MemRefType>(op.getW().getType()).getShape();
    auto outShape = mlir::cast<MemRefType>(op.getY().getType()).getShape();

    meta.bounds.assign({outShape[0], outShape[1], outShape[2], outShape[3],
                        inShape[1], wShape[2], wShape[3]});

    if (auto s = op.getStrides()) {
        for (auto val : s.value()) meta.strides.push_back(mlir::cast<IntegerAttr>(val).getInt());
    } else meta.strides = {1, 1};

    if (auto d = op.getDilations()) {
        for (auto val : d.value()) meta.dilations.push_back(mlir::cast<IntegerAttr>(val).getInt());
    } else meta.dilations = {1, 1};

    if (auto p = op.getPads()) {
        for (auto val : p.value()) meta.pads.push_back(mlir::cast<IntegerAttr>(val).getInt());
    } else meta.pads = {0, 0, 0, 0};

    meta.elementType = mlir::cast<MemRefType>(op.getX().getType()).getElementType();
    return meta;
}

StationaryBodyBuilderFn BuildTiledDirectConvBody(
    ConvOp op,
    Value actualInput,
    ConvMetadata meta,
    ArrayRef<int64_t> tileSizes,
    ArrayRef<int> safeLoopOrder,
    Value finalResult)
{
    return [=](OpBuilder &builder, Location loc, ValueRange ivs) mutable
    {
        llvm::outs() << "\n[DEBUG-DirectConv] Entering Core Tile Generation...\n";

        Value zero_idx = builder.create<arith::ConstantIndexOp>(loc, 0);
        Value iv_n = zero_idx, iv_k = zero_idx, iv_p = zero_idx, iv_q = zero_idx;

        for (size_t i = 0; i < std::min((size_t)4, ivs.size()); ++i) {
            if (safeLoopOrder[i] == DimN) iv_n = ivs[i];
            if (safeLoopOrder[i] == DimK) iv_k = ivs[i];
            if (safeLoopOrder[i] == DimP) iv_p = ivs[i];
            if (safeLoopOrder[i] == DimQ) iv_q = ivs[i];
        }

        int64_t T_N = tileSizes[DimN];
        int64_t T_K = tileSizes[DimK];
        int64_t T_P = tileSizes[DimP];
        int64_t T_Q = tileSizes[DimQ];

        int64_t T_H_in = (T_P - 1) * meta.strides[0] + (meta.bounds[DimR] - 1) * meta.dilations[0] + 1;
        int64_t T_W_in = (T_Q - 1) * meta.strides[1] + (meta.bounds[DimS] - 1) * meta.dilations[1] + 1;

        unsigned OpId = 0;
        MemRefType tileTypeX = MemRefType::get({T_N, meta.bounds[DimC], T_H_in, T_W_in}, meta.elementType);
        MemRefType tileTypeW = MemRefType::get({T_K, meta.bounds[DimC], meta.bounds[DimR], meta.bounds[DimS]}, meta.elementType);
        MemRefType rowPeTypeY = MemRefType::get({1, T_K, 1, T_Q}, meta.elementType);

        // 1. Data Transfer: Load Input (X)
        SmallVector<AffineExpr, 4> xExprs = {
            builder.getAffineDimExpr(0),
            builder.getAffineConstantExpr(0),
            builder.getAffineDimExpr(1) * meta.strides[0] - meta.pads[0],
            builder.getAffineDimExpr(2) * meta.strides[1] - meta.pads[1]
        };
        AffineMap mapX = AffineMap::get(3, 0, xExprs, builder.getContext());
        auto loadX = builder.create<ADORA::DataBlockLoadOp>(loc, actualInput, mapX, ValueRange{iv_n, iv_p, iv_q}, tileTypeX);
        loadX.setId(std::to_string(OpId++));
        loadX.setKernelName("ConvDirect");
        setPingpongAttr(loadX);

        // 2. Data Transfer: Load Weight (W)
        SmallVector<AffineExpr, 4> wExprs = {
            builder.getAffineDimExpr(0),
            builder.getAffineConstantExpr(0),
            builder.getAffineConstantExpr(0),
            builder.getAffineConstantExpr(0)
        };
        AffineMap mapW = AffineMap::get(1, 0, wExprs, builder.getContext());
        auto loadW = builder.create<ADORA::DataBlockLoadOp>(loc, op.getW(), mapW, ValueRange{iv_k}, tileTypeW);
        loadW.setId(std::to_string(OpId++));
        loadW.setKernelName("ConvDirect");
        setPingpongAttr(loadW);

        // 3. 外部静态分配 Local SRAM：满足 Mapper 的 IO 分配约束，只在行维度展开
        SmallVector<Value> yInits, yAllocs;
        SmallVector<std::string> yAllocIDs;
        for (int p = 0; p < T_P; ++p) {
            SmallVector<AffineExpr, 4> yExprs = {
                builder.getAffineDimExpr(0),
                builder.getAffineDimExpr(1),
                builder.getAffineDimExpr(2) + builder.getAffineConstantExpr(p),
                builder.getAffineDimExpr(3)
            };
            AffineMap mapY_block = AffineMap::get(4, 0, yExprs, builder.getContext());

            auto yInit = builder.create<ADORA::DataBlockLoadOp>(loc, finalResult, mapY_block, ValueRange{iv_n, iv_k, iv_p, iv_q}, rowPeTypeY);
            yInit.setId(std::to_string(OpId++));
            yInit.setKernelName("ConvDirect");
            setPingpongAttr(yInit);
            yInits.push_back(yInit.getResult());

            auto yAlloc = builder.create<ADORA::LocalMemAllocOp>(loc, rowPeTypeY);
            std::string allocID = std::to_string(OpId++);
            yAlloc.setId(allocID);
            yAlloc.setKernelName("ConvDirect");
            setPingpongAttr(yAlloc);

            yAllocs.push_back(yAlloc.getResult());
            yAllocIDs.push_back(allocID);
        }

        // 4. Kernel 硬件核心
        SmallVector<int> device_bounds = { static_cast<int>(T_K), static_cast<int>(T_Q) };

        auto deviceBodyBuilder = [&](OpBuilder &b3, Location microLoc, ValueRange dev_ivs) {
            Value k_inner = dev_ivs[0];
            Value q_inner = dev_ivs[1];

            // 4.1 将 yInit 的初始值读入 yAlloc (SRAM 内部流转)
            for (int p = 0; p < T_P; ++p) {
                SmallVector<AffineExpr, 4> yInnerExprs = {
                    b3.getAffineConstantExpr(0), b3.getAffineDimExpr(0),
                    b3.getAffineConstantExpr(0), b3.getAffineDimExpr(1)
                };
                AffineMap mapYInner = AffineMap::get(2, 0, yInnerExprs, b3.getContext());

                auto initVal = b3.create<affine::AffineLoadOp>(microLoc, yInits[p], mapYInner, ValueRange{k_inner, q_inner});
                b3.create<affine::AffineStoreOp>(microLoc, initVal, yAllocs[p], mapYInner, ValueRange{k_inner, q_inner});
            }

            // 4.2 通道 C 的归约循环
            auto cLoop = b3.create<affine::AffineForOp>(microLoc, 0, meta.bounds[DimC], 1);
            OpBuilder cBuilder = OpBuilder::atBlockTerminator(cLoop.getBody());
            Value c_inner = cLoop.getInductionVar();

            // 4.3 核心计算累加：彻底抛弃违规的 alloc，直接对 SRAM 原地累加
            for (int p = 0; p < T_P; ++p) {
                SmallVector<AffineExpr, 4> yInnerExprs = {
                    cBuilder.getAffineConstantExpr(0), cBuilder.getAffineDimExpr(0),
                    cBuilder.getAffineConstantExpr(0), cBuilder.getAffineDimExpr(1)
                };
                AffineMap mapYInner = AffineMap::get(2, 0, yInnerExprs, cBuilder.getContext());

                // 读出现有累加值
                Value acc = cBuilder.create<affine::AffineLoadOp>(microLoc, yAllocs[p], mapYInner, ValueRange{k_inner, q_inner});

                for (int r = 0; r < meta.bounds[DimR]; ++r) {
                    for (int s = 0; s < meta.bounds[DimS]; ++s) {
                        SmallVector<AffineExpr, 4> xInnerExprs = {
                            cBuilder.getAffineConstantExpr(0),
                            cBuilder.getAffineDimExpr(0),
                            cBuilder.getAffineConstantExpr(p * meta.strides[0] + r * meta.dilations[0]),
                            cBuilder.getAffineDimExpr(1) * meta.strides[1] + cBuilder.getAffineConstantExpr(s * meta.dilations[1])
                        };
                        AffineMap mapXInner = AffineMap::get(2, 0, xInnerExprs, cBuilder.getContext());
                        auto valX = cBuilder.create<affine::AffineLoadOp>(microLoc, loadX.getResult(), mapXInner, ValueRange{c_inner, q_inner});

                        SmallVector<AffineExpr, 4> wInnerExprs = {
                            cBuilder.getAffineDimExpr(0),
                            cBuilder.getAffineDimExpr(1),
                            cBuilder.getAffineConstantExpr(r),
                            cBuilder.getAffineConstantExpr(s)
                        };
                        AffineMap mapWInner = AffineMap::get(2, 0, wInnerExprs, cBuilder.getContext());
                        auto valW = cBuilder.create<affine::AffineLoadOp>(microLoc, loadW.getResult(), mapWInner, ValueRange{k_inner, c_inner});

                        Value mul = genArithMulOpAccordingToDataType(cBuilder, microLoc, valX, valW)->getResult(0);
                        acc = genArithAddOpAccordingToDataType(cBuilder, microLoc, acc, mul)->getResult(0);
                    }
                }
                // 存回累加值
                cBuilder.create<affine::AffineStoreOp>(microLoc, acc, yAllocs[p], mapYInner, ValueRange{k_inner, q_inner});
            }

            b3.create<affine::AffineYieldOp>(microLoc);
        };

        AffineForOp computeLoop = GenerateOnDeviceNestedLoop(builder, loc, 2, device_bounds, deviceBodyBuilder);
        (void)SpecifiedAffineFortoKernel(computeLoop, "ConvDirect");

        // 5. BlockStore 回写
        for (int p = 0; p < T_P; ++p) {
            SmallVector<AffineExpr, 4> yExprs = {
                builder.getAffineDimExpr(0),
                builder.getAffineDimExpr(1),
                builder.getAffineDimExpr(2) + builder.getAffineConstantExpr(p),
                builder.getAffineDimExpr(3)
            };
            AffineMap mapY_block = AffineMap::get(4, 0, yExprs, builder.getContext());

            auto storeY = builder.create<ADORA::DataBlockStoreOp>(
                loc, yAllocs[p], finalResult, mapY_block, ValueRange{iv_n, iv_k, iv_p, iv_q});
            storeY.setKernelName("ConvDirect");
            storeY.setId(yAllocIDs[p]);
            setPingpongAttr(storeY);
        }

        builder.create<affine::AffineYieldOp>(loc);
    };
}

mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
{
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);

    ConvMetadata meta = getConvMetadata(op);

    SmallVector<int, 4> safeLoopOrder;
    if (config.loopOrder.size() >= 4) {
        for (int i = 0; i < 4; ++i) safeLoopOrder.push_back(config.loopOrder[i]);
    } else {
        safeLoopOrder = {DimN, DimK, DimP, DimQ};
    }

    SmallVector<int64_t, 7> tileSizes(7, 1);
    if (!config.tileSizes.empty()) {
        if (config.tileSizes.size() == 4) {
            for (size_t i = 0; i < 4; ++i) tileSizes[safeLoopOrder[i]] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[safeLoopOrder[i]]));
        } else if (config.tileSizes.size() == 7) {
            for (size_t i = 0; i < 7; ++i) tileSizes[i] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[i]));
        } else {
            op.emitError("DirectConv requires a tile_size array of length 4 or 7.");
            return AffineForOp();
        }
    } else {
        tileSizes[DimN] = std::min<int64_t>(1, meta.bounds[DimN]);
        tileSizes[DimK] = std::min<int64_t>(4, meta.bounds[DimK]);
        tileSizes[DimP] = std::min<int64_t>(4, meta.bounds[DimP]);
        tileSizes[DimQ] = std::min<int64_t>(4, meta.bounds[DimQ]);
    }

    SmallVector<int> outerUpperBounds_int;
    SmallVector<int> outerSteps_int;
    for (int i = 0; i < 4; ++i) {
        int dim = safeLoopOrder[i];
        outerUpperBounds_int.push_back(static_cast<int>(meta.bounds[dim]));
        outerSteps_int.push_back(static_cast<int>(tileSizes[dim]));
    }

    Value actualInput = op.getX();
    auto finalOutputType = mlir::cast<MemRefType>(op.getY().getType());
    Value finalResult = b.create<memref::AllocOp>(loc, finalOutputType);

    AffineForOp topLoop = OffDeviceNestedLoop(
        b, loc, 4, outerUpperBounds_int, outerSteps_int,
        BuildTiledDirectConvBody(op, actualInput, meta, tileSizes, safeLoopOrder, finalResult));

    op.replaceAllUsesWith(finalResult);

    topLoop.walk([&](Operation *inst) { inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); });

    return topLoop;
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir
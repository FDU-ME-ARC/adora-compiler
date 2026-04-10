//===------------------ GenericDirectConv.cpp - ADORATensor Lowering ------------------===//
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

#include "llvm/Support/raw_ostream.h"
#include <string>

using namespace ::mlir;
using namespace ::mlir::affine;
using namespace ::mlir::ADORA::ADORATensor;

namespace mlir {
namespace ADORA {
namespace ADORATensor {

ConvMetadata getConvMetadata(ConvOp op) {
    ConvMetadata meta;
    auto inShape = mlir::cast<MemRefType>(op.getX().getType()).getShape();
    auto wShape = mlir::cast<MemRefType>(op.getW().getType()).getShape();
    auto outShape = mlir::cast<MemRefType>(op.getY().getType()).getShape();

    meta.bounds.assign({outShape[0], outShape[1], outShape[2], outShape[3], inShape[1], wShape[2], wShape[3]});
    if (auto s = op.getStrides()) for (auto val : s.value()) meta.strides.push_back(mlir::cast<IntegerAttr>(val).getInt());
    else meta.strides = {1, 1};
    if (auto d = op.getDilations()) for (auto val : d.value()) meta.dilations.push_back(mlir::cast<IntegerAttr>(val).getInt());
    else meta.dilations = {1, 1};
    if (auto p = op.getPads()) for (auto val : p.value()) meta.pads.push_back(mlir::cast<IntegerAttr>(val).getInt());
    else meta.pads = {0, 0, 0, 0};
    meta.elementType = mlir::cast<MemRefType>(op.getX().getType()).getElementType();
    return meta;
}

StationaryBodyBuilderFn BuildTiledDirectConvBody(
    ConvOp op, Value actualInput, ConvMetadata meta,
    ArrayRef<int64_t> tileSizes, ArrayRef<int> safeLoopOrder, Value finalResult)
{
    return [=](OpBuilder &builder, Location loc, ValueRange ivs) mutable
    {
        llvm::outs() << "\n[DEBUG-DirectConv] Generating Ultra-Low IO OS-style Conv...\n";

        Value iv_n = ivs[0], iv_k = ivs[1], iv_p = ivs[2], iv_q = ivs[3];
        int64_t T_N = tileSizes[DimN], T_K = tileSizes[DimK], T_P = tileSizes[DimP], T_Q = tileSizes[DimQ];
        int64_t T_H_in = (T_P - 1) * meta.strides[0] + (meta.bounds[DimR] - 1) * meta.dilations[0] + 1;
        int64_t T_W_in = (T_Q - 1) * meta.strides[1] + (meta.bounds[DimS] - 1) * meta.dilations[1] + 1;

        unsigned OpId = 0;
        MemRefType tileTypeX = MemRefType::get({T_N, meta.bounds[DimC], T_H_in, T_W_in}, meta.elementType);
        MemRefType tileTypeW = MemRefType::get({T_K, meta.bounds[DimC], meta.bounds[DimR], meta.bounds[DimS]}, meta.elementType);
        MemRefType tileTypeY = MemRefType::get({T_K, T_Q, T_P}, meta.elementType); // T_P 在最内层支持 Vector

        // 1. DataBlockLoad X & W
        auto mapX = AffineMap::get(3, 0, {builder.getAffineDimExpr(0), builder.getAffineConstantExpr(0), builder.getAffineDimExpr(1) * meta.strides[0] - meta.pads[0], builder.getAffineDimExpr(2) * meta.strides[1] - meta.pads[1]}, builder.getContext());
        auto loadX = builder.create<ADORA::DataBlockLoadOp>(loc, actualInput, mapX, ValueRange{iv_n, iv_p, iv_q}, tileTypeX);
        loadX.setId(std::to_string(OpId++)); loadX.setKernelName("ConvDirect");
        ADORA::setPingpongAttr(loadX);

        auto mapW = AffineMap::get(1, 0, {builder.getAffineDimExpr(0), builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0), builder.getAffineConstantExpr(0)}, builder.getContext());
        auto loadW = builder.create<ADORA::DataBlockLoadOp>(loc, op.getW(), mapW, ValueRange{iv_k}, tileTypeW);
        loadW.setId(std::to_string(OpId++)); loadW.setKernelName("ConvDirect");
        ADORA::setPingpongAttr(loadW);

        // 2. DataBlockLoad Y_init & Alloc
        auto mapY_block = AffineMap::get(4, 0, {builder.getAffineDimExpr(0), builder.getAffineDimExpr(1), builder.getAffineDimExpr(2), builder.getAffineDimExpr(3)}, builder.getContext());
        auto yInit = builder.create<ADORA::DataBlockLoadOp>(loc, finalResult, mapY_block, ValueRange{iv_n, iv_k, iv_p, iv_q}, tileTypeY);
        yInit.setId(std::to_string(OpId++)); yInit.setKernelName("ConvDirect");
        ADORA::setPingpongAttr(yInit);

        auto yAlloc = builder.create<ADORA::LocalMemAllocOp>(loc, tileTypeY);
        std::string allocID = std::to_string(OpId++); yAlloc.setId(allocID); yAlloc.setKernelName("ConvDirect");
        ADORA::setPingpongAttr(yAlloc);

        // 3. Kernel 硬件核心
        SmallVector<int> device_bounds = { static_cast<int>(T_K), static_cast<int>(T_Q) }; 

        auto deviceBodyBuilder = [&](OpBuilder &b3, Location microLoc, ValueRange dev_ivs) {
            Value k_inner = dev_ivs[0], q_inner = dev_ivs[1];
            mlir::Type dtype = meta.elementType;

            // 💡 A: 寄存器赋初值 0.0 (消灭初始访存)
            mlir::Value zero = ADORA::getConstantOpAccordingToDataType(b3, microLoc, dtype, 0.0);
            SmallVector<Value> initial_accs(T_P, zero);

            // 💡 B: 将 C, R, S 彻底下放为 MLIR 的 affine.for，将 IO 数量砍掉 90%！
            auto cLoop = b3.create<affine::AffineForOp>(microLoc, 0, meta.bounds[DimC], 1, initial_accs);
            b3.setInsertionPointToStart(cLoop.getBody());
            Value c_i = cLoop.getInductionVar();

            auto rLoop = b3.create<affine::AffineForOp>(microLoc, 0, meta.bounds[DimR], 1, cLoop.getRegionIterArgs());
            b3.setInsertionPointToStart(rLoop.getBody());
            Value r_i = rLoop.getInductionVar();

            auto sLoop = b3.create<affine::AffineForOp>(microLoc, 0, meta.bounds[DimS], 1, rLoop.getRegionIterArgs());
            b3.setInsertionPointToStart(sLoop.getBody());
            Value s_i = sLoop.getInductionVar();

            SmallVector<Value> current_accs;
            for (auto arg : sLoop.getRegionIterArgs()) current_accs.push_back(arg);

            // W 在 sLoop 内只读取 1 次！
            auto mapWInner = AffineMap::get(4, 0, {b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), b3.getAffineDimExpr(2), b3.getAffineDimExpr(3)}, b3.getContext());
            auto wVal = b3.create<affine::AffineLoadOp>(microLoc, loadW.getResult(), mapWInner, ValueRange{k_inner, c_i, r_i, s_i});
            ADORA::setPingpongAttr(wVal);

            SmallVector<Value> next_accs;
            for (int p = 0; p < T_P; ++p) {
                // X 根据 p 展开，读取 T_P 次，极简 I/O
                auto mapXInner = AffineMap::get(4, 0, {
                    b3.getAffineConstantExpr(0), b3.getAffineDimExpr(0),
                    b3.getAffineConstantExpr(p * meta.strides[0]) + b3.getAffineDimExpr(1) * meta.dilations[0],
                    b3.getAffineDimExpr(2) * meta.strides[1] + b3.getAffineDimExpr(3) * meta.dilations[1]
                }, b3.getContext());

                auto xVal = b3.create<affine::AffineLoadOp>(microLoc, loadX.getResult(), mapXInner, ValueRange{c_i, r_i, q_inner, s_i});
                ADORA::setPingpongAttr(xVal);

                auto mul = ADORA::genArithMulOpAccordingToDataType(b3, microLoc, xVal, wVal)->getResult(0);
                auto add = ADORA::genArithAddOpAccordingToDataType(b3, microLoc, current_accs[p], mul)->getResult(0);
                next_accs.push_back(add);
            }

            b3.create<affine::AffineYieldOp>(microLoc, next_accs);
            b3.setInsertionPointAfter(sLoop);
            b3.create<affine::AffineYieldOp>(microLoc, sLoop.getResults());
            b3.setInsertionPointAfter(rLoop);
            b3.create<affine::AffineYieldOp>(microLoc, rLoop.getResults());
            b3.setInsertionPointAfter(cLoop);

            SmallVector<Value> final_accs;
            for (auto res : cLoop.getResults()) final_accs.push_back(res);

            // 💡 C: 完美复刻 OSGemm 的写回逻辑 (块大小为 4 的 Interleaver + 标量余数)
            int p = 0;
            if (T_P >= 4) {
                for (; p + 4 <= T_P; p += 4) {
                    SmallVector<int64_t, 4> shape = {4};
                    auto vecType = VectorType::get(shape, dtype);
                    auto mapYVec = AffineMap::get(2, 0, {b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), b3.getAffineConstantExpr(p)}, b3.getContext());

                    // VectorLoad 读取初始 C
                    auto vecLoadY = b3.create<affine::AffineVectorLoadOp>(microLoc, vecType, yInit.getResult(), mapYVec, ValueRange{k_inner, q_inner});
                    ADORA::setPingpongAttr(vecLoadY);

                    // Deinterleaver4
                    OperationState deinterState(microLoc, "ADORA.deinterleaver");
                    deinterState.addOperands(vecLoadY.getResult());
                    deinterState.addTypes(SmallVector<Type>(4, dtype));
                    deinterState.addAttribute("deinterleaveNumber", b3.getI32IntegerAttr(4));
                    Operation* deinterOp = b3.create(deinterState);

                    // 4个标量独立相加
                    SmallVector<mlir::Value> to_interleave;
                    for(int i = 0; i < 4; ++i) {
                        Value scalarAdd = ADORA::genArithAddOpAccordingToDataType(
                            b3, microLoc, final_accs[p + i], deinterOp->getResult(i))->getResult(0);
                        to_interleave.push_back(scalarAdd);
                    }

                    // Interleaver4 打包
                    OperationState interState(microLoc, "ADORA.interleaver");
                    interState.addOperands(to_interleave);
                    interState.addTypes(vecType);
                    interState.addAttribute("interleaveNumber", b3.getI32IntegerAttr(4));
                    Operation* interOp = b3.create(interState);

                    auto vecStore = b3.create<affine::AffineVectorStoreOp>(microLoc, interOp->getResult(0), yAlloc.getResult(), mapYVec, ValueRange{k_inner, q_inner});
                    ADORA::setPingpongAttr(vecStore);
                }
            }
            // 处理不足 4 的余数
            for (; p < T_P; p++) {
                auto mapYScalar = AffineMap::get(2, 0, {b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), b3.getAffineConstantExpr(p)}, b3.getContext());
                auto scalarLoadY = b3.create<affine::AffineLoadOp>(microLoc, yInit.getResult(), mapYScalar, ValueRange{k_inner, q_inner});
                ADORA::setPingpongAttr(scalarLoadY);

                Value scalarAdd = ADORA::genArithAddOpAccordingToDataType(b3, microLoc, final_accs[p], scalarLoadY.getResult())->getResult(0);

                auto scalarStore = b3.create<affine::AffineStoreOp>(microLoc, scalarAdd, yAlloc.getResult(), mapYScalar, ValueRange{k_inner, q_inner});
                ADORA::setPingpongAttr(scalarStore);
            }

            b3.create<affine::AffineYieldOp>(microLoc);
        };

        AffineForOp computeLoop = ADORA::GenerateOnDeviceNestedLoop(builder, loc, 2, device_bounds, deviceBodyBuilder);
        (void)ADORA::SpecifiedAffineFortoKernel(computeLoop, "ConvDirect");

        // 4. BlockStore
        auto storeY = builder.create<ADORA::DataBlockStoreOp>(loc, yAlloc.getResult(), finalResult, mapY_block, ValueRange{iv_n, iv_k, iv_p, iv_q});
        storeY.setKernelName("ConvDirect"); storeY.setId(allocID);
        ADORA::setPingpongAttr(storeY);
        
        builder.create<affine::AffineYieldOp>(loc);
    };
}

mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
{
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);

    ConvMetadata meta = getConvMetadata(op);
    SmallVector<int, 4> safeLoopOrder = {DimN, DimK, DimP, DimQ};
    if (config.loopOrder.size() >= 4) for (int i=0; i<4; ++i) safeLoopOrder[i] = config.loopOrder[i];

    SmallVector<int64_t, 7> tileSizes(7, 1);
    if (!config.tileSizes.empty()) {
        for (size_t i=0; i<config.tileSizes.size(); ++i) {
            int dim = config.tileSizes.size() == 4 ? safeLoopOrder[i] : i;
            tileSizes[dim] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[dim]));
        }
    } else {
        tileSizes[DimN]=1; tileSizes[DimK]=4; tileSizes[DimP]=4; tileSizes[DimQ]=4;
    }

    SmallVector<int> outerUBs, outerSteps;
    for (int i=0; i<4; ++i) { outerUBs.push_back(meta.bounds[safeLoopOrder[i]]); outerSteps.push_back(tileSizes[safeLoopOrder[i]]); }

    Value finalResult = b.create<memref::AllocOp>(loc, mlir::cast<MemRefType>(op.getY().getType()));

    AffineForOp topLoop = ADORA::OffDeviceNestedLoop(b, loc, 4, outerUBs, outerSteps, BuildTiledDirectConvBody(op, op.getX(), meta, tileSizes, safeLoopOrder, finalResult));
    op.replaceAllUsesWith(finalResult);

    topLoop.walk([&](Operation *inst) { 
        inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); 
        if (isa<ADORA::DataBlockLoadOp, ADORA::DataBlockStoreOp, ADORA::LocalMemAllocOp, affine::AffineLoadOp, affine::AffineStoreOp, affine::AffineVectorLoadOp, affine::AffineVectorStoreOp>(inst))
            inst->setAttr("Pingpong", UnitAttr::get(topLoop.getContext()));
    });

    return topLoop;
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir
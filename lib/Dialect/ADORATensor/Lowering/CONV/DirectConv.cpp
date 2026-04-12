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

#include "mlir/Pass/PassManager.h"
#include "ADORA/Dialect/ADORA/Transforms/Passes.h"

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
        llvm::outs() << "\n[DEBUG-DirectConv] Generating OS-style Conv with R,S as outer loops...\n";

        // 外层传进来了 6 个维度的循环变量
        Value iv_n = ivs[0], iv_k = ivs[1], iv_p = ivs[2], iv_q = ivs[3], iv_r = ivs[4], iv_s = ivs[5];
        int64_t T_N = tileSizes[DimN], T_K = tileSizes[DimK], T_P = tileSizes[DimP], T_Q = tileSizes[DimQ];

        // 由于 r 和 s 在单词 kernel 调用中是固定的，X 只需要加载 T_P 和 T_Q 的大小，极大地缩小了 SPAD 占用！
        int64_t T_H_in = (T_P - 1) * meta.strides[0] + 1;
        int64_t T_W_in = (T_Q - 1) * meta.strides[1] + 1;

        unsigned OpId = 0;
        SmallVector<ADORA::DataBlockStoreOp> stores;

        // 1. X BlockLoad
        MemRefType tileTypeX = MemRefType::get({T_N, meta.bounds[DimC], T_H_in, T_W_in}, meta.elementType);
        auto mapX = AffineMap::get(5, 0, {
            builder.getAffineDimExpr(0),
            builder.getAffineConstantExpr(0),
            builder.getAffineDimExpr(1) * meta.strides[0] + builder.getAffineDimExpr(3) * meta.dilations[0] - meta.pads[0],
            builder.getAffineDimExpr(2) * meta.strides[1] + builder.getAffineDimExpr(4) * meta.dilations[1] - meta.pads[1]
        }, builder.getContext());
        auto loadX = builder.create<ADORA::DataBlockLoadOp>(loc, actualInput, mapX, ValueRange{iv_n, iv_p, iv_q, iv_r, iv_s}, tileTypeX);
        loadX.setId(std::to_string(OpId++)); loadX.setKernelName("ConvDirect");
        ADORA::setPingpongAttr(loadX);

        // 2. 维持 K 维度的多块切分 (Multiple BlockStore)
        int K_chunk = (T_K >= 4 && T_K % 4 == 0) ? 4 : 1;
        if (T_K == 16) K_chunk = 4;
        int num_K_blocks = T_K / K_chunk;

        SmallVector<Value> wLoads, yInits, yAllocs;

        for (int kb = 0; kb < num_K_blocks; ++kb) {
            // W BlockLoad (每次只加载当前确定的 r 和 s，大小为 1x1)
            MemRefType typeW = MemRefType::get({K_chunk, meta.bounds[DimC], 1, 1}, meta.elementType);
            auto mapW = AffineMap::get(3, 0, {
                builder.getAffineDimExpr(0) + builder.getAffineConstantExpr(kb * K_chunk),
                builder.getAffineConstantExpr(0),
                builder.getAffineDimExpr(1), // iv_r
                builder.getAffineDimExpr(2)  // iv_s
            }, builder.getContext());
            auto loadW = builder.create<ADORA::DataBlockLoadOp>(loc, op.getW(), mapW, ValueRange{iv_k, iv_r, iv_s}, typeW);
            loadW.setId(std::to_string(OpId++)); loadW.setKernelName("ConvDirect");
            ADORA::setPingpongAttr(loadW);
            wLoads.push_back(loadW.getResult());

            // Y BlockLoad & Alloc
            MemRefType typeY = MemRefType::get({K_chunk, T_Q, T_P}, meta.elementType);
            auto mapY = AffineMap::get(4, 0, {
                builder.getAffineDimExpr(0), builder.getAffineDimExpr(1) + builder.getAffineConstantExpr(kb * K_chunk),
                builder.getAffineDimExpr(2), builder.getAffineDimExpr(3)
            }, builder.getContext());

            auto yInit = builder.create<ADORA::DataBlockLoadOp>(loc, finalResult, mapY, ValueRange{iv_n, iv_k, iv_p, iv_q}, typeY);
            yInit.setId(std::to_string(OpId++)); yInit.setKernelName("ConvDirect");

            auto yAlloc = builder.create<ADORA::LocalMemAllocOp>(loc, typeY);
            std::string aId = std::to_string(OpId);
            yAlloc.setId(aId); yAlloc.setKernelName("ConvDirect");

            auto storeY = builder.create<ADORA::DataBlockStoreOp>(loc, yAlloc.getResult(), finalResult, mapY, ValueRange{iv_n, iv_k, iv_p, iv_q});
            storeY.setId(std::to_string(OpId++));
            storeY.setKernelName("ConvDirect");

            // ADORA::setPingpongAttr(yInit);
            yInits.push_back(yInit.getResult());
            yAllocs.push_back(yAlloc.getResult());
            stores.push_back(storeY);
        }

        // 3. Kernel body (现在内部严格只有3层嵌套：k_inner, q_inner, c_inner)
        SmallVector<int> device_bounds = { static_cast<int>(K_chunk), static_cast<int>(T_Q) };
        auto deviceBodyBuilder = [&](OpBuilder &b3, Location microLoc, ValueRange dev_ivs) {
            Value k_inner = dev_ivs[0], q_inner = dev_ivs[1];
            mlir::Type dtype = meta.elementType;

            mlir::Value zero = ADORA::getConstantOpAccordingToDataType(b3, microLoc, dtype, 0.0);
            SmallVector<Value> initial_accs(num_K_blocks * T_P, zero);

            // 唯一一层深度循环：c_inner
            auto cLoop = b3.create<affine::AffineForOp>(microLoc, 0, meta.bounds[DimC], 1, initial_accs);
            b3.setInsertionPointToStart(cLoop.getBody());
            Value c_i = cLoop.getInductionVar();

            SmallVector<Value> next_accs;
            for (auto arg : cLoop.getRegionIterArgs()) next_accs.push_back(arg);

            // 没有 R 和 S 的宏观展开！它们已经是外层循环了。
            SmallVector<Value> xVals(T_P);
            for (int p = 0; p < T_P; ++p) {
                auto mapXInner = AffineMap::get(2, 0, {
                    b3.getAffineConstantExpr(0), b3.getAffineDimExpr(0), // n, c_i
                    b3.getAffineConstantExpr(p * meta.strides[0]),
                    b3.getAffineDimExpr(1) * meta.strides[1] // q_inner
                }, b3.getContext());
                xVals[p] = b3.create<affine::AffineLoadOp>(microLoc, loadX.getResult(), mapXInner, ValueRange{c_i, q_inner});
                ADORA::setPingpongAttr(xVals[p].getDefiningOp());
            }

            for (int kb = 0; kb < num_K_blocks; ++kb) {
                auto mapWInner = AffineMap::get(2, 0, {
                    b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), // k_inner, c_i
                    b3.getAffineConstantExpr(0), b3.getAffineConstantExpr(0)
                }, b3.getContext());
                auto wVal = b3.create<affine::AffineLoadOp>(microLoc, wLoads[kb], mapWInner, ValueRange{k_inner, c_i});
                ADORA::setPingpongAttr(wVal);

                for (int p = 0; p < T_P; ++p) {
                    auto mul = ADORA::genArithMulOpAccordingToDataType(b3, microLoc, xVals[p], wVal)->getResult(0);
                    int acc_idx = kb * T_P + p;
                    next_accs[acc_idx] = ADORA::genArithAddOpAccordingToDataType(b3, microLoc, next_accs[acc_idx], mul)->getResult(0);
                }
            }

            b3.create<affine::AffineYieldOp>(microLoc, next_accs);
            b3.setInsertionPointAfter(cLoop);

            SmallVector<Value> final_accs;
            for (auto res : cLoop.getResults()) final_accs.push_back(res);

            // Interleaver & Store (🚨向 OSGemm 靠拢：不在此处给向量操作施加 Pingpong)
            for (int kb = 0; kb < num_K_blocks; ++kb) {
                int p = 0;
                if (T_P >= 3) {
                    for (; p + 3 <= T_P; p += 3) {
                        SmallVector<int64_t, 3> shape = {3};
                        auto vecType = VectorType::get(shape, dtype);
                        auto mapYVec = AffineMap::get(2, 0, {b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), b3.getAffineConstantExpr(p)}, b3.getContext());

                        auto vecLoadY = b3.create<affine::AffineVectorLoadOp>(microLoc, vecType, yInits[kb], mapYVec, ValueRange{k_inner, q_inner});

                        OperationState deinterState(microLoc, "ADORA.deinterleaver");
                        deinterState.addOperands(vecLoadY.getResult());
                        deinterState.addTypes(SmallVector<Type>(3, dtype));
                        deinterState.addAttribute("deinterleaveNumber", b3.getI32IntegerAttr(3));
                        Operation* deinterOp = b3.create(deinterState);

                        SmallVector<mlir::Value> to_interleave;
                        for(int i = 0; i < 3; ++i) {
                            Value scalarAdd = ADORA::genArithAddOpAccordingToDataType(
                                b3, microLoc, final_accs[kb * T_P + p + i], deinterOp->getResult(i))->getResult(0);
                            to_interleave.push_back(scalarAdd);
                        }

                        OperationState interState(microLoc, "ADORA.interleaver");
                        interState.addOperands(to_interleave);
                        interState.addTypes(vecType);
                        interState.addAttribute("interleaveNumber", b3.getI32IntegerAttr(3));
                        Operation* interOp = b3.create(interState);

                        b3.create<affine::AffineVectorStoreOp>(microLoc, interOp->getResult(0), yAllocs[kb], mapYVec, ValueRange{k_inner, q_inner});
                    }
                }
                for (; p < T_P; p++) {
                    auto mapYScalar = AffineMap::get(2, 0, {b3.getAffineDimExpr(0), b3.getAffineDimExpr(1), b3.getAffineConstantExpr(p)}, b3.getContext());
                    auto scalarLoadY = b3.create<affine::AffineLoadOp>(microLoc, yInits[kb], mapYScalar, ValueRange{k_inner, q_inner});

                    Value scalarAdd = ADORA::genArithAddOpAccordingToDataType(b3, microLoc, final_accs[kb * T_P + p], scalarLoadY.getResult())->getResult(0);
                    auto scalarStore = b3.create<affine::AffineStoreOp>(microLoc, scalarAdd, yAllocs[kb], mapYScalar, ValueRange{k_inner, q_inner});
                }
            }

            b3.create<affine::AffineYieldOp>(microLoc);
        };

        AffineForOp computeLoop = ADORA::GenerateOnDeviceNestedLoop(builder, loc, 2, device_bounds, deviceBodyBuilder);
        (void)ADORA::SpecifiedAffineFortoKernel(computeLoop, "ConvDirect");

        affine::AffineYieldOp yield = builder.create<affine::AffineYieldOp>(loc);
        for (auto store : stores) {
            store.getOperation()->moveBefore(yield);
        }
    };
}

mlir::affine::AffineForOp LowerDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
{
    Location loc = op.getLoc();
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(op);

    ConvMetadata meta = getConvMetadata(op);

    // 🚨 调度外围的 6 层嵌套循环：N, K, P, Q, 加上 R, S
    SmallVector<int, 6> safeLoopOrder = {DimN, DimK, DimP, DimQ, DimR, DimS};

    SmallVector<int64_t, 7> tileSizes(7, 1);
    if (!config.tileSizes.empty()) {
        for (size_t i=0; i<config.tileSizes.size(); ++i) {
            int dim = config.tileSizes.size() == 4 ? safeLoopOrder[i] : i;
            tileSizes[dim] = std::max<int64_t>(1, std::min<int64_t>(config.tileSizes[i], meta.bounds[dim]));
        }
    } else {
        tileSizes[DimN]=1; tileSizes[DimK]=4; tileSizes[DimP]=4; tileSizes[DimQ]=4;
        tileSizes[DimR]=1; tileSizes[DimS]=1;
    }

    SmallVector<int> outerUBs, outerSteps;
    for (int i=0; i<6; ++i) {
        outerUBs.push_back(meta.bounds[safeLoopOrder[i]]);
        outerSteps.push_back(tileSizes[safeLoopOrder[i]]);
    }

    Value finalResult = b.create<memref::AllocOp>(loc, mlir::cast<MemRefType>(op.getY().getType()));

    AffineForOp topLoop = ADORA::OffDeviceNestedLoop(b, loc, 6, outerUBs, outerSteps, BuildTiledDirectConvBody(op, op.getX(), meta, tileSizes, safeLoopOrder, finalResult));
    op.replaceAllUsesWith(finalResult);

    // 🚨 废弃之前的全量赋 Pingpong 逻辑，依赖上方已针对标量单独触发的赋值
    topLoop.walk([&](Operation *inst) {
        inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext()));
    });

    return topLoop;
}

mlir::affine::AffineForOp LowerDirectConvPipeline(OpBuilder &b, ConvOp op, SystolicConfig config) {
    Location loc = op.getLoc();
    ConvMetadata meta = getConvMetadata(op);
    
    Value input = op.getX();
    Value weight = op.getW();
    
    // 1. 创建输出分配 (Alloc)
    auto outType = mlir::cast<MemRefType>(op.getY().getType());
    Value finalResult = b.create<memref::AllocOp>(loc, outType);

    // 2. 依次生成 7 层极其干净的纯量循环：N, K, P, Q, C, R, S
    // 我们不做任何切块，全权信任后续的 AdjustMemoryFootprint Pass
    auto nLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimN]);
    b.setInsertionPointToStart(nLoop.getBody());
    
    auto kLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimK]);
    b.setInsertionPointToStart(kLoop.getBody());
    
    auto pLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimP]);
    b.setInsertionPointToStart(pLoop.getBody());
    
    auto qLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimQ]);
    b.setInsertionPointToStart(qLoop.getBody());
    
    auto cLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimC]);
    b.setInsertionPointToStart(cLoop.getBody());
    
    auto rLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimR]);
    b.setInsertionPointToStart(rLoop.getBody());
    
    auto sLoop = b.create<affine::AffineForOp>(loc, 0, meta.bounds[DimS]);
    b.setInsertionPointToStart(sLoop.getBody());

    {
        // 3. 进入最内层循环，执行标量级乘加运算
        OpBuilder::InsertionGuard guard(b);
        Value iv_n = nLoop.getInductionVar();
        Value iv_k = kLoop.getInductionVar();
        Value iv_p = pLoop.getInductionVar();
        Value iv_q = qLoop.getInductionVar();
        Value iv_c = cLoop.getInductionVar();
        Value iv_r = rLoop.getInductionVar();
        Value iv_s = sLoop.getInductionVar();

        // 构造动态地址映射 (AffineMap) 代替手工算地址
        // h_in = p * stride + r * dilation - pad
        auto mapX = AffineMap::get(6, 0, {
            b.getAffineDimExpr(0), // n
            b.getAffineDimExpr(1), // c
            b.getAffineDimExpr(2) * meta.strides[0] + b.getAffineDimExpr(3) * meta.dilations[0] - meta.pads[0], // h
            b.getAffineDimExpr(4) * meta.strides[1] + b.getAffineDimExpr(5) * meta.dilations[1] - meta.pads[1]  // w
        }, b.getContext());

        auto mapW = AffineMap::get(4, 0, {
            b.getAffineDimExpr(0), b.getAffineDimExpr(1), b.getAffineDimExpr(2), b.getAffineDimExpr(3) // k, c, r, s
        }, b.getContext());

        auto mapY = AffineMap::get(4, 0, {
            b.getAffineDimExpr(0), b.getAffineDimExpr(1), b.getAffineDimExpr(2), b.getAffineDimExpr(3) // n, k, p, q
        }, b.getContext());

        // 纯标量内存读写与计算 (Load-Add-Store)
        // 这里的冗余 Store 会被后续的 MoveLoopCarriedInitailValue 自动优化为寄存器驻留 (OS)
        Value xVal = b.create<affine::AffineLoadOp>(loc, input, mapX, ValueRange{iv_n, iv_c, iv_p, iv_r, iv_q, iv_s});
        Value wVal = b.create<affine::AffineLoadOp>(loc, weight, mapW, ValueRange{iv_k, iv_c, iv_r, iv_s});
        Value yVal = b.create<affine::AffineLoadOp>(loc, finalResult, mapY, ValueRange{iv_n, iv_k, iv_p, iv_q});

        Value mul = ADORA::genArithMulOpAccordingToDataType(b, loc, xVal, wVal)->getResult(0);
        Value res = ADORA::genArithAddOpAccordingToDataType(b, loc, yVal, mul)->getResult(0);

        b.create<affine::AffineStoreOp>(loc, res, finalResult, mapY, ValueRange{iv_n, iv_k, iv_p, iv_q});
    }

    // 4. 重构并打标签 (相当于手动触发了 AffineForToKernelPass)
    b.setInsertionPointAfter(nLoop);
    
    // 调用开放的 C++ 接口，将我们构建的 7 层循环直接封装进 ADORA.kernel
    if (mlir::succeeded(ADORA::SpecifiedAffineFortoKernel(nLoop))) {
        // 给刚刚生成的 Kernel 打上标签，引导后续的 Pass 识别它
        nLoop->getParentOp()->walk([&](ADORA::KernelOp kernel) {
            kernel.setKernelName("ConvDirect_Auto");
            kernel->setAttr("ADORAConv", b.getUnitAttr());
        });
    } else {
        llvm::errs() << "[Warning] LowerDirectConvPipeline: Failed to wrap loops into Kernel.\n";
    }

    // 5. 替换掉原始的高层 ConvOp
    op.replaceAllUsesWith(finalResult);

    return nLoop;
}

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir
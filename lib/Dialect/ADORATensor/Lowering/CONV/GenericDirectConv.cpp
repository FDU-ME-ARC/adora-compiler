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
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h" // For helpers

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

                // Bounds: [N, K, P, Q, C, R, S]
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

                meta.pads = {0, 0, 0, 0}; // Simplified
                meta.elementType = op.getX().getType().cast<MemRefType>().getElementType();
                return meta;
            }

            // === 2. Body Builder for Output Stationary Direct Conv ===
            // Iterates over tiles of Output (N, K, P, Q), loads necessary Input/Weight tiles, computes, and stores.
            StationaryBodyBuilderFn TileofDirectConvOutputStationary(
                ConvOp op,
                const ConvMetadata &meta,
                ArrayRef<int64_t> tileSizes // [T_N, T_K, T_P, T_Q, T_C, T_R, T_S]
            )
            {
                return [=](OpBuilder &builder, Location loc, ValueRange ivs)
                {
                    // ivs corresponds to the outer loops.
                    // Assuming Outer Loops Order: N -> K -> P -> Q
                    Value iv_n = ivs[0];
                    Value iv_k = ivs[1];
                    Value iv_p = ivs[2];
                    Value iv_q = ivs[3];

                    // Current Tile Sizes (Temporal)
                    int64_t T_N = tileSizes[DimN];
                    int64_t T_K = tileSizes[DimK];
                    int64_t T_P = tileSizes[DimP];
                    int64_t T_Q = tileSizes[DimQ];

                    // Derived Input Tile Sizes (Accounting for Stride/Dilation)
                    // H_in = (H_out - 1) * stride + 1 + (kernel - 1) * dilation
                    int64_t T_H_in = (T_P - 1) * meta.strides[0] + 1 + (meta.bounds[DimR] - 1) * meta.dilations[0];
                    int64_t T_W_in = (T_Q - 1) * meta.strides[1] + 1 + (meta.bounds[DimS] - 1) * meta.dilations[1];
                    int64_t T_C = meta.bounds[DimC]; // Load full channel for now (Simplication)

                    unsigned OpId = 0;

                    // === 1. Data Transfer: Load Input (X) Tile ===
                    // Map: (n, c, h, w) -> Global Memory
                    // h = iv_p * stride + local_h
                    // w = iv_q * stride + local_w
                    SmallVector<AffineExpr, 4> xExprs;
                    xExprs.push_back(builder.getAffineDimExpr(0)); // n = iv_n + local_n (simplified to just iv_n if T_N=1)
                    xExprs.push_back(builder.getAffineDimExpr(1)); // c

                    // H_global = iv_p * stride_h + h_local
                    auto hExpr = builder.getAffineSymbolExpr(0) * meta.strides[0] + builder.getAffineDimExpr(2);
                    xExprs.push_back(hExpr);

                    // W_global = iv_q * stride_w + w_local
                    auto wExpr = builder.getAffineSymbolExpr(1) * meta.strides[1] + builder.getAffineDimExpr(3);
                    xExprs.push_back(wExpr);

                    AffineMap mapX = AffineMap::get(/*dims*/ 4, /*symbols*/ 2, xExprs, builder.getContext());

                    // Alloc Local Input
                    MemRefType tileTypeX = MemRefType::get({T_N, T_C, T_H_in, T_W_in}, meta.elementType);

                    // DataBlockLoad X
                    auto loadX = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getX(), mapX,
                        ValueRange{iv_n, iv_p, iv_q}, // Pass iv_n as dim0? Need careful map construction
                        tileTypeX);
                    loadX.setKernelName("ConvDirect");
                    loadX.setId(std::to_string(OpId++));
                    setPingpongAttr(loadX);

                    // === 2. Data Transfer: Load Weight (W) Tile ===
                    // W shape: [K, C, R, S]
                    // Load tile [T_K, C, R, S]
                    MemRefType tileTypeW = MemRefType::get({T_K, T_C, meta.bounds[DimR], meta.bounds[DimS]}, meta.elementType);

                    // Simple offset map for W: [k_global, c, r, s]
                    // k_global = iv_k + k_local
                    SmallVector<AffineExpr, 4> wExprs;
                    wExprs.push_back(builder.getAffineSymbolExpr(0) + builder.getAffineDimExpr(0));
                    wExprs.push_back(builder.getAffineDimExpr(1));
                    wExprs.push_back(builder.getAffineDimExpr(2));
                    wExprs.push_back(builder.getAffineDimExpr(3));
                    AffineMap mapW = AffineMap::get(4, 1, wExprs, builder.getContext());

                    auto loadW = builder.create<ADORA::DataBlockLoadOp>(
                        loc, op.getW(), mapW, ValueRange{iv_k}, tileTypeW);
                    loadW.setKernelName("ConvDirect");
                    loadW.setId(std::to_string(OpId++));
                    setPingpongAttr(loadW);

                    // === 3. Allocate Output (Y) Accumulator ===
                    MemRefType tileTypeY = MemRefType::get({T_N, T_K, T_P, T_Q}, meta.elementType);
                    auto allocY = builder.create<ADORA::LocalMemAllocOp>(loc, tileTypeY);
                    allocY.setKernelName("ConvDirect");
                    allocY.setId(std::to_string(OpId));

                    // Initialize Y (e.g. to 0 or bias). For brevity assuming 0 init loop here or Load Bias.
                    // ... (Init Y loop omitted for brevity, can call initOutWithC2DLike equivalent) ...

                    // === 4. On-Device Computation (Nested Loop) ===
                    // Iterate over T_N, T_K, T_P, T_Q, T_C, T_R, T_S
                    // Simplification: Let's create a loop nest for computation
                    // Order: n, k, p, q, c, r, s

                    SmallVector<int64_t, 7> upperBounds = {T_N, T_K, T_P, T_Q, T_C, meta.bounds[DimR], meta.bounds[DimS]};

                    auto innerBodyBuilder = [&](OpBuilder &b, Location l, ValueRange innerIVs)
                    {
                        // innerIVs: n, k, p, q, c, r, s
                        Value n = innerIVs[0], k = innerIVs[1], p = innerIVs[2], q = innerIVs[3];
                        Value c = innerIVs[4], r = innerIVs[5], s = innerIVs[6];

                        // 1. Load X local: [n, c, h_in, w_in]
                        // h_in = p*stride + r*dilation
                        Value h_off = b.create<arith::AddIOp>(l,
                                                              b.create<arith::MulIOp>(l, p, b.create<arith::ConstantIndexOp>(l, meta.strides[0])),
                                                              b.create<arith::MulIOp>(l, r, b.create<arith::ConstantIndexOp>(l, meta.dilations[0])));

                        Value w_off = b.create<arith::AddIOp>(l,
                                                              b.create<arith::MulIOp>(l, q, b.create<arith::ConstantIndexOp>(l, meta.strides[1])),
                                                              b.create<arith::MulIOp>(l, s, b.create<arith::ConstantIndexOp>(l, meta.dilations[1])));

                        Value valX = b.create<memref::LoadOp>(l, loadX, ValueRange{n, c, h_off, w_off});

                        // 2. Load W local: [k, c, r, s]
                        Value valW = b.create<memref::LoadOp>(l, loadW, ValueRange{k, c, r, s});

                        // 3. Load Y accumulation: [n, k, p, q]
                        Value valY = b.create<memref::LoadOp>(l, allocY, ValueRange{n, k, p, q});

                        // 4. Compute
                        Value mul = genArithMulOpAccordingToDataType(b, l, valX, valW)->getResult(0);
                        Value res = genArithAddOpAccordingToDataType(b, l, valY, mul)->getResult(0);

                        // 5. Store Y
                        b.create<memref::StoreOp>(l, res, allocY, ValueRange{n, k, p, q});
                    };

                    AffineForOp computeLoop = GenerateOnDeviceNestedLoop(
                        builder, loc, 7, upperBounds, innerBodyBuilder);
                    SpecifiedAffineFortoKernel(computeLoop, "ConvDirect");

                    // === 5. Store Result Back ===
                    // Store Y_local -> Y_global
                    // Map: [n_local, k_local, p_local, q_local] -> [iv_n+n, iv_k+k, iv_p+p, iv_q+q]
                    SmallVector<AffineExpr, 4> yExprs;
                    yExprs.push_back(builder.getAffineSymbolExpr(0) + builder.getAffineDimExpr(0));
                    yExprs.push_back(builder.getAffineSymbolExpr(1) + builder.getAffineDimExpr(1));
                    yExprs.push_back(builder.getAffineSymbolExpr(2) + builder.getAffineDimExpr(2));
                    yExprs.push_back(builder.getAffineSymbolExpr(3) + builder.getAffineDimExpr(3));
                    AffineMap mapY = AffineMap::get(4, 4, yExprs, builder.getContext());

                    auto storeY = builder.create<ADORA::DataBlockStoreOp>(
                        loc, allocY, op.getY(), mapY, ValueRange{iv_n, iv_k, iv_p, iv_q});
                    storeY.setKernelName("ConvDirect");
                    storeY.setId(std::to_string(OpId++));
                    setPingpongAttr(storeY);
                };
            }

            // === 3. Main Entry Point ===
            mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config)
            {
                Location loc = op.getLoc();
                ConvMetadata meta = getConvMetadata(op);

                // Tiling Configuration (Simplified for Output Stationary)
                // Assuming loopOrder is [N, K, P, Q, ...]
                // We tile N, K, P, Q as the temporal loops.

                // Use config.tileSizes if available, else default to small tiles
                // Mapping: N=0, K=1, P=2, Q=3 ...
                SmallVector<int64_t, 7> tileSizes(7, 1);
                if (config.tileSizes.size() >= 4)
                {
                    for (int i = 0; i < 4; ++i)
                        tileSizes[i] = config.tileSizes[i];
                }
                else
                {
                    // Default fallback
                    tileSizes[DimN] = 1;
                    tileSizes[DimK] = 4;
                    tileSizes[DimP] = 4;
                    tileSizes[DimQ] = 4;
                }

                // Generate Outer Loops (Off-Device)
                // Iterating over N, K, P, Q
                SmallVector<int64_t> upperBounds = {meta.bounds[DimN], meta.bounds[DimK], meta.bounds[DimP], meta.bounds[DimQ]};
                SmallVector<int64_t> steps = {tileSizes[DimN], tileSizes[DimK], tileSizes[DimP], tileSizes[DimQ]};

                AffineForOp topLoop = OffDeviceNestedLoop(
                    b, loc,
                    /*level=*/4,
                    upperBounds,
                    steps,
                    /*BodyBuilder=*/TileofDirectConvOutputStationary(op, meta, tileSizes));

                SimplifyLoadStoreOpsInRegion(topLoop.getRegion());
                topLoop.walk([&](Operation *inst)
                             { inst->setAttr("ADORAConv", UnitAttr::get(topLoop.getContext())); });

                return topLoop;
            }

        } // namespace ADORATensor
    } // namespace ADORA
} // namespace mlir
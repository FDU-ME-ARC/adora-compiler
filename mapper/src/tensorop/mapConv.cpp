//===------------------ mapConv.cpp - ADORATensor Lower process ----------------------===//
/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
#include "ADORA/Dialect/ADORA/Transforms/SimplifyLoadStore.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

#include "tensorop/TensorOp.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"

#include "ir/adg_ir.h"
#include "ir/dfg_ir.h"
#include "mapper/mapper_sa.h"

using namespace ::mlir::ADORA::ADORATensor;
using namespace ::mlir::affine;

namespace mlir
{
    namespace ADORA
    {
        extern void tryToMoveOutBlockAccessOp(affine::AffineForOp forop);

        static void injectDefaultConvAttributesIfNeeded(ADORATensor::ConvOp op, OpBuilder &builder)
        {
            bool modified = false;

            if (!op->hasAttr("algorithm"))
            {
                op->setAttr("algorithm", builder.getStringAttr("Conv_Im2Col"));
                modified = true;
            }
            if (!op->hasAttr("stationary_kind"))
            {
                op->setAttr("stationary_kind", builder.getStringAttr("InputStationary"));
                modified = true;
            }
            if (!op->hasAttr("tile_size"))
            {
                op->setAttr("tile_size", builder.getDenseI64ArrayAttr({128, 16, 4, 9}));
                modified = true;
            }

            if (modified)
            {
                llvm::errs() << "\n[Warning] Missing systolic attributes on ConvOp.\n"
                             << "          Injected fallback defaults: algorithm=Conv_Im2Col, "
                             << "stationary_kind=InputStationary, tileSize=[128,16,4,9].\n"
                             << "          Did you bypass the strategy-decision pass?\n\n";
            }
        }

        bool TensorDataflowGen::visitOp(ADORATensor::ConvOp op)
        {
            op->getContext()->loadDialect<mlir::memref::MemRefDialect>();
            op->getContext()->loadDialect<mlir::arith::ArithDialect>();
            op->getContext()->loadDialect<mlir::affine::AffineDialect>();

            injectDefaultConvAttributesIfNeeded(op, opbuilder);

            // 1. Get Systolic configuration (uniformly parse algorithm, loopOrder, tileSizes, etc.)
            SystolicConfig config = parseSystolicConfig(op);
            AffineForOp newfor;

            // 2. Dispatch to the corresponding Lowering function based on the convolution algorithm
            switch (config.algorithm)
            {
            case ComputeAlgorithm::Conv_Direct:
                // newfor = LowerDirectConv(opbuilder, op, config);
                newfor = LowerDirectConvPipeline(opbuilder, op, config);
                break;

            case ComputeAlgorithm::Conv_Im2Col:
                newfor = LowerVirtualIm2ColConv(opbuilder, op, config);
                break;

            case ComputeAlgorithm::Conv_Winograd:
                // Winograd is not yet fully supported in the mapper
                llvm::errs() << "[Error] Winograd algorithm mapping is not yet supported in mapper.\n";
                return false;

            case ComputeAlgorithm::GEMM_Standard:
                llvm::errs() << "[Error] ConvOp should not use GEMM_Standard algorithm directly.\n";
                return false;

            default:
                llvm::errs() << "[Error] Unknown Conv Algorithm.\n";
                return false;
            }

            // Check if Lowering successfully generated a nested loop
            if (!newfor)
            {
                llvm::errs() << "[Error] Failed to lower ConvOp to AffineForOp.\n";
                return false;
            }

            // 3. Perform loop level simplification and memory access optimization on the generated nested loop
            simplifyLoopLevelsInRegion(newfor.getRegion(), /*donttouchkernel=*/true);

            if (_verbose)
            {
                llvm::errs() << "\n[mapConv] After simplifyLoopLevelsInRegion:\n";
                newfor.dump();
            }

            // Try to hoist BlockLoad to the outer loop
            tryToMoveOutBlockAccessOp(newfor);

            if (_verbose)
            {
                llvm::errs() << "\n[mapConv] After tryToMoveOutBlockAccessOp:\n";
                newfor.dump();
            }

            // Simplify redundant memory reads and writes
            SimplifyBlockAccessOp(newfor.getRegion());

            if (_verbose)
            {
                llvm::errs() << "\n[mapConv] After SimplifyBlockAccessOp:\n";
                newfor.dump();
            }

            // 4. Initialize the CGRA mapper, and execute hardware mapping and Python/C configuration generation
            ADORA_TENSOR_MAPPER *mapper = new ADORA_TENSOR_MAPPER(_adg, _timeout_ms, _max_iters, _objOpt);
            if (failed(MapNestedForOrKernel(mapper, newfor,
                                            _OpNameFile_str))) {
                return false;
            }
            mappers.push_back(mapper);

            ADORA::KernelOp kernel = nullptr;
            newfor.walk([&](ADORA::KernelOp k) { kernel = k; });
            if (kernel) {
                if (pyEmitter)  pyEmitter->GenerateCGRACFGAndEXE(kernel, mapper);
                // if (sdkEmitter) sdkEmitter->GenerateCGRACFGAndEXE(kernel, mapper);
                // if (cEmitter)   cEmitter->GenerateCGRACFGAndEXE(kernel, mapper);
            }

            op->moveBefore(newfor);

            // op.erase();

            return true;
        }

    } // namespace ADORA
} // namespace mlir

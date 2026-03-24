//===----------------------------------------------------------------------===//
//
// This file implements Data flow graph generation for ADORA Tensor
//
//===----------------------------------------------------------------------===//

/// builtin dialect
#include "mlir/Dialect/Affine/IR/AffineOps.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Misc/DFG.h"
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOpLowerToKernel.h"
// #include "PassDetail.h"

// lower tensor op
#include "../../../../mapper/include/tensorop/TensorOp.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerConv.h"
// // DFG
// ADORA::KernelOp* _kernel_toDFG;
// int _variable_config_cnt = 0;

using namespace mlir;
using namespace mlir::affine;
using namespace mlir::ADORA;
using namespace mlir::ADORA::ADORATensor;

#define DEBUG_TYPE "adora-tensor-op-lower"

namespace mlir
{
  namespace ADORA
  {
    namespace ADORATensor
    {

      bool TensorOpCDFGVisitor::visitOp(ADORATensor::GemmOp op)
      {
        SystolicImplInterface SystolicPara(op);
        ArrayRef<int64_t> tilesize = SystolicPara.getTileSize();
        StringRef strategy = SystolicPara.getStationaryKind();
        AffineForOp newfor;
        if (strategy == getDataflowStrategyStrRef(DataflowStrategy::WeightStationary))
        {
          newfor = TiledWeightStationaryGemm(opbuilder, op, tilesize);
        }
        else if (strategy == getDataflowStrategyStrRef(DataflowStrategy::InputStationary))
        {
          newfor = TiledInputStationaryGemm(opbuilder, op, tilesize);
        }
        else if (strategy == getDataflowStrategyStrRef(DataflowStrategy::OutputStationary))
        {
          newfor = TiledOutputStationaryGemm(opbuilder, op, tilesize);
        }

        op.erase();

        return true;
      }

      bool TensorOpCDFGVisitor::visitOp(ADORATensor::ConvOp op)
      {
        // 1. Parse configuration in a unified way
        // This step hides the complex Attribute parsing details
        SystolicConfig config = parseSystolicConfig(op);

        AffineForOp newfor;

        // 2. Dispatch by algorithm
        switch (config.algorithm)
        {
        case ComputeAlgorithm::Conv_Direct:
          // Invoke the Direct Conv generator
          // LowerGenericDirectConv reads config.loopOrder to decide whether
          // to generate OS (P,Q outer) or WS (R,S outer)
          newfor = LowerGenericDirectConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::Conv_Im2Col:
          // Im2Col usually involves Transform then GEMM
          // newfor = LowerIm2ColConv(opbuilder, op, config);
          newfor = LowerVirtualIm2ColConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::Conv_Winograd:
          // Winograd transform -> matrix multiply -> inverse transform
          // newfor = LowerWinogradConv(opbuilder, op, config);
          break;

        case ComputeAlgorithm::GEMM_Standard:
          llvm::errs() << "[Error] ConvOp should not use GEMM_Standard algorithm directly.\n";
          return false;

        default:
          llvm::errs() << "[Error] Unknown Conv Algorithm.\n";
          return false;
        }

        return true;
      }

    }
  }
}

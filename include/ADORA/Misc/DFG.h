//===----------------------------------------------------------------------===//
// For DFG
// Before #include DFG.h, you should include mlir_cdfg.h first.
//===----------------------------------------------------------------------===//
#ifndef ADORA_DFG_GEN_H
#define ADORA_DFG_GEN_H
#include "../../../lib/DFG/inc/mlir_cdfg.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "ADORA/Dialect/ADORA/Utility/Utility.h"
namespace mlir {
namespace ADORA {
#define GeneralOpNameFile std::getenv("GeneralOpNameFile") //// linux env
/* Class define */
class DFGInfo { 
  public: int Num_ALU = 0 , Num_LSU = 0;
};
std::string GenDFGfromAffinewithCMD
    (std::string KernelsDir, std::string kernelFnName, std::string llvmCDFGPass);
DFGInfo GetDFGinfo(std::string DFGPath);
DFGInfo GetDFGinfo(LLVMCDFG* CDFG);
LogicalResult generateCDFGfromKernel(LLVMCDFG* &CDFG, ADORA::KernelOp kernel,
                                     bool verbose = true);

} // namespace ADORA
} // namespace mlir

bool isInteger(const std::string& str);
#endif

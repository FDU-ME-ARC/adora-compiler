//FileName: LowerConv.h
#ifndef ADORA_TENSOR_CONV_OP_LOWER_H
#define ADORA_TENSOR_CONV_OP_LOWER_H

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Builders.h"

/// ADORA dialect
#include "ADORA/Dialect/ADORATensor/IR/ADORATensor.h"
#include "ADORA/Dialect/ADORATensor/Interface/SystolicImplInterface.h"

// Reuse Utils and TypeDefs from LowerGemm.h (e.g. StationaryBodyBuilderFn, GenerateOnDeviceNestedLoop)
#include "ADORA/Dialect/ADORATensor/Lowering/TensorOps/LowerGemm.h"

namespace mlir {
namespace ADORA {
namespace ADORATensor {

// === 1. Common Definitions for Conv Variants ===

// Standard Conv Dimension Indices
enum ConvDim {
  DimN = 0, // Batch
  DimK = 1, // Output Channel
  DimP = 2, // Output Height
  DimQ = 3, // Output Width
  DimC = 4, // Input Channel
  DimR = 5, // Kernel Height
  DimS = 6  // Kernel Width
};

// Common Metadata used by Direct, Im2Col, and Winograd
struct ConvMetadata {
  SmallVector<int64_t> bounds; // Indexed by ConvDim (0..6)
  SmallVector<int64_t> strides;
  SmallVector<int64_t> dilations;
  SmallVector<int64_t> pads;
  Type elementType;
};

// Helper to extract metadata from ConvOp
ConvMetadata getConvMetadata(ConvOp op);

// === 2. Direct Convolution Lowering ===

/// @brief Lower ConvOp to affine loops using Direct Convolution algorithm.
/// Supports arbitrary loop orders defined in SystolicConfig.
mlir::affine::AffineForOp LowerGenericDirectConv(OpBuilder &b, ConvOp op, SystolicConfig config);

// === 3. Future Interfaces (Placeholders) ===

mlir::affine::AffineForOp LowerIm2ColConv(OpBuilder &b, ConvOp op, SystolicConfig config);
// AffineForOp LowerWinogradConv(OpBuilder &b, ConvOp op, SystolicConfig config);

} // namespace ADORATensor
} // namespace ADORA
} // namespace mlir

#endif // ADORA_TENSOR_CONV_OP_LOWER_H
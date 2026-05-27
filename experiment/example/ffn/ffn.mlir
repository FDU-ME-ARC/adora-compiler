// Feed-Forward Network (FFN) kernel — two-layer MLP with ReLU
//
// Architecture:
//   FC1  : Y   = X  * W1   (64x32 → 64x64)
//   ReLU : Y   = max(Y, 0)           [applied in-place before FC2]
//   FC2  : Out = Y  * W2   (64x64 → 64x32)
//
// Both FC layers are exposed as separate adoracc kernels so the
// scheduler can pipeline them with explicit data-block transfers.
//
// Run with adoracc:
//   adoracc.py ffn.mlir --work-dir /tmp/ffn_out -o /tmp/ffn_out/result.mlir

module attributes {} {

  // -----------------------------------------------------------------------
  // FC1 : Y[i,j] = sum_k X[i,k] * W1[k,j]
  //   X  : 64 x 32   (input activations)
  //   W1 : 32 x 64   (weight matrix)
  //   Y  : 64 x 64   (hidden activations)
  // -----------------------------------------------------------------------
  func.func @ffn_fc1(%arg0: memref<64x32xi32>, %arg1: memref<32x64xi32>,
                     %arg2: memref<64x64xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        %sum = affine.for %k = 0 to 32 iter_args(%acc = %c0_i32) -> (i32) {
          %a = affine.load %arg0[%i, %k] : memref<64x32xi32>
          %b = affine.load %arg1[%k, %j] : memref<32x64xi32>
          %p = arith.muli %a, %b         : i32
          %n = arith.addi %acc, %p       : i32
          affine.yield %n                : i32
        }
        affine.store %sum, %arg2[%i, %j] : memref<64x64xi32>
      }
    }
    return
  }

  // -----------------------------------------------------------------------
  // FC2 : Out[i,j] = sum_k Y[i,k] * W2[k,j]
  //   Y  : 64 x 64   (hidden activations, ReLU applied in-place before call)
  //   W2 : 64 x 32   (weight matrix)
  //   Out: 64 x 32   (output activations)
  // -----------------------------------------------------------------------
  func.func @ffn_fc2(%arg0: memref<64x64xi32>, %arg1: memref<64x32xi32>,
                     %arg2: memref<64x32xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 32 {
        %sum = affine.for %k = 0 to 64 iter_args(%acc = %c0_i32) -> (i32) {
          %a = affine.load %arg0[%i, %k] : memref<64x64xi32>
          %b = affine.load %arg1[%k, %j] : memref<64x32xi32>
          %p = arith.muli %a, %b         : i32
          %n = arith.addi %acc, %p       : i32
          affine.yield %n                : i32
        }
        affine.store %sum, %arg2[%i, %j] : memref<64x32xi32>
      }
    }
    return
  }

}

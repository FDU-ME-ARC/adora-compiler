// FFN: FC1(64x32 -> 64x64) + ReLU + FC2(64x64 -> 64x32)
// Realistic size: batch=64, input_dim=32, hidden_dim=64
// Converted to affine dialect
module {
  // FC1: Y[i,j] = sum_k X[i,k] * W1[k,j]
  func.func @ffn_fc1(%arg0: memref<64x32xi32>, %arg1: memref<32x64xi32>,
                     %arg2: memref<64x64xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        %sum = affine.for %k = 0 to 32 iter_args(%acc = %c0_i32) -> (i32) {
          %a = affine.load %arg0[%i, %k] : memref<64x32xi32>
          %b = affine.load %arg1[%k, %j] : memref<32x64xi32>
          %p = arith.muli %a, %b : i32
          %n = arith.addi %acc, %p : i32
          affine.yield %n : i32
        }
        affine.store %sum, %arg2[%i, %j] : memref<64x64xi32>
      }
    }
    return
  }

  // FC2: O[i,j] = sum_k Y[i,k] * W2[k,j]  (after ReLU applied in-place)
  func.func @ffn_fc2(%arg0: memref<64x64xi32>, %arg1: memref<64x32xi32>,
                     %arg2: memref<64x32xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 32 {
        %sum = affine.for %k = 0 to 64 iter_args(%acc = %c0_i32) -> (i32) {
          %a = affine.load %arg0[%i, %k] : memref<64x64xi32>
          %b = affine.load %arg1[%k, %j] : memref<64x32xi32>
          %p = arith.muli %a, %b : i32
          %n = arith.addi %acc, %p : i32
          affine.yield %n : i32
        }
        affine.store %sum, %arg2[%i, %j] : memref<64x32xi32>
      }
    }
    return
  }
}

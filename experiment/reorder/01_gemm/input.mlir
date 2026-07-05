// 01_gemm — classic i,j,k gemm with C[i,j] += A[i,k] * B[k,j].
// Original nest order is i,j,k. The reuse-group cost model finds that the
// %j loop is the cheapest as the innermost loop (B[k,j] and C[i,j] both
// become unit-stride, A[i,k] is loop-invariant), so it sinks %j inward,
// reordering the nest to i,k,j.
module {
  func.func @gemm(%A: memref<64x64xi32>, %B: memref<64x64xi32>, %C: memref<64x64xi32>) {
    affine.for %i = 0 to 64 {
      affine.for %j = 0 to 64 {
        affine.for %k = 0 to 64 {
          %a = affine.load %A[%i, %k] : memref<64x64xi32>
          %b = affine.load %B[%k, %j] : memref<64x64xi32>
          %c = affine.load %C[%i, %j] : memref<64x64xi32>
          %m = arith.muli %a, %b : i32
          %s = arith.addi %c, %m : i32
          affine.store %s, %C[%i, %j] : memref<64x64xi32>
        }
      }
    }
    return
  }
}

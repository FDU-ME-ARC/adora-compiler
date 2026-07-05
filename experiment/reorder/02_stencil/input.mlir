// 02_stencil — row-major elementwise copy B[i,j] = A[i,j] + 1.
// The nest is written with %j outer and %i inner, which leaves the strided
// %i loop innermost (bad spatial locality). The reuse-group cost model finds
// %j cheapest as the innermost loop (A[i,j] / B[i,j] are unit-stride in j),
// so it swaps the loops, reordering the nest from j,i to i,j.
module {
  func.func @row_copy(%A: memref<128x128xf32>, %B: memref<128x128xf32>) {
    affine.for %j = 0 to 128 {
      affine.for %i = 0 to 128 {
        %a = affine.load %A[%i, %j] : memref<128x128xf32>
        %c = arith.constant 1.0 : f32
        %s = arith.addf %a, %c : f32
        affine.store %s, %B[%i, %j] : memref<128x128xf32>
      }
    }
    return
  }
}

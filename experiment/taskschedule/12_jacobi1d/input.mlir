#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @jacobi_1d(%arg0: memref<120xf32>, %arg1: memref<120xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 40 {
       ADORA.kernel{
      affine.for %arg3 = 0 to 118 {
        %0 = affine.apply #map(%arg3)
        %1 = affine.load %arg0[%0 - 1] : memref<120xf32>
        %2 = affine.load %arg0[%0] : memref<120xf32>
        %3 = arith.addf %1, %2 : f32
        %4 = affine.load %arg0[%0 + 1] : memref<120xf32>
        %5 = arith.addf %3, %4 : f32
        %6 = arith.mulf %5, %cst : f32
        affine.store %6, %arg1[%0] : memref<120xf32>
      }
      ADORA.terminator}

      ADORA.kernel{
      affine.for %arg3 = 0 to 118 {
        %0 = affine.apply #map(%arg3)
        %1 = affine.load %arg1[%0 - 1] : memref<120xf32>
        %2 = affine.load %arg1[%0] : memref<120xf32>
        %3 = arith.addf %1, %2 : f32
        %4 = affine.load %arg1[%0 + 1] : memref<120xf32>
        %5 = arith.addf %3, %4 : f32
        %6 = arith.mulf %5, %cst : f32
        affine.store %6, %arg0[%0] : memref<120xf32>
      }
      ADORA.terminator}
    }
    return
  }
}


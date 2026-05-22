module {
  func.func @atax(%arg0: memref<390x410xf32>, %arg1: memref<410xf32>, %arg2: memref<410xf32>, %arg3: memref<390xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    // affine.for %arg4 = 0 to 410 {
    //   affine.store %cst, %arg2[%arg4] : memref<410xf32>
    // }
    affine.for %arg4 = 0 to 390 {
      affine.store %cst, %arg3[%arg4] : memref<390xf32>

      ADORA.kernel{
      affine.for %arg5 = 0 to 410 {
        %0 = affine.load %arg3[%arg4] : memref<390xf32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<390x410xf32>
        %2 = affine.load %arg1[%arg5] : memref<410xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        affine.store %4, %arg3[%arg4] : memref<390xf32>
      }
      ADORA.terminator}

      ADORA.kernel{
      affine.for %arg5 = 0 to 390 {
        %0 = affine.load %arg2[%arg5] : memref<410xf32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<390x410xf32>
        %2 = affine.load %arg3[%arg4] : memref<390xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        affine.store %4, %arg2[%arg5] : memref<410xf32>
      }
      ADORA.terminator}
    }
    return
  }
}


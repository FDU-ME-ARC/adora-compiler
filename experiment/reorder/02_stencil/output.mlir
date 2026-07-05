module {
  func.func @row_copy(%arg0: memref<128x128xf32>, %arg1: memref<128x128xf32>) {
    affine.for %arg2 = 0 to 128 {
      affine.for %arg3 = 0 to 128 {
        %0 = affine.load %arg0[%arg2, %arg3] : memref<128x128xf32>
        %cst = arith.constant 1.000000e+00 : f32
        %1 = arith.addf %0, %cst : f32
        affine.store %1, %arg1[%arg2, %arg3] : memref<128x128xf32>
      }
    }
    return
  }
}


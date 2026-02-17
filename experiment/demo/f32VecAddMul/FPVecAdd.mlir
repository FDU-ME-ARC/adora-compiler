module attributes {} {
  func.func @f32VecAddMul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %argr: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
      affine.for %arg3 = 0 to 20 {
        %3 = affine.load %arg0[%arg3] : memref<?xf32>
        %4 = affine.load %arg1[%arg3] : memref<?xf32>
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %arg2[%arg3] : memref<?xf32>
        %6 = arith.mulf %3, %4 : f32
        affine.store %6, %argr[%arg3] : memref<?xf32>
      }
    return
  }
}


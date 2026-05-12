module {
  func.func @f32VecAddMul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg4 = 0 to 20 {
      %0 = affine.load %arg0[%arg4] : memref<?xf32>
      %1 = affine.load %arg1[%arg4] : memref<?xf32>
      %2 = arith.addf %0, %1 : f32
      affine.store %2, %arg2[%arg4] : memref<?xf32>
      %3 = arith.mulf %0, %1 : f32
      affine.store %3, %arg3[%arg4] : memref<?xf32>
    }
    return
  }
}


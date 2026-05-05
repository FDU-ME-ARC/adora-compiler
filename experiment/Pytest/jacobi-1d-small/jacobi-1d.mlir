module attributes {} {
  func.func @jacobi_1d(%arg0: memref<?xf32>, %arg1: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
    affine.for %arg2 = 0 to 40 {
      affine.for %arg3 = 1 to 119 {
        %0 = affine.load %arg0[%arg3 - 1] : memref<?xf32>
        %1 = affine.load %arg0[%arg3] : memref<?xf32>
        %2 = arith.addf %0, %1 : f32
        %3 = affine.load %arg0[%arg3 + 1] : memref<?xf32>
        %4 = arith.addf %2, %3 : f32
        %5 = arith.mulf %4, %cst : f32
        affine.store %5, %arg1[%arg3] : memref<?xf32>
      }
      affine.for %arg3 = 1 to 119 {
        %0 = affine.load %arg1[%arg3 - 1] : memref<?xf32>
        %1 = affine.load %arg1[%arg3] : memref<?xf32>
        %2 = arith.addf %0, %1 : f32
        %3 = affine.load %arg1[%arg3 + 1] : memref<?xf32>
        %4 = arith.addf %2, %3 : f32
        %5 = arith.mulf %4, %cst : f32
        affine.store %5, %arg0[%arg3] : memref<?xf32>
      }
    }
    return
  }
}

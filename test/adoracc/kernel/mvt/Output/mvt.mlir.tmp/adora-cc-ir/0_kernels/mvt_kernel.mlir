module {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          %0 = affine.load %arg0[%arg5] : memref<?xf32>
          %1 = affine.load %arg4[%arg5, %arg6] : memref<?x40xf32>
          %2 = affine.load %arg2[%arg6] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %0, %3 : f32
          affine.store %4, %arg0[%arg5] : memref<?xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          %0 = affine.load %arg1[%arg5] : memref<?xf32>
          %1 = affine.load %arg4[%arg6, %arg5] : memref<?x40xf32>
          %2 = affine.load %arg3[%arg6] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %0, %3 : f32
          affine.store %4, %arg1[%arg5] : memref<?xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_1"}
    return
  }
}


module {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"}
    %1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"}
    %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"}
    %3 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %0[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 step 2 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %1[%arg5, %arg6] : memref<40x40xf32>
          %11 = affine.load %2[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          %14 = affine.load %1[%arg5, %arg6 + 1] : memref<40x40xf32>
          %15 = affine.load %2[%arg6 + 1] : memref<40xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %13, %16 : f32
          affine.yield %17 : f32
        }
        affine.store %9, %3[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.BlockStore %3, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    %4 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_1"}
    %5 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_1"}
    %6 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_1"}
    %7 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %4[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %5[%arg6, %arg5] : memref<40x40xf32>
          %11 = affine.load %6[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          affine.yield %13 : f32
        }
        affine.store %9, %7[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_1"}
    ADORA.BlockStore %7, %arg1 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    return
  }
}

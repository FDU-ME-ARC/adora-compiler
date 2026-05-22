module {
  func.func @gesummv_kernel_0(%arg0: memref<250xf32>, %arg1: memref<250xf32>, %arg2: memref<250x250xf32>, %arg3: memref<250xf32>, %arg4: memref<250x250xf32>) attributes {Kernel, gesummv_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %cst_1 = arith.constant 1.200000e+00 : f32
    affine.for %arg5 = 0 to 250 step 5 {
      %0 = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "0", KernelName = "gesummv_kernel_0"}
      %1 = ADORA.BlockLoad %arg3 [0] : memref<250xf32> -> memref<250xf32>  {Id = "1", KernelName = "gesummv_kernel_0"}
      %2 = ADORA.BlockLoad %arg4 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "2", KernelName = "gesummv_kernel_0"}
      %3 = ADORA.LocalMemAlloc memref<6xf32>  {Id = "3", KernelName = "gesummv_kernel_0"}
      %4 = ADORA.LocalMemAlloc memref<6xf32>  {Id = "4", KernelName = "gesummv_kernel_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 5 {
          %5:2 = affine.for %arg7 = 0 to 250 step 2 iter_args(%arg8 = %cst, %arg9 = %cst) -> (f32, f32) {
            %9 = affine.load %0[%arg6, %arg7] : memref<5x250xf32>
            %10 = affine.load %1[%arg7] : memref<250xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %11, %arg8 : f32
            %13 = affine.load %2[%arg6, %arg7] : memref<5x250xf32>
            %14 = arith.mulf %13, %10 : f32
            %15 = arith.addf %14, %arg9 : f32
            %16 = affine.load %0[%arg6, %arg7 + 1] : memref<5x250xf32>
            %17 = affine.load %1[%arg7 + 1] : memref<250xf32>
            %18 = arith.mulf %16, %17 : f32
            %19 = arith.addf %18, %12 : f32
            %20 = affine.load %2[%arg6, %arg7 + 1] : memref<5x250xf32>
            %21 = arith.mulf %20, %17 : f32
            %22 = arith.addf %21, %15 : f32
            affine.yield %19, %22 : f32, f32
          }
          // affine.store %5#1, %4[%arg6] : memref<6xf32>
          affine.store %5#0, %3[%arg6] : memref<6xf32>
          %6 = arith.mulf %5#0, %cst_0 : f32
          %7 = arith.mulf %5#1, %cst_1 : f32
          %8 = arith.addf %6, %7 : f32
          affine.store %8, %4[%arg6] : memref<6xf32>
        }
        ADORA.terminator
      } {KernelName = "gesummv_kernel_0"}
      ADORA.BlockStore %4, %arg1 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "4", KernelName = "gesummv_kernel_0"}
      ADORA.BlockStore %3, %arg0 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "3", KernelName = "gesummv_kernel_0"}
    }
    return
  }
}
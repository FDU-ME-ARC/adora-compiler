module {
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>, %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>, %arg6: memref<?x22xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0"}
    %1 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0"}
    %2 = ADORA.LocalMemAlloc memref<16x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 18 {
          %9 = affine.for %arg9 = 0 to 20 iter_args(%arg10 = %cst) -> (f32) {
            %10 = affine.load %0[%arg7, %arg9] : memref<16x20xf32>
            %11 = affine.load %1[%arg9, %arg8] : memref<20x18xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = arith.addf %arg10, %12 : f32
            affine.yield %13 : f32
          }
          affine.store %9, %2[%arg7, %arg8] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0"}
    ADORA.BlockStore %2, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    %3 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1"}
    %4 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1"}
    %5 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 18 {
        affine.for %arg8 = 0 to 22 {
          %9 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %cst) -> (f32) {
            %10 = affine.load %3[%arg7, %arg9] : memref<18x24xf32>
            %11 = affine.load %4[%arg9, %arg8] : memref<24x22xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = arith.addf %arg10, %12 : f32
            affine.yield %13 : f32
          }
          affine.store %9, %5[%arg7, %arg8] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1"}
    ADORA.BlockStore %5, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %6 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x18xf32> -> memref<16x18xf32>  {Id = "0", KernelName = "kernel_3mm_2"}
    %7 = ADORA.BlockLoad %arg3 [0, 0] : memref<?x22xf32> -> memref<18x22xf32>  {Id = "1", KernelName = "kernel_3mm_2"}
    %8 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 22 {
          %9 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %cst) -> (f32) {
            %10 = affine.load %6[%arg7, %arg9] : memref<16x18xf32>
            %11 = affine.load %7[%arg9, %arg8] : memref<18x22xf32>
            %12 = arith.mulf %10, %11 : f32
            %13 = arith.addf %arg10, %12 : f32
            affine.yield %13 : f32
          }
          affine.store %9, %8[%arg7, %arg8] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2"}
    ADORA.BlockStore %8, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    return
  }
}


module attributes {adora.scheduled} {
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>, %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>, %arg6: memref<?x22xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 10 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}]}]} {
    %cst = arith.constant 0.000000e+00 : f32
    %result = ADORA.BlockLoad %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0"}
    %result_0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0"}
    %0 = ADORA.LocalMemAlloc memref<16x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 18 {
          %3 = affine.for %arg9 = 0 to 20 iter_args(%arg10 = %cst) -> (f32) {
            %4 = affine.load %result[%arg7, %arg9] : memref<16x20xf32>
            %5 = affine.load %result_0[%arg9, %arg8] : memref<20x18xf32>
            %6 = arith.mulf %4, %5 : f32
            %7 = arith.addf %arg10, %6 : f32
            affine.yield %7 : f32
          }
          affine.store %3, %0[%arg7, %arg8] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0"}
    ADORA.BlockStore %0, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    %result_1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1"}
    %result_2 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1"}
    %1 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 18 {
        affine.for %arg8 = 0 to 22 {
          %3 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %cst) -> (f32) {
            %4 = affine.load %result_1[%arg7, %arg9] : memref<18x24xf32>
            %5 = affine.load %result_2[%arg9, %arg8] : memref<24x22xf32>
            %6 = arith.mulf %4, %5 : f32
            %7 = arith.addf %arg10, %6 : f32
            affine.yield %7 : f32
          }
          affine.store %3, %1[%arg7, %arg8] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1"}
    ADORA.BlockStore %1, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %result_3 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x18xf32> -> memref<16x18xf32>  {Id = "0", KernelName = "kernel_3mm_2"}
    %result_4 = ADORA.BlockLoad %arg3 [0, 0] : memref<?x22xf32> -> memref<18x22xf32>  {Id = "1", KernelName = "kernel_3mm_2"}
    %2 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 22 {
          %3 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %cst) -> (f32) {
            %4 = affine.load %result_3[%arg7, %arg9] : memref<16x18xf32>
            %5 = affine.load %result_4[%arg9, %arg8] : memref<18x22xf32>
            %6 = arith.mulf %4, %5 : f32
            %7 = arith.addf %arg10, %6 : f32
            affine.yield %7 : f32
          }
          affine.store %3, %2[%arg7, %arg8] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2"}
    ADORA.BlockStore %2, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    return
  }
}


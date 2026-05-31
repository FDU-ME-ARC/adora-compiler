module attributes {adora.scheduled} {
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>, %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>, %arg6: memref<?x22xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 11 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 10 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 13 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}]}]} {
    %cst = arith.constant 0.000000e+00 : f32
    %result, %asyncToken = ADORA.BlockLoad async [] %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0"}
    %result_0, %asyncToken_1 = ADORA.BlockLoad async [] %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0"}
    %0 = ADORA.LocalMemAlloc memref<16x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 18 {
          %8 = affine.for %arg9 = 0 to 20 iter_args(%arg10 = %cst) -> (f32) {
            %9 = affine.load %result[%arg7, %arg9] : memref<16x20xf32>
            %10 = affine.load %result_0[%arg9, %arg8] : memref<20x18xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg10, %11 : f32
            affine.yield %12 : f32
          }
          affine.store %8, %0[%arg7, %arg8] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0"}
    %2 = ADORA.BlockStore async [%1] %0, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    %result_2, %asyncToken_3 = ADORA.BlockLoad async [] %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1"}
    %result_4, %asyncToken_5 = ADORA.BlockLoad async [] %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1"}
    %3 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %4 = ADORA.kernel async [%asyncToken_5, %asyncToken_3] {
      affine.for %arg7 = 0 to 18 {
        affine.for %arg8 = 0 to 22 {
          %8 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %cst) -> (f32) {
            %9 = affine.load %result_2[%arg7, %arg9] : memref<18x24xf32>
            %10 = affine.load %result_4[%arg9, %arg8] : memref<24x22xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg10, %11 : f32
            affine.yield %12 : f32
          }
          affine.store %8, %3[%arg7, %arg8] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1"}
    %5 = ADORA.BlockStore async [%4] %3, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %6 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    %7 = ADORA.kernel async [%5, %2] {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 22 {
          %8 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %cst) -> (f32) {
            %9 = affine.load %0[%arg7, %arg9] : memref<16x18xf32>
            %10 = affine.load %3[%arg9, %arg8] : memref<18x22xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg10, %11 : f32
            affine.yield %12 : f32
          }
          affine.store %8, %6[%arg7, %arg8] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2"}
    ADORA.BlockStore %6, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    return
  }
}


module attributes {adora.scheduled} {
  func.func @gemm_tiled(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 5 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}]} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        affine.for %arg5 = 0 to 4 {
          %result = ADORA.BlockLoad %arg2 [%arg3 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "0", KernelName = "gemm_tiled_tk"}
          %result_0 = ADORA.BlockLoad %arg0 [%arg3 * 16, %arg5 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "1", KernelName = "gemm_tiled_tk"}
          %result_1 = ADORA.BlockLoad %arg1 [%arg5 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "2", KernelName = "gemm_tiled_tk"}
          %0 = ADORA.LocalMemAlloc memref<16x16xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          ADORA.kernel {
            affine.for %arg6 = 0 to 16 {
              affine.for %arg7 = 0 to 16 {
                %1 = affine.load %result[%arg6, %arg7] : memref<16x16xf32>
                %2 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %1) -> (f32) {
                  %3 = affine.load %result_0[%arg6, %arg8] : memref<16x16xf32>
                  %4 = affine.load %result_1[%arg8, %arg7] : memref<16x16xf32>
                  %5 = arith.mulf %3, %4 : f32
                  %6 = arith.addf %arg9, %5 : f32
                  affine.yield %6 : f32
                }
                affine.store %2, %0[%arg6, %arg7] : memref<16x16xf32>
              }
            }
            ADORA.terminator
          } {KernelName = "gemm_tiled_tk"}
          ADORA.BlockStore %0, %arg2 [%arg3 * 16, %arg4 * 16] : memref<16x16xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
        }
      }
    }
    return
  }
}


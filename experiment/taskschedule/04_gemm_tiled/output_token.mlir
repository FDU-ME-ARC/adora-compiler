module attributes {adora.scheduled} {
  func.func @gemm_tiled(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 5 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}]} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %0 = ADORA.event.create -> !ADORA.token
        %1 = ADORA.event.create -> !ADORA.token
        %2 = ADORA.event.create -> !ADORA.token
        %3:3 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %0, %arg7 = %1, %arg8 = %2) -> (!ADORA.token, !ADORA.token, !ADORA.token) {
          %result, %asyncToken = ADORA.BlockLoad async [%arg7] %arg2 [%arg3 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "0", KernelName = "gemm_tiled_tk"}
          %result_0, %asyncToken_1 = ADORA.BlockLoad async [] %arg0 [%arg3 * 16, %arg5 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "1", KernelName = "gemm_tiled_tk"}
          %result_2, %asyncToken_3 = ADORA.BlockLoad async [] %arg1 [%arg5 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "2", KernelName = "gemm_tiled_tk"}
          %4 = ADORA.LocalMemAlloc memref<16x16xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          %5 = ADORA.kernel async [%asyncToken_3, %asyncToken_1, %asyncToken] {
            affine.for %arg9 = 0 to 16 {
              affine.for %arg10 = 0 to 16 {
                %7 = affine.load %result[%arg9, %arg10] : memref<16x16xf32>
                %8 = affine.for %arg11 = 0 to 16 iter_args(%arg12 = %7) -> (f32) {
                  %9 = affine.load %result_0[%arg9, %arg11] : memref<16x16xf32>
                  %10 = affine.load %result_2[%arg11, %arg10] : memref<16x16xf32>
                  %11 = arith.mulf %9, %10 : f32
                  %12 = arith.addf %arg12, %11 : f32
                  affine.yield %12 : f32
                }
                affine.store %8, %4[%arg9, %arg10] : memref<16x16xf32>
              }
            }
            ADORA.terminator
          } {KernelName = "gemm_tiled_tk"}
          %6 = ADORA.BlockStore async [%5, %asyncToken, %arg6, %arg8] %4, %arg2 [%arg3 * 16, %arg4 * 16] : memref<16x16xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          affine.yield %asyncToken, %6, %6 : !ADORA.token, !ADORA.token, !ADORA.token
        }
      }
    }
    return
  }
}


module attributes {adora.scheduled} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 10 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 7 : i64, kind = "RAR", overlap = true, src = 1 : i64}, {dst = 11 : i64, kind = "WAR", overlap = true, src = 6 : i64}, {dst = 5 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"} -> !ADORA.token
    %result_2, %asyncToken_3 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    %1 = ADORA.kernel async [%asyncToken_3, %asyncToken_1, %asyncToken] {
      affine.for %arg5 = 0 to 40 {
        %4 = affine.load %result[%arg5] : memref<40xf32>
        %5 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %4) -> (f32) {
          %6 = affine.load %result_0[%arg5, %arg6] : memref<40x40xf32>
          %7 = affine.load %result_2[%arg6] : memref<40xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %arg7, %8 : f32
          affine.yield %9 : f32
        }
        affine.store %5, %0[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.BlockStore async [%1, %asyncToken] %0, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    %result_4, %asyncToken_5 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_1"} -> !ADORA.token
    %result_6, %asyncToken_7 = ADORA.BlockLoad async [%asyncToken_1] %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_1"} -> !ADORA.token
    %result_8, %asyncToken_9 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_1"} -> !ADORA.token
    %2 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    %3 = ADORA.kernel async [%asyncToken_9, %asyncToken_7, %asyncToken_5] {
      affine.for %arg5 = 0 to 40 {
        %4 = affine.load %result_4[%arg5] : memref<40xf32>
        %5 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %4) -> (f32) {
          %6 = affine.load %result_6[%arg6, %arg5] : memref<40x40xf32>
          %7 = affine.load %result_8[%arg6] : memref<40xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %arg7, %8 : f32
          affine.yield %9 : f32
        }
        affine.store %5, %2[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_1"}
    ADORA.BlockStore async [%3, %asyncToken_5] %2, %arg1 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    return
  }
}


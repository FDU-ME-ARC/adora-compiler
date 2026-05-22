module attributes {adora.scheduled} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 9 : i64, kind = "WAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 9 : i64, kind = "WAR", overlap = true, src = 4 : i64}, {dst = 9 : i64, kind = "WAR", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "WAR", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-WAW", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-WAW", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-WAW", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %0 = ADORA.event.create -> !ADORA.token
    %1 = ADORA.event.create -> !ADORA.token
    %2 = ADORA.event.create -> !ADORA.token
    %3 = ADORA.event.create -> !ADORA.token
    %4 = ADORA.event.create -> !ADORA.token
    %5 = ADORA.event.create -> !ADORA.token
    %6 = ADORA.event.create -> !ADORA.token
    %7 = ADORA.event.create -> !ADORA.token
    %8 = ADORA.event.create -> !ADORA.token
    %9 = ADORA.event.create -> !ADORA.token
    %10 = ADORA.event.create -> !ADORA.token
    %11 = ADORA.event.create -> !ADORA.token
    %12:12 = affine.for %arg3 = 0 to 20 iter_args(%arg4 = %0, %arg5 = %1, %arg6 = %2, %arg7 = %3, %arg8 = %4, %arg9 = %5, %arg10 = %6, %arg11 = %7, %arg12 = %8, %arg13 = %9, %arg14 = %10, %arg15 = %11) -> (!ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token) {
      %result, %asyncToken = ADORA.BlockLoad async [%arg6, %arg12] %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"} -> !ADORA.token
      %13 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "gemm_0"}
      %14 = ADORA.kernel async [%asyncToken] {
        affine.for %arg16 = 0 to 25 {
          %19 = affine.load %result[0, %arg16] : memref<1x25xf32>
          %20 = arith.mulf %19, %cst : f32
          affine.store %20, %13[0, %arg16] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_0"}
      %15 = ADORA.BlockStore async [%14, %asyncToken, %arg4, %arg7, %arg10, %arg13] %13, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"} -> !ADORA.token
      %result_1, %asyncToken_2 = ADORA.BlockLoad async [%15, %asyncToken, %arg8, %arg14] %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_1"} -> !ADORA.token
      %result_3, %asyncToken_4 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "gemm_1"} -> !ADORA.token
      %result_5, %asyncToken_6 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "gemm_1"} -> !ADORA.token
      %16 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "gemm_1"}
      %17 = ADORA.kernel async [%asyncToken_6, %asyncToken_4, %asyncToken_2] {
        affine.for %arg16 = 0 to 25 {
          %19 = affine.load %result_1[0, %arg16] : memref<1x25xf32>
          %20 = affine.for %arg17 = 0 to 30 iter_args(%arg18 = %19) -> (f32) {
            %21 = affine.load %result_3[0, %arg17] : memref<1x30xf32>
            %22 = arith.mulf %21, %cst_0 : f32
            %23 = affine.load %result_5[%arg17, %arg16] : memref<30x25xf32>
            %24 = arith.mulf %22, %23 : f32
            %25 = arith.addf %arg18, %24 : f32
            affine.yield %25 : f32
          }
          affine.store %20, %16[0, %arg16] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "gemm_1"}
      %18 = ADORA.BlockStore async [%17, %15, %asyncToken_2, %asyncToken, %arg5, %arg9, %arg11, %arg15] %16, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_1"} -> !ADORA.token
      affine.yield %asyncToken, %asyncToken, %15, %15, %15, %15, %asyncToken_2, %asyncToken_2, %18, %18, %18, %18 : !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token
    }
    return
  }
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 9 : i64, kind = "WAR", overlap = true, src = 4 : i64}, {dst = 9 : i64, kind = "WAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 9 : i64, kind = "WAR", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAR", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_0"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
    %1 = ADORA.kernel async [%asyncToken] {
      affine.for %arg3 = 0 to 20 {
        affine.for %arg4 = 0 to 25 {
          %5 = affine.load %result[%arg3, %arg4] : memref<20x25xf32>
          %6 = arith.mulf %5, %cst : f32
          affine.store %6, %0[%arg3, %arg4] : memref<20x25xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "gemm_opt_0"}
    %2 = ADORA.BlockStore async [%1, %asyncToken] %0, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"} -> !ADORA.token
    %result_1, %asyncToken_2 = ADORA.BlockLoad async [%2, %asyncToken] %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_1"} -> !ADORA.token
    %result_3, %asyncToken_4 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x30xf32> -> memref<20x30xf32>  {Id = "1", KernelName = "gemm_opt_1"} -> !ADORA.token
    %result_5, %asyncToken_6 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "gemm_opt_1"} -> !ADORA.token
    %3 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
    %4 = ADORA.kernel async [%asyncToken_2, %asyncToken_6, %asyncToken_4] {
      affine.for %arg3 = 0 to 20 {
        affine.for %arg4 = 0 to 25 {
          %5 = affine.load %result_1[%arg3, %arg4] : memref<20x25xf32>
          %6 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %5) -> (f32) {
            %7 = affine.load %result_3[%arg3, %arg5] : memref<20x30xf32>
            %8 = arith.mulf %7, %cst_0 : f32
            %9 = affine.load %result_5[%arg5, %arg4] : memref<30x25xf32>
            %10 = arith.mulf %8, %9 : f32
            %11 = arith.addf %arg6, %10 : f32
            affine.yield %11 : f32
          }
          affine.store %6, %3[%arg3, %arg4] : memref<20x25xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "gemm_opt_1"}
    ADORA.BlockStore async [%4, %asyncToken_2, %2, %asyncToken] %3, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
    return
  }
}


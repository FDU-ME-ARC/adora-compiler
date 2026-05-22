#map = affine_map<(d0) -> (d0 + 1)>
module attributes {adora.scheduled} {
  func.func @jacobi_1d(%arg0: memref<2000xf32>, %arg1: memref<2000xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 10 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 6 : i64}, {dst = 8 : i64, kind = "RAR", overlap = false, src = 6 : i64}, {dst = 8 : i64, kind = "RAR", overlap = false, src = 7 : i64}, {dst = 6 : i64, kind = "RAW", overlap = false, src = 5 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 8 : i64, kind = "RAW", overlap = false, src = 5 : i64}, {dst = 11 : i64, kind = "WAR", overlap = false, src = 2 : i64}, {dst = 11 : i64, kind = "WAR", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 11 : i64, kind = "WAR", overlap = false, src = 0 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 1 : i64, kind = "RAR", overlap = false, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = false, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 3.333300e-01 : f32
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
    %12 = ADORA.event.create -> !ADORA.token
    %13 = ADORA.event.create -> !ADORA.token
    %14:14 = affine.for %arg2 = 0 to 500 iter_args(%arg3 = %0, %arg4 = %1, %arg5 = %2, %arg6 = %3, %arg7 = %4, %arg8 = %5, %arg9 = %6, %arg10 = %7, %arg11 = %8, %arg12 = %9, %arg13 = %10, %arg14 = %11, %arg15 = %12, %arg16 = %13) -> (!ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token) {
      %result, %asyncToken = ADORA.BlockLoad async [%arg13] %arg0 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_0"} -> !ADORA.token
      %result_0, %asyncToken_1 = ADORA.BlockLoad async [%asyncToken, %arg14] %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_0"} -> !ADORA.token
      %result_2, %asyncToken_3 = ADORA.BlockLoad async [%asyncToken_1, %asyncToken, %arg15] %arg0 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_0"} -> !ADORA.token
      %15 = ADORA.LocalMemAlloc memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_0"}
      %16 = ADORA.kernel async [%asyncToken_3, %asyncToken_1, %asyncToken] {
        affine.for %arg17 = 0 to 1998 {
          %21 = affine.apply #map(%arg17)
          %22 = affine.load %result[%21 - 1] : memref<2000xf32>
          %23 = affine.load %result_0[%21 - 1] : memref<2000xf32>
          %24 = arith.addf %22, %23 : f32
          %25 = affine.load %result_2[%21 - 1] : memref<2000xf32>
          %26 = arith.addf %24, %25 : f32
          %27 = arith.mulf %26, %cst : f32
          affine.store %27, %15[%21 - 1] : memref<2000xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_jacobi_1d_0"}
      %17 = ADORA.BlockStore async [%16, %arg6, %arg10, %arg11, %arg12] %15, %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_0"} -> !ADORA.token
      %result_4, %asyncToken_5 = ADORA.BlockLoad async [%17, %arg7] %arg1 [0] : memref<2000xf32> -> memref<2000xf32>  {Id = "0", KernelName = "kernel_jacobi_1d_1"} -> !ADORA.token
      %result_6, %asyncToken_7 = ADORA.BlockLoad async [%asyncToken_5, %17, %arg8] %arg1 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "1", KernelName = "kernel_jacobi_1d_1"} -> !ADORA.token
      %result_8, %asyncToken_9 = ADORA.BlockLoad async [%asyncToken_5, %asyncToken_7, %17, %arg9] %arg1 [2] : memref<2000xf32> -> memref<2000xf32>  {Id = "2", KernelName = "kernel_jacobi_1d_1"} -> !ADORA.token
      %18 = ADORA.LocalMemAlloc memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_1"}
      %19 = ADORA.kernel async [%asyncToken_5, %asyncToken_7, %asyncToken_9] {
        affine.for %arg17 = 0 to 1998 {
          %21 = affine.apply #map(%arg17)
          %22 = affine.load %result_4[%21 - 1] : memref<2000xf32>
          %23 = affine.load %result_6[%21 - 1] : memref<2000xf32>
          %24 = arith.addf %22, %23 : f32
          %25 = affine.load %result_8[%21 - 1] : memref<2000xf32>
          %26 = arith.addf %24, %25 : f32
          %27 = arith.mulf %26, %cst : f32
          affine.store %27, %18[%21 - 1] : memref<2000xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_jacobi_1d_1"}
      %20 = ADORA.BlockStore async [%19, %asyncToken_3, %asyncToken_1, %asyncToken, %arg3, %arg4, %arg5, %arg16] %18, %arg0 [1] : memref<2000xf32> -> memref<2000xf32>  {Id = "3", KernelName = "kernel_jacobi_1d_1"} -> !ADORA.token
      affine.yield %asyncToken, %asyncToken_1, %asyncToken_3, %17, %17, %17, %17, %asyncToken_5, %asyncToken_7, %asyncToken_9, %20, %20, %20, %20 : !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token
    }
    return
  }
}


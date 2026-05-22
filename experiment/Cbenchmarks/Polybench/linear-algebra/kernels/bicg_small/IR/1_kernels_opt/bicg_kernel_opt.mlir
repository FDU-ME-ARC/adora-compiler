module attributes {adora.scheduled} {
  func.func @bicg(%arg0: memref<?x116xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 6 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 7 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 4 : i64}, {exact = true, kind = "LC-WAR", step = 4 : i64}, {exact = false, kind = "LC-RAR", step = 4 : i64}, {exact = false, kind = "LC-RAR", step = 4 : i64}, {exact = true, kind = "LC-RAR", step = 4 : i64}, {exact = true, kind = "LC-RAW", step = 4 : i64}, {exact = true, kind = "LC-WAW", step = 4 : i64}, {exact = false, kind = "LC-WAW", step = 4 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg5 = 0 to 116 {
      affine.store %cst, %arg1[%arg5] : memref<?xf32>
    }
    %0 = ADORA.event.create -> !ADORA.token
    %1 = ADORA.event.create -> !ADORA.token
    %2 = ADORA.event.create -> !ADORA.token
    %3 = ADORA.event.create -> !ADORA.token
    %4:4 = affine.for %arg5 = 0 to 124 step 4 iter_args(%arg6 = %0, %arg7 = %1, %arg8 = %2, %arg9 = %3) -> (!ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token) {
      %result, %asyncToken = ADORA.BlockLoad async [%arg7] %arg1 [0] : memref<?xf32> -> memref<116xf32>  {Id = "0", KernelName = "bicg"} -> !ADORA.token
      %result_0, %asyncToken_1 = ADORA.BlockLoad %arg4 [%arg5] : memref<?xf32> -> memref<4xf32>  {Id = "1", KernelName = "bicg"} -> !ADORA.token
      %result_2, %asyncToken_3 = ADORA.BlockLoad %arg0 [%arg5, 0] : memref<?x116xf32> -> memref<4x116xf32>  {Id = "2", KernelName = "bicg"} -> !ADORA.token
      %result_4, %asyncToken_5 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<116xf32>  {Id = "3", KernelName = "bicg"} -> !ADORA.token
      %5 = ADORA.LocalMemAlloc memref<4xf32>  {Id = "4", KernelName = "bicg"}
      %6 = ADORA.LocalMemAlloc memref<116xf32>  {Id = "5", KernelName = "bicg"}
      %7 = ADORA.kernel async [%asyncToken_5, %asyncToken_3, %asyncToken_1, %asyncToken] {
        affine.for %arg10 = 0 to 4 {
          %10 = affine.for %arg11 = 0 to 116 iter_args(%arg12 = %cst) -> (f32) {
            %11 = affine.load %result[%arg11] : memref<116xf32>
            %12 = affine.load %result_0[%arg10] : memref<4xf32>
            %13 = affine.load %result_2[%arg10, %arg11] : memref<4x116xf32>
            %14 = arith.mulf %12, %13 : f32
            %15 = arith.addf %11, %14 : f32
            affine.store %15, %6[%arg11] : memref<116xf32>
            %16 = affine.load %result_4[%arg11] : memref<116xf32>
            %17 = arith.mulf %13, %16 : f32
            %18 = arith.addf %arg12, %17 : f32
            affine.yield %18 : f32
          }
          affine.store %10, %5[%arg10] : memref<4xf32>
        }
        ADORA.terminator
      } {KernelName = "bicg"}
      %8 = ADORA.BlockStore async [%7, %asyncToken, %arg6, %arg8] %6, %arg1 [0] : memref<116xf32> -> memref<?xf32>  {Id = "5", KernelName = "bicg"} -> !ADORA.token
      %9 = ADORA.BlockStore async [%7, %arg9] %5, %arg2 [%arg5] : memref<4xf32> -> memref<?xf32>  {Id = "4", KernelName = "bicg"} -> !ADORA.token
      affine.yield %asyncToken, %8, %8, %9 : !ADORA.token, !ADORA.token, !ADORA.token, !ADORA.token
    }
    return
  }
}


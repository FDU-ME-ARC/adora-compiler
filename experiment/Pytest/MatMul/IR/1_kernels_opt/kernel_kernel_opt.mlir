module attributes {adora.scheduled} {
  func.func @merge_MATMUL_4x4(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 6 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 5 : i64}, {dst = 6 : i64, kind = "RAR", overlap = false, src = 5 : i64}, {dst = 4 : i64, kind = "RAR", overlap = false, src = 3 : i64}, {dst = 4 : i64, kind = "RAR", overlap = false, src = 2 : i64}, {dst = 3 : i64, kind = "RAR", overlap = false, src = 2 : i64}, {dst = 4 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 3 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 6 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 5 : i64, kind = "RAR", overlap = false, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_2, %asyncToken_3 = ADORA.BlockLoad async [%asyncToken_1] %arg1 [0, 1] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_4, %asyncToken_5 = ADORA.BlockLoad async [%asyncToken_3, %asyncToken_1] %arg1 [0, 2] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_6, %asyncToken_7 = ADORA.BlockLoad async [%asyncToken_5, %asyncToken_3, %asyncToken_1] %arg1 [0, 3] : memref<?x36xi32> -> memref<36x33xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_8, %asyncToken_9 = ADORA.BlockLoad async [%asyncToken] %arg0 [1, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_10, %asyncToken_11 = ADORA.BlockLoad async [%asyncToken_9, %asyncToken] %arg0 [2, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %result_12, %asyncToken_13 = ADORA.BlockLoad async [%asyncToken_11, %asyncToken_9, %asyncToken] %arg0 [3, 0] : memref<?x36xi32> -> memref<33x36xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<36x36xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4"}
    %1 = ADORA.kernel async [%asyncToken_13, %asyncToken_11, %asyncToken_9, %asyncToken_7, %asyncToken_5, %asyncToken_3, %asyncToken_1, %asyncToken] {
      affine.for %arg3 = 0 to 9 {
        affine.for %arg4 = 0 to 9 {
          %2:16 = affine.for %arg5 = 0 to 36 iter_args(%arg6 = %c0_i32, %arg7 = %c0_i32, %arg8 = %c0_i32, %arg9 = %c0_i32, %arg10 = %c0_i32, %arg11 = %c0_i32, %arg12 = %c0_i32, %arg13 = %c0_i32, %arg14 = %c0_i32, %arg15 = %c0_i32, %arg16 = %c0_i32, %arg17 = %c0_i32, %arg18 = %c0_i32, %arg19 = %c0_i32, %arg20 = %c0_i32, %arg21 = %c0_i32) -> (i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32) {
            %3 = affine.load %result[%arg3 * 4, %arg5] : memref<33x36xi32>
            %4 = affine.load %result_0[%arg5, %arg4 * 4] : memref<36x33xi32>
            %5 = arith.muli %3, %4 : i32
            %6 = arith.addi %arg6, %5 : i32
            %7 = affine.load %result_2[%arg5, %arg4 * 4] : memref<36x33xi32>
            %8 = arith.muli %3, %7 : i32
            %9 = arith.addi %arg7, %8 : i32
            %10 = affine.load %result_4[%arg5, %arg4 * 4] : memref<36x33xi32>
            %11 = arith.muli %3, %10 : i32
            %12 = arith.addi %arg8, %11 : i32
            %13 = affine.load %result_6[%arg5, %arg4 * 4] : memref<36x33xi32>
            %14 = arith.muli %3, %13 : i32
            %15 = arith.addi %arg9, %14 : i32
            %16 = affine.load %result_8[%arg3 * 4, %arg5] : memref<33x36xi32>
            %17 = arith.muli %16, %4 : i32
            %18 = arith.addi %arg10, %17 : i32
            %19 = arith.muli %16, %7 : i32
            %20 = arith.addi %arg11, %19 : i32
            %21 = arith.muli %16, %10 : i32
            %22 = arith.addi %arg12, %21 : i32
            %23 = arith.muli %16, %13 : i32
            %24 = arith.addi %arg13, %23 : i32
            %25 = affine.load %result_10[%arg3 * 4, %arg5] : memref<33x36xi32>
            %26 = arith.muli %25, %4 : i32
            %27 = arith.addi %arg14, %26 : i32
            %28 = arith.muli %25, %7 : i32
            %29 = arith.addi %arg15, %28 : i32
            %30 = arith.muli %25, %10 : i32
            %31 = arith.addi %arg16, %30 : i32
            %32 = arith.muli %25, %13 : i32
            %33 = arith.addi %arg17, %32 : i32
            %34 = affine.load %result_12[%arg3 * 4, %arg5] : memref<33x36xi32>
            %35 = arith.muli %34, %4 : i32
            %36 = arith.addi %arg18, %35 : i32
            %37 = arith.muli %34, %7 : i32
            %38 = arith.addi %arg19, %37 : i32
            %39 = arith.muli %34, %10 : i32
            %40 = arith.addi %arg20, %39 : i32
            %41 = arith.muli %34, %13 : i32
            %42 = arith.addi %arg21, %41 : i32
            affine.yield %6, %9, %12, %15, %18, %20, %22, %24, %27, %29, %31, %33, %36, %38, %40, %42 : i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32
          }
          affine.store %2#15, %0[%arg3 * 4 + 3, %arg4 * 4 + 3] : memref<36x36xi32>
          affine.store %2#14, %0[%arg3 * 4 + 3, %arg4 * 4 + 2] : memref<36x36xi32>
          affine.store %2#13, %0[%arg3 * 4 + 3, %arg4 * 4 + 1] : memref<36x36xi32>
          affine.store %2#12, %0[%arg3 * 4 + 3, %arg4 * 4] : memref<36x36xi32>
          affine.store %2#11, %0[%arg3 * 4 + 2, %arg4 * 4 + 3] : memref<36x36xi32>
          affine.store %2#10, %0[%arg3 * 4 + 2, %arg4 * 4 + 2] : memref<36x36xi32>
          affine.store %2#9, %0[%arg3 * 4 + 2, %arg4 * 4 + 1] : memref<36x36xi32>
          affine.store %2#8, %0[%arg3 * 4 + 2, %arg4 * 4] : memref<36x36xi32>
          affine.store %2#7, %0[%arg3 * 4 + 1, %arg4 * 4 + 3] : memref<36x36xi32>
          affine.store %2#6, %0[%arg3 * 4 + 1, %arg4 * 4 + 2] : memref<36x36xi32>
          affine.store %2#5, %0[%arg3 * 4 + 1, %arg4 * 4 + 1] : memref<36x36xi32>
          affine.store %2#4, %0[%arg3 * 4 + 1, %arg4 * 4] : memref<36x36xi32>
          affine.store %2#3, %0[%arg3 * 4, %arg4 * 4 + 3] : memref<36x36xi32>
          affine.store %2#2, %0[%arg3 * 4, %arg4 * 4 + 2] : memref<36x36xi32>
          affine.store %2#1, %0[%arg3 * 4, %arg4 * 4 + 1] : memref<36x36xi32>
          affine.store %2#0, %0[%arg3 * 4, %arg4 * 4] : memref<36x36xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4"}
    ADORA.BlockStore async [%1] %0, %arg2 [0, 0] : memref<36x36xi32> -> memref<?x36xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4"}
    return
  }
}


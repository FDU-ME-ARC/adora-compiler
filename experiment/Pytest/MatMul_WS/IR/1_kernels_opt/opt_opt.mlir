module attributes {adora.scheduled} {
  func.func @merge_MATMUL_4x4_IS(%arg0: memref<?x36xi32>, %arg1: memref<?x36xi32>, %arg2: memref<?x36xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 12 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 12 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 13 : i64, kind = "WAR", overlap = false, src = 1 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 3 : i64, kind = "RAR", overlap = false, src = 1 : i64}, {dst = 10 : i64, kind = "WAR", overlap = false, src = 1 : i64}, {dst = 11 : i64, kind = "WAR", overlap = false, src = 1 : i64}, {dst = 12 : i64, kind = "WAR", overlap = true, src = 1 : i64}, {dst = 13 : i64, kind = "WAR", overlap = false, src = 2 : i64}, {dst = 3 : i64, kind = "RAR", overlap = false, src = 2 : i64}, {dst = 10 : i64, kind = "WAR", overlap = false, src = 2 : i64}, {dst = 11 : i64, kind = "WAR", overlap = true, src = 2 : i64}, {dst = 12 : i64, kind = "WAR", overlap = false, src = 2 : i64}, {dst = 13 : i64, kind = "WAR", overlap = false, src = 3 : i64}, {dst = 10 : i64, kind = "WAR", overlap = true, src = 3 : i64}, {dst = 11 : i64, kind = "WAR", overlap = false, src = 3 : i64}, {dst = 12 : i64, kind = "WAR", overlap = false, src = 3 : i64}, {dst = 6 : i64, kind = "RAR", overlap = false, src = 4 : i64}, {dst = 8 : i64, kind = "RAR", overlap = false, src = 4 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 4 : i64}, {dst = 13 : i64, kind = "WAR", overlap = true, src = 0 : i64}, {dst = 1 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 3 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 10 : i64, kind = "WAR", overlap = false, src = 0 : i64}, {dst = 11 : i64, kind = "WAR", overlap = false, src = 0 : i64}, {dst = 12 : i64, kind = "WAR", overlap = false, src = 0 : i64}, {dst = 8 : i64, kind = "RAR", overlap = false, src = 6 : i64}, {dst = 7 : i64, kind = "RAR", overlap = false, src = 6 : i64}, {dst = 8 : i64, kind = "RAR", overlap = false, src = 7 : i64}, {dst = 13 : i64, kind = "WAW", overlap = false, src = 10 : i64}, {dst = 11 : i64, kind = "WAW", overlap = false, src = 10 : i64}, {dst = 12 : i64, kind = "WAW", overlap = false, src = 10 : i64}, {dst = 13 : i64, kind = "WAW", overlap = false, src = 11 : i64}, {dst = 12 : i64, kind = "WAW", overlap = false, src = 11 : i64}, {dst = 13 : i64, kind = "WAW", overlap = false, src = 12 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %result, %asyncToken = ADORA.BlockLoad %arg2 [0, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad async [%asyncToken] %arg2 [2, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_2, %asyncToken_3 = ADORA.BlockLoad async [%asyncToken_1, %asyncToken] %arg2 [1, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_4, %asyncToken_5 = ADORA.BlockLoad async [%asyncToken_1, %asyncToken_3, %asyncToken] %arg2 [3, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_6, %asyncToken_7 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "4", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_8, %asyncToken_9 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x36xi32> -> memref<36x36xi32>  {Id = "5", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_10, %asyncToken_11 = ADORA.BlockLoad async [%asyncToken_7] %arg0 [1, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "6", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_12, %asyncToken_13 = ADORA.BlockLoad async [%asyncToken_7, %asyncToken_11] %arg0 [2, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "7", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %result_14, %asyncToken_15 = ADORA.BlockLoad async [%asyncToken_7, %asyncToken_11, %asyncToken_13] %arg0 [3, 0] : memref<?x36xi32> -> memref<9x36xi32>  {Id = "8", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %0 = ADORA.kernel async [%asyncToken_15, %asyncToken_13, %asyncToken_11, %asyncToken_9, %asyncToken_7, %asyncToken_5, %asyncToken_3, %asyncToken_1, %asyncToken] {
      affine.for %arg3 = 0 to 36 {
        affine.for %arg4 = 0 to 36 {
          affine.for %arg5 = 0 to 9 {
            %4 = affine.load %result_6[%arg5, %arg4] : memref<9x36xi32>
            %5 = affine.load %result_8[%arg4, %arg3] : memref<36x36xi32>
            %6 = arith.muli %4, %5 : i32
            %7 = affine.load %result[%arg5, %arg3] : memref<9x36xi32>
            %8 = arith.addi %7, %6 : i32
            affine.store %8, %result[%arg5, %arg3] : memref<9x36xi32>
            %9 = affine.load %result_10[%arg5, %arg4] : memref<9x36xi32>
            %10 = arith.muli %9, %5 : i32
            %11 = affine.load %result_2[%arg5, %arg3] : memref<9x36xi32>
            %12 = arith.addi %11, %10 : i32
            affine.store %12, %result_2[%arg5, %arg3] : memref<9x36xi32>
            %13 = affine.load %result_12[%arg5, %arg4] : memref<9x36xi32>
            %14 = arith.muli %13, %5 : i32
            %15 = affine.load %result_0[%arg5, %arg3] : memref<9x36xi32>
            %16 = arith.addi %15, %14 : i32
            affine.store %16, %result_0[%arg5, %arg3] : memref<9x36xi32>
            %17 = affine.load %result_14[%arg5, %arg4] : memref<9x36xi32>
            %18 = arith.muli %17, %5 : i32
            %19 = affine.load %result_4[%arg5, %arg3] : memref<9x36xi32>
            %20 = arith.addi %19, %18 : i32
            affine.store %20, %result_4[%arg5, %arg3] : memref<9x36xi32>
            %21 = ADORA.interleaver %20, %19 : i32, i32 -> vector<2xi32>
            %22:2 = ADORA.deinterleaver %21 : vector<2xi32> -> (i32, i32)
            affine.vector_store %21, %arg2[%arg5 + 3, %arg3] : memref<?x36xi32>, vector<2xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "merge_MATMUL_4x4_IS"}
    %1 = ADORA.BlockStore async [%0, %asyncToken_5, %asyncToken_1, %asyncToken_3, %asyncToken] %result_4, %arg2 [3, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "3", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %2 = ADORA.BlockStore async [%0, %asyncToken_3, %asyncToken_1, %asyncToken_5, %asyncToken, %1] %result_2, %arg2 [1, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "2", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    %3 = ADORA.BlockStore async [%0, %asyncToken_1, %asyncToken_3, %asyncToken_5, %asyncToken, %1, %2] %result_0, %arg2 [2, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "1", KernelName = "merge_MATMUL_4x4_IS"} -> !ADORA.token
    ADORA.BlockStore async [%asyncToken, %0, %asyncToken_1, %asyncToken_3, %asyncToken_5, %1, %2, %3] %result, %arg2 [0, 0] : memref<9x36xi32> -> memref<?x36xi32>  {Id = "0", KernelName = "merge_MATMUL_4x4_IS"}
    return
  }
}


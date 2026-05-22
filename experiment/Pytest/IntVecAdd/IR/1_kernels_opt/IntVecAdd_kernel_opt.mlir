module attributes {adora.scheduled} {
  func.func @IntVecAdd(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<20xi32>  {Id = "0", KernelName = "IntVecAdd"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<20xi32>  {Id = "1", KernelName = "IntVecAdd"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<20xi32>  {Id = "2", KernelName = "IntVecAdd"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      affine.for %arg3 = 0 to 20 {
        %2 = affine.load %result[%arg3] : memref<20xi32>
        %3 = affine.load %result_0[%arg3] : memref<20xi32>
        %4 = arith.addi %2, %3 : i32
        affine.store %4, %0[%arg3] : memref<20xi32>
      }
      ADORA.terminator
    } {KernelName = "IntVecAdd"}
    ADORA.BlockStore async [%1] %0, %arg2 [0] : memref<20xi32> -> memref<?xi32>  {Id = "2", KernelName = "IntVecAdd"}
    return
  }
}


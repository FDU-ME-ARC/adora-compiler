module attributes {adora.scheduled} {
  func.func @fanin(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}]}]} {
    %result, %asyncToken = ADORA.BlockLoad async [] %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "fanin_k", stream = 0 : i32}
    %result_0, %asyncToken_1 = ADORA.BlockLoad async [] %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "fanin_k", stream = 1 : i32}
    %0 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "2", KernelName = "fanin_k"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      ADORA.terminator
    } {KernelName = "fanin_k", stream = 0 : i32}
    ADORA.BlockStore %0, %arg2 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "2", KernelName = "fanin_k"}
    return
  }
}


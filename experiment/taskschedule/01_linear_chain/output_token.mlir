module attributes {adora.scheduled} {
  func.func @linear_chain(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}]}]} {
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "linear_k"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "1", KernelName = "linear_k"}
    %1 = ADORA.kernel async [%asyncToken] {
      ADORA.terminator
    } {KernelName = "linear_k"}
    ADORA.BlockStore async [%1] %0, %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "linear_k"}
    return
  }
}


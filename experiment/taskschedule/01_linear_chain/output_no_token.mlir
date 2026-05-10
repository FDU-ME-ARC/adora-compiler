module attributes {adora.scheduled} {
  func.func @linear_chain(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = []}]} {
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "linear_k"}
    %0 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "1", KernelName = "linear_k"}
    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "linear_k"}
    ADORA.BlockStore %0, %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "linear_k"}
    return
  }
}


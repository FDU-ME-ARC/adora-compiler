module attributes {adora.scheduled} {
  func.func @fanin(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = []}]} {
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "fanin_k"}
    %result_0 = ADORA.BlockLoad %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "fanin_k"}
    %0 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "2", KernelName = "fanin_k"}
    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "fanin_k"}
    ADORA.BlockStore %0, %arg2 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "2", KernelName = "fanin_k"}
    return
  }
}


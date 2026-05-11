// Example 01: Linear chain — Load → Kernel → Store
//
// Dep graph:
//   BlockLoad(%A) → kernel → BlockStore(result → %C)
//
// Expected token chain (emit-token=true):
//   %res, %tok0 = ADORA.BlockLoad %A  → !ADORA.token
//   ADORA.kernel async [%tok0]
//   ADORA.BlockStore async [%tokK] %local, %C
//
// This is the simplest schedule case: one kernel fed by one load,
// writing to one store. No parallelism possible.

module {
  func.func @linear_chain(%arg0: memref<32x32xf32>,   // A (input)
                           %arg1: memref<32x32xf32>) { // C (output)
    %a_tile = ADORA.BlockLoad %arg0 [0, 0]
        : memref<32x32xf32> -> memref<32x32xf32>
        {Id = "0", KernelName = "linear_k"}

    %c_local = ADORA.LocalMemAlloc memref<32x32xf32>
        {Id = "1", KernelName = "linear_k"}

    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "linear_k"}

    ADORA.BlockStore %c_local, %arg1 [0, 0]
        : memref<32x32xf32> -> memref<32x32xf32>
        {Id = "1", KernelName = "linear_k"}

    return
  }
}

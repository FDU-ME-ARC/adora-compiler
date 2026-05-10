// Example 02: Fan-in — two parallel Loads → one Kernel → Store
//
// Dep graph:
//   BlockLoad(%A) ──┐
//                   ├→ kernel → BlockStore(result → %C)
//   BlockLoad(%B) ──┘
//
// Expected token chain (emit-token=true):
//   %a_res, %tok0 = ADORA.BlockLoad %A → !ADORA.token
//   %b_res, %tok1 = ADORA.BlockLoad %B → !ADORA.token
//   ADORA.kernel async [%tok0, %tok1]
//   ADORA.BlockStore async [%tokK] %local, %C
//
// The two loads are independent (different memrefs) → can run in parallel
// on different DMA streams. The kernel waits for both tokens.

module {
  func.func @fanin(%arg0: memref<32x32xf32>,   // A
                   %arg1: memref<32x32xf32>,   // B
                   %arg2: memref<32x32xf32>) { // C
    %a_tile = ADORA.BlockLoad %arg0 [0, 0]
        : memref<32x32xf32> -> memref<32x32xf32>
        {Id = "0", KernelName = "fanin_k"}

    %b_tile = ADORA.BlockLoad %arg1 [0, 0]
        : memref<32x32xf32> -> memref<32x32xf32>
        {Id = "1", KernelName = "fanin_k"}

    %c_local = ADORA.LocalMemAlloc memref<32x32xf32>
        {Id = "2", KernelName = "fanin_k"}

    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "fanin_k"}

    ADORA.BlockStore %c_local, %arg2 [0, 0]
        : memref<32x32xf32> -> memref<32x32xf32>
        {Id = "2", KernelName = "fanin_k"}

    return
  }
}

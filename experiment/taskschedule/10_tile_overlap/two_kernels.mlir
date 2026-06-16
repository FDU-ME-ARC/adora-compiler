// 10_tile_overlap/two_kernels.mlir
// Two INDEPENDENT kernels (disjoint memrefs => no cross-kernel data dependency).
//
// With a multi-tile ADG + the llm-pipeline-schedule pass, TileAssignment should
// place k0 and k1 on DIFFERENT tiles (round-robin / LLM) so they can overlap;
// the mapper then constrains each kernel's COMPUTE nodes (mulf/addf) to its
// assigned tile's GPEs.  IO nodes (Load/Store) keep their SPAD-bank constraints.
module {
  func.func @two_indep(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                       %arg2: memref<?xf32>, %arg3: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    // ---- kernel k0: reads arg0, writes arg1 ----
    %a0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32> {Id = "0", KernelName = "k0"}
    %o0 = ADORA.LocalMemAlloc memref<8xf32> {Id = "1", KernelName = "k0"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %a0[%i] : memref<8xf32>
        %1 = arith.mulf %0, %0 : f32
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %o0[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k0"}
    ADORA.BlockStore %o0, %arg1 [0] : memref<8xf32> -> memref<?xf32> {Id = "1", KernelName = "k0"}

    // ---- kernel k1: reads arg2, writes arg3 (disjoint from k0) ----
    %a1 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<8xf32> {Id = "2", KernelName = "k1"}
    %o1 = ADORA.LocalMemAlloc memref<8xf32> {Id = "3", KernelName = "k1"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %a1[%i] : memref<8xf32>
        %1 = arith.mulf %0, %0 : f32
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %o1[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k1"}
    ADORA.BlockStore %o1, %arg3 [0] : memref<8xf32> -> memref<?xf32> {Id = "3", KernelName = "k1"}

    return
  }
}

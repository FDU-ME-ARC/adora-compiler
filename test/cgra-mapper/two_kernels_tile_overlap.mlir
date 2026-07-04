// Two independent kernels (disjoint memrefs => no cross-kernel dep).
// With multi-tile ADG + adora-llm-pipeline-schedule, TileAssignment should place
// them on DIFFERENT tiles (round-robin / LLM) so they overlap; the mapper
// then constrains each kernel's compute nodes to its assigned tile.
//
// Used for end-to-end verification of Stage 2 (tile-based placement
// constraints in cgra-mapper). Not a lit test.
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

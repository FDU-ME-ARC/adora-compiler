// Test: TileAssignment writes adora.tile_set on each kernel (dry-run).
//
// TileAssignment runs as Step 0 of llm-pipeline-schedule.  In dry-run (no LLM)
// each kernel gets the conservative default tile_set = [0 .. minTiles-1], where
// minTiles = ceil(computeNodes / pe_per_tile).  Both kernels below have few
// compute ops, so minTiles = 1 and tile_set = [0].
//
// RUN: cgra-opt %s --llm-pipeline-schedule --llm-pipeline-schedule-dry-run \
// RUN:   --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16 \
// RUN:   2>/dev/null | FileCheck %s

module {
  func.func @two_kernels(%arg0: memref<?x25xf32>, %arg1: memref<?x25xf32>) {
    %l0 = "ADORA.BlockLoad"(%arg0)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    // The tile_set attr is printed in the kernel op's trailing attribute
    // dictionary (after the region), on the closing-brace line.
    // CHECK: KernelName = "k0", adora.tile_set = array<i64: 0>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k0"} : () -> ()
    "ADORA.BlockStore"(%l0, %arg0)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "1", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    %l1 = "ADORA.BlockLoad"(%arg1)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "2", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    // CHECK: KernelName = "k1", adora.tile_set = array<i64: 0>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k1"} : () -> ()
    "ADORA.BlockStore"(%l1, %arg1)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "3", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()
    return
  }
}

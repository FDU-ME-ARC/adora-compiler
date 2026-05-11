// PR4 commit A — fan-in: two root Loads feed one Store.
// Store inherits the minimum stream ID among its predecessors (min(0,1) = 0).
//
// Dep graph:  Load0(root) -tok0→ Store(dep)
//             Load1(root) -tok1→ Store(dep)
// Expected:   Load0.stream = 0,  Load1.stream = 1,  Store.stream = 0
//
// RUN: cgra-opt --adora-assign-streams --mlir-print-op-generic %s 2>/dev/null \
// RUN:   | FileCheck %s

// CHECK-LABEL: sym_name = "fan_in"

// First root → stream 0.
// CHECK: "ADORA.BlockLoad"
// CHECK: stream = 0 : i32

// Second root → stream 1.
// CHECK: "ADORA.BlockLoad"
// CHECK: stream = 1 : i32

// Fan-in consumer: inherits min(0, 1) = 0.
// CHECK: "ADORA.BlockStore"
// CHECK: stream = 0 : i32

module attributes {adora.scheduled} {
  func.func @fan_in(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>,
                    %arg2: memref<?x32xf32>) {
    // Root op 0: no asyncDeps → stream 0.
    %r0, %tok0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    // Root op 1: no asyncDeps → stream 1.
    %r1, %tok1 = "ADORA.BlockLoad"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "1", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    // Consumer: two asyncDeps (tok0, tok1) → operandSegmentSizes: src=1, tgt=1, indices=0, asyncDeps=2.
    // Inherits min(stream(Load0), stream(Load1)) = min(0, 1) = 0.
    "ADORA.BlockStore"(%r0, %arg2, %tok0, %tok1) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>}>
        {Id = "2", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<16x32xf32>, memref<?x32xf32>, !ADORA.token, !ADORA.token) -> ()
    return
  }
}

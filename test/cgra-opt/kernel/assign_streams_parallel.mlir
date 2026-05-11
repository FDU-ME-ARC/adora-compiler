// PR4 commit A — two independent root ops get distinct stream IDs.
//
// Dep graph:  Load0(root)   Load1(root)   [no edges between them]
// Expected:   Load0.stream = 0,  Load1.stream = 1
//
// BFS processes roots in program order: Load0 first (stream 0),
// Load1 second (stream 1).
//
// RUN: cgra-opt --adora-assign-streams --mlir-print-op-generic %s 2>/dev/null \
// RUN:   | FileCheck %s

// CHECK-LABEL: sym_name = "parallel_loads"

// First root → stream 0.
// CHECK: "ADORA.BlockLoad"
// CHECK: stream = 0 : i32

// Second root → stream 1.
// CHECK: "ADORA.BlockLoad"
// CHECK: stream = 1 : i32

module attributes {adora.scheduled} {
  func.func @parallel_loads(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>,
                             %arg2: memref<?x32xf32>) {
    // Root op 0: no asyncDeps → stream 0.
    %r0, %tok0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    // Root op 1: no asyncDeps → stream 1.
    %r1, %tok1 = "ADORA.BlockLoad"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "1", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    return
  }
}

// UNSUPPORTED: true
// NOTE: async-token backend pass deregistered; re-enable when re-registered.
// PR4 commit A — linear chain: Load → Store.
// Both ops share the same stream because Store inherits Load's stream.
//
// Dep graph:  Load(root) -tok0→ Store(dep)
// Expected:   Load.stream = 0,  Store.stream = 0   (inherited min)
//
// --mlir-print-op-generic guarantees all attributes appear in output.
//
// RUN: cgra-opt --adora-assign-streams --mlir-print-op-generic %s 2>/dev/null \
// RUN:   | FileCheck %s

// CHECK-LABEL: sym_name = "linear_chain"

// Load is a root → assigned stream 0.
// CHECK: "ADORA.BlockLoad"
// CHECK: stream = 0 : i32

// Store inherits stream 0 from its only predecessor.
// CHECK: "ADORA.BlockStore"
// CHECK: stream = 0 : i32

module attributes {adora.scheduled} {
  func.func @linear_chain(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>) {
    // Root op: produces tok0, no asyncDeps.
    %r0, %tok0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    // Dependent op: consumes tok0 → inherits stream 0.
    "ADORA.BlockStore"(%r0, %arg1, %tok0) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>}>
        {Id = "1", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<16x32xf32>, memref<?x32xf32>, !ADORA.token) -> ()
    return
  }
}

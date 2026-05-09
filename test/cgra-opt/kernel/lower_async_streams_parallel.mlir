// PR4 commit B — verify that adora-assign-streams stream IDs flow through
// adora-lower-async-tokens into signal/wait stream operands.
//
// Dep graph:  Load0(root,stream=0)   Load1(root,stream=1)   [no edges]
// Expected after lowering:
//   Load0 → ADORA.signal on stream 0
//   Load1 → ADORA.signal on stream 1
//   (no wait ops: neither op depends on the other)
//
// RUN: cgra-opt %s \
// RUN:   --adora-assign-streams \
// RUN:   --adora-lower-async-tokens \
// RUN:   2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @parallel_streams

// First root op (Load0) → event created, signal on stream 0.
// CHECK:      ADORA.event.create
// CHECK:      ADORA.BlockLoad
// CHECK-NEXT: ADORA.signal %{{.*}} on stream 0 : !ADORA.token

// Second root op (Load1) → separate event, signal on stream 1.
// CHECK:      ADORA.event.create
// CHECK:      ADORA.BlockLoad
// CHECK-NEXT: ADORA.signal %{{.*}} on stream 1 : !ADORA.token

// No cross-stream waits expected.
// CHECK-NOT:  ADORA.wait

module attributes {adora.scheduled} {
  func.func @parallel_streams(%arg0: memref<?x32xf32>,
                               %arg1: memref<?x32xf32>) {
    // Root op 0: no asyncDeps → assign-streams gives stream 0.
    %r0, %tok0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)

    // Root op 1: no asyncDeps → assign-streams gives stream 1.
    %r1, %tok1 = "ADORA.BlockLoad"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "1", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)

    return
  }
}

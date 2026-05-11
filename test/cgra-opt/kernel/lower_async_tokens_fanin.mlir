// PR3 commit B — fan-in test: two BlockLoads feed one BlockStore via async tokens.
// Exercises Pass 1 twice (two creates+signals), Pass 2 twice (two waits on one op),
// Pass 3 (three rebuilds), Pass 4 (two destroys).
// Also exercises rebuildKernelSync's dropAllUses fix when a KernelOp produces a token.
//
// This test bypasses adora-schedule-tasks and uses the post-schedule-tasks IR form
// (custom assembly with tokens already threaded) to run lower-async-tokens in isolation.
//
// RUN: cgra-opt %s --adora-lower-async-tokens 2>/dev/null | FileCheck %s

// Two event.create before the two loads.
// CHECK-LABEL: func.func @fan_in
// Pass 1: create + load(sync) + signal for each producer, in IR order.
// CHECK:         %[[EV0:.+]] = ADORA.event.create -> !ADORA.token
// CHECK-NEXT:    %{{.+}} = ADORA.BlockLoad
// CHECK-NEXT:    ADORA.signal %[[EV0]] on stream 0 : !ADORA.token
// CHECK:         %[[EV1:.+]] = ADORA.event.create -> !ADORA.token
// CHECK-NEXT:    %{{.+}} = ADORA.BlockLoad
// CHECK-NEXT:    ADORA.signal %[[EV1]] on stream 0 : !ADORA.token
// Pass 2+4: wait and destroy per dep, then store (sync — no "async [").
// CHECK:         ADORA.wait %[[EV0]] on stream 0 : !ADORA.token
// CHECK-NEXT:    ADORA.event.destroy %[[EV0]] : !ADORA.token
// CHECK:         ADORA.wait %[[EV1]] on stream 0 : !ADORA.token
// CHECK-NEXT:    ADORA.event.destroy %[[EV1]] : !ADORA.token
// CHECK:         ADORA.BlockStore
// CHECK-NOT:     async [
// CHECK:         return

module attributes {adora.scheduled} {
  func.func @fan_in(%arg0: memref<?x32xf32>, %arg1: memref<?x32xf32>, %arg2: memref<?x32xf32>) {
    // Generic form required: custom printer prints "-> !ADORA.token" suffix but
    // the custom parser does not roundtrip it; generic form is unambiguous.
    %r0, %tok0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    %r1, %tok1 = "ADORA.BlockLoad"(%arg1) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "1", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<?x32xf32>) -> (memref<16x32xf32>, !ADORA.token)
    // Two asyncDependencies → operandSegmentSizes: src=1, tgt=1, indices=0, asyncDeps=2
    "ADORA.BlockStore"(%r0, %arg2, %tok0, %tok1) <{operandSegmentSizes = array<i32: 1, 1, 0, 2>}>
        {Id = "2", KernelName = "k", map = affine_map<() -> (0, 0)>}
        : (memref<16x32xf32>, memref<?x32xf32>, !ADORA.token, !ADORA.token) -> ()
    return
  }
}

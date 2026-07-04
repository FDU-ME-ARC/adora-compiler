// UNSUPPORTED: true
// NOTE: async-token backend pass deregistered; re-enable when re-registered.
// PR3 commit B — verify --adora-lower-async-tokens lowers !ADORA.token to
// event.create / signal / wait / event.destroy.
//
// The input is the same minimal chain (BlockLoad -> Kernel -> BlockStore) used
// in schedule_cgra_tasks_tokens.mlir.  We run the full two-pass pipeline:
//   1. adora-schedule-tasks (emit-token=true)  → inserts async SSA tokens
//   2. adora-lower-async-tokens                → lowers to event ops
//
// RUN: cgra-opt %s \
// RUN:   --adora-schedule-tasks \
// RUN:   --adora-lower-async-tokens \
// RUN:   2>/dev/null | FileCheck %s

// CHECK-LABEL: func.func @chain
// CHECK:         %[[EV:.+]] = ADORA.event.create -> !ADORA.token
// CHECK-NEXT:    %{{.*}} = ADORA.BlockLoad
// CHECK-NEXT:    ADORA.signal %[[EV]] on stream 0 : !ADORA.token
// CHECK:         ADORA.wait  %[[EV]] on stream 0 : !ADORA.token
// CHECK:         ADORA.event.destroy %[[EV]] : !ADORA.token
// CHECK:         ADORA.BlockStore
// CHECK-NOT:     !ADORA.token
// CHECK:         return

module {
  func.func @chain(%arg0: memref<?x25xf32>, %arg1: memref<?x25xf32>, %arg2: memref<?x25xf32>) {
    %0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
         {Id = "0", KernelName = "chain_0", map = affine_map<() -> (0, 0)>}
         : (memref<?x25xf32>) -> memref<20x25xf32>

    "ADORA.kernel"() ({
      "ADORA.terminator"() : () -> ()
    }) {KernelName = "chain_0"} : () -> ()

    "ADORA.BlockStore"(%0, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
         {Id = "1", KernelName = "chain_0", map = affine_map<() -> (0, 0)>}
         : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    return
  }
}

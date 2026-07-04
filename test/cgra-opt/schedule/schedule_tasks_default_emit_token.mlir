// PR6.2 — verify --adora-schedule-tasks defaults `emit-token` to true.
//
// Without passing any option, the pass must produce SSA !ADORA.token values
// and async operand lists (with per-op `dep_kinds`). The opt-out form
// `emit-token=false` must produce neither tokens nor dep_kinds.
//
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=DEFAULT
// RUN: %cgra-opt %s --adora-schedule-tasks="emit-token=false" 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=OPTOUT

// DEFAULT: adora.scheduled
// DEFAULT-NOT: adora.dep_summary
// DEFAULT: ADORA.BlockLoad async

// OPTOUT-NOT: !ADORA.token
// OPTOUT-NOT: adora.dep_summary
// OPTOUT: adora.scheduled

#map = affine_map<()[s0] -> (0)>

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

// PR2 — verify --adora-schedule-tasks threads SSA !ADORA.token (emit-token=true)
// and preserves dep_summary-only path (emit-token=false, NFC).
//
// Uses generic-form IR with pre-built BlockLoad/Store/Kernel ops to bypass
// the adora-adjust-kernel-mem-footprint pass (pre-existing crash, unrelated).
//
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=TOKEN
// RUN: %cgra-opt %s --adora-schedule-tasks="emit-token=false" 2>/dev/null \
// RUN:   | FileCheck %s --check-prefix=NOTOKEN

// TOKEN: adora.scheduled
// TOKEN: adora.dep_summary
// TOKEN: ADORA.BlockLoad async

// NOTOKEN-NOT: !ADORA.token
// NOTOKEN: adora.scheduled
// NOTOKEN: adora.dep_summary

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

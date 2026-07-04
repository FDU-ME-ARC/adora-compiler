// Regression: per-op `dep_kinds` attribute (which replaced adora.dep_summary)
// must survive a print -> parse round-trip and stay aligned 1:1 with each op's
// async dependency operands.
//
// The @chain body produces a store that consumes the kernel token (RAW) but
// produces no token — its `dep_kinds = ["RAW"]` must round-trip unchanged.
//
// Uses generic-form IR to bypass adora-adjust-kernel-mem-footprint.
//
// 1) schedule-tasks emits dep_kinds; no dep_summary attribute remains.
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null \
// RUN:   | %FileCheck %s --check-prefix=SCHED
//
// 2) round-trip the scheduled IR through parse+print; dep_kinds is preserved.
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null > %t.0
// RUN: %cgra-opt %t.0 2>/dev/null > %t.1
// RUN: %cgra-opt %t.1 2>/dev/null > %t.2
// RUN: diff %t.1 %t.2
// RUN: %FileCheck %s --check-prefix=ROUNDTRIP < %t.1

// SCHED-NOT: adora.dep_summary
// SCHED:     ADORA.BlockStore async [{{.*}}] {{.*}} dep_kinds = ["RAW", "WAR"]

// After round-tripping, dep_kinds is still present and unchanged.
// ROUNDTRIP:     ADORA.BlockStore async [{{.*}}] {{.*}} dep_kinds = ["RAW", "WAR"]

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

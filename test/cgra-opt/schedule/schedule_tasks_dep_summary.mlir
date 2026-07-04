// End-to-end test: verify per-op `dep_kinds` attribute carries RAR/WAR/WAW
// dependency kinds (from analyzeDependencyInGraph), aligned with async deps.
// (Replaces the retired adora.dep_summary attribute channel.)
//
// Uses generic-form IR with pre-built BlockLoad/Store/Kernel ops to bypass
// adora-adjust-kernel-mem-footprint (pre-existing crash, unrelated).
//
// RUN: cgra-opt %s --adora-schedule-tasks 2>/dev/null | FileCheck %s

// CHECK: adora.scheduled
// CHECK-NOT: adora.dep_summary
// dep kinds now live on each async op's `dep_kinds` attribute, aligned 1:1
// with its async dependency operands.
// CHECK-DAG: dep_kinds = ["RAR"]
// CHECK-DAG: dep_kinds = ["RAW", "WAR", "WAR"]
// CHECK-DAG: dep_kinds = ["RAW", "WAW", "WAR", "WAR"]

module {
  // Three loads and one store on the same underlying memref: produces
  //   RAR(load0 -> load1), WAR(load0 -> store3), WAW(store0 -> store3)
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>,
                      %arg2: memref<?x25xf32>) {
    // load0: read arg0 (will have RAR with load1, WAR with store3)
    %l0 = "ADORA.BlockLoad"(%arg0)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "gemm_opt_0",
         map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>

    // load1: read arg1 (RAR with load0 via overlap of output memref arg0)
    %l1 = "ADORA.BlockLoad"(%arg0)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "1", KernelName = "gemm_opt_0",
         map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>

    // load2: read arg2
    %l2 = "ADORA.BlockLoad"(%arg2)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "2", KernelName = "gemm_opt_0",
         map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>

    "ADORA.kernel"() ({
      "ADORA.terminator"() : () -> ()
    }) {KernelName = "gemm_opt_0"} : () -> ()

    // store3: write arg0 (WAR with load0/load1, WAW with hypothetical earlier store)
    "ADORA.BlockStore"(%l0, %arg0)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "3", KernelName = "gemm_opt_0",
         map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    // store4: also write arg0 (WAW with store3)
    "ADORA.BlockStore"(%l1, %arg0)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "4", KernelName = "gemm_opt_0",
         map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    return
  }
}

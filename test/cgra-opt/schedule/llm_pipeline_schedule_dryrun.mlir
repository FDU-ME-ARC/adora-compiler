// Test for LLMPipelineSchedule pass in dry-run mode (no LLM call, deterministic).
//
// In dry-run the pass always selects plan idx 0 (conservative serial schedule),
// so every task after the first gets hw_dep_type = "LD_DEP_ST_LAST_TASK" on both
// its load and store.  The first task has no predecessor and stays unannotated.
//
// RUN: cgra-opt %s --llm-pipeline-schedule --llm-pipeline-schedule-dry-run 2>/dev/null | FileCheck %s

module {
  func.func @two_tasks(%arg0: memref<?x25xf32>, %arg1: memref<?x25xf32>) {
    // task 0: first task, no predecessor -> no hw_dep_type expected.
    // CHECK: ADORA.BlockLoad %arg0
    // CHECK-NOT: hw_dep_type
    %l0 = "ADORA.BlockLoad"(%arg0)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "0", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k0"} : () -> ()
    "ADORA.BlockStore"(%l0, %arg0)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "1", KernelName = "k0", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    // task 1: gets the dry-run default dep_type on both load and store.
    // CHECK: ADORA.BlockLoad %arg1 {{.*}}hw_dep_type = "LD_DEP_ST_LAST_TASK"
    %l1 = "ADORA.BlockLoad"(%arg1)
        <{operandSegmentSizes = array<i32: 1, 0, 0>}>
        {Id = "2", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<?x25xf32>) -> memref<20x25xf32>
    "ADORA.kernel"() ({ "ADORA.terminator"() : () -> () }) {KernelName = "k1"} : () -> ()
    // CHECK: ADORA.BlockStore {{.*}}hw_dep_type = "LD_DEP_ST_LAST_TASK"
    "ADORA.BlockStore"(%l1, %arg1)
        <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
        {Id = "3", KernelName = "k1", map = affine_map<() -> (0, 0)>}
        : (memref<20x25xf32>, memref<?x25xf32>) -> ()
    return
  }
}

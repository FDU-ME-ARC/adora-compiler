// Test: fan-in async-token dependency (AND-join) surfaces in the emitted
// pytest as two concurrent tasks gathered together — the schedule-derived
// AND-join and the generated Python concurrency MUST match.
//
// Topology (fan-in / AND-join):
//   k0: ... -> BlockStore %a0, arg1   (produces token t0)
//   k1: ... -> BlockStore %a1, arg2   (produces token t1)
//   k2: BlockLoad arg1 (RAW on k0), BlockLoad arg2 (RAW on k1) -> ...
//       k2 depends on BOTH t0 and t1.
//
// schedule-tasks threads t0 and t1 into k2's two loads. Both producing stores
// have downstream consumers, so EmitPytest launches each as a task and makes
// k2 gather on both — the two-token AND-join becomes gather(task_1, task_0).
//
// CHECK-DAG throughout: placement search order is not fixed across runs.
//
// RUN: rm -rf %t && mkdir -p %t
//
// -- pytest: both producer stores become tasks; consumer gathers both --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=pytest --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/fanin_py.py
// RUN: test -s %t/fanin_py.py
// RUN: %FileCheck %s --check-prefix=CHECK-PY --input-file=%t/fanin_py.py

// Two producer stores each launch a task; the AND-join consumer gathers both.
// CHECK-PY-DAG: task_0 = asyncio.create_task(
// CHECK-PY-DAG: task_1 = asyncio.create_task(
// CHECK-PY: await asyncio.gather(task_{{[0-9]+}}, task_{{[0-9]+}})

module {
  func.func @fanin(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                   %arg2: memref<?xf32>, %arg3: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    // k0: a0 = arg0 * arg0; store -> arg1 (token t0)
    %l0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k0"}
    %a0 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "1", KernelName = "k0"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l0[%i] : memref<8xf32>
        %2 = arith.mulf %0, %0 : f32
        affine.store %2, %a0[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k0"}
    ADORA.BlockStore %a0, %arg1 [0] : memref<8xf32> -> memref<?xf32>  {Id = "1", KernelName = "k0"}

    // k1: a1 = arg2 * arg2; store -> arg2 (token t1)
    %l1 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k1"}
    %a1 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "1", KernelName = "k1"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l1[%i] : memref<8xf32>
        %2 = arith.mulf %0, %0 : f32
        affine.store %2, %a1[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k1"}
    ADORA.BlockStore %a1, %arg2 [0] : memref<8xf32> -> memref<?xf32>  {Id = "1", KernelName = "k1"}

    // k2: reads arg1 (RAW on k0) AND arg2 (RAW on k1) -> two-token AND-join
    %l2 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k2"}
    %l3 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<8xf32>  {Id = "1", KernelName = "k2"}
    %a2 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "2", KernelName = "k2"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l2[%i] : memref<8xf32>
        %1 = affine.load %l3[%i] : memref<8xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %a2[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k2"}
    ADORA.BlockStore %a2, %arg3 [0] : memref<8xf32> -> memref<?xf32>  {Id = "2", KernelName = "k2"}

    return
  }
}

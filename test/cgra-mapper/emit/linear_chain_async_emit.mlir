// Test: async-token dependency across two kernels surfaces in the emitted
// pytest as an asyncio task graph — i.e. the schedule-derived token
// dependency and the generated Python concurrency MUST match.
//
// Topology (linear chain):
//   k0: BlockLoad arg0, arg1 -> kernel(mul) -> BlockStore %b0, arg1   (produces token)
//   k1: BlockLoad arg1 (RAW on k0's store), arg0 -> kernel(add) -> BlockStore arg2
//
// schedule-tasks threads a token from k0's store into k1's load; because k0's
// store now has a downstream consumer, EmitPytest launches it as a task and
// makes k1 gather on it. This is exactly the "dependency <-> pytest" match we
// want to lock down.
//
// CHECK-DAG is used because the mapper's placement search does not fix the
// emission order across runs.
//
// RUN: rm -rf %t && mkdir -p %t
//
// -- schedule dump: cross-kernel token dep is present (C path shows async) --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=c --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/lc_c.c
// RUN: test -s %t/lc_c.c
// RUN: %FileCheck %s --check-prefix=CHECK-C --input-file=%t/lc_c.c
//
// -- pytest: token dep becomes create_task + gather --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=pytest --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/lc_py.py
// RUN: test -s %t/lc_py.py
// RUN: %FileCheck %s --check-prefix=CHECK-PY --input-file=%t/lc_py.py

// The producing k0 store carries a dep flag; the consuming k1 waits on it.
// CHECK-C-DAG: BlockStore async [%
// CHECK-C-DAG: EX_DEP_ST_LAST_TASK

// The k0 store (with a downstream consumer) launches as a task; k1 gathers it.
// CHECK-PY-DAG: task_{{[0-9]+}} = asyncio.create_task(
// CHECK-PY-DAG: await asyncio.gather(task_

module {
  func.func @two_kernel(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                        %arg2: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    // k0: b0 = arg0 * arg1; store b0 -> arg1 (this store produces a token)
    %l0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k0"}
    %l1 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<8xf32>  {Id = "1", KernelName = "k0"}
    %b0 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "2", KernelName = "k0"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l0[%i] : memref<8xf32>
        %1 = affine.load %l1[%i] : memref<8xf32>
        %2 = arith.mulf %0, %1 : f32
        affine.store %2, %b0[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k0"}
    ADORA.BlockStore %b0, %arg1 [0] : memref<8xf32> -> memref<?xf32>  {Id = "2", KernelName = "k0"}

    // k1: reads arg1 (RAW dep on k0's store) + arg0; b1 = arg1 + arg0
    %l2 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k1"}
    %l3 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32>  {Id = "1", KernelName = "k1"}
    %b1 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "2", KernelName = "k1"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l2[%i] : memref<8xf32>
        %1 = affine.load %l3[%i] : memref<8xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %b1[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k1"}
    ADORA.BlockStore %b1, %arg2 [0] : memref<8xf32> -> memref<?xf32>  {Id = "2", KernelName = "k1"}

    return
  }
}

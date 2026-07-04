// Test: a terminal BlockStore that CONSUMES the kernel token but produces no
// token (writes an output nobody reads back) must, in the emitted code:
//   - C:      still carry the execute() dependency flag (it waits on the kernel)
//   - pytest: NOT launch an asyncio task (no downstream consumer) — it runs via
//             a plain `await aux_stream(...)`, no create_task / gather.
//
// This is the "consume-but-not-produce" store shape (async [%tok] with no
// `-> !ADORA.token` result). It is the counterpart of linear_chain_async_emit
// (where the store DOES have a consumer and therefore becomes a task).
//
// CHECK-DAG where order is not fixed; CHECK-NOT asserts the absence of tasks.
//
// RUN: rm -rf %t && mkdir -p %t
//
// -- C: terminal store still emits the kernel-dep execute() flag --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=c --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/term_c.c
// RUN: test -s %t/term_c.c
// RUN: %FileCheck %s --check-prefix=CHECK-C --input-file=%t/term_c.c
//
// -- pytest: terminal store runs via await, NOT create_task --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=pytest --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/term_py.py
// RUN: test -s %t/term_py.py
// RUN: %FileCheck %s --check-prefix=CHECK-PY --input-file=%t/term_py.py

// C: the terminal store waits on the kernel (dependency flag present).
// CHECK-C: EX_DEP_ST_LAST_TASK

// pytest: a plain await, and NO asyncio task is created for this terminal store.
// CHECK-PY: await aux_stream(
// CHECK-PY-NOT: asyncio.create_task(

module {
  func.func @terminal(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                      %arg2: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %l0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "k0"}
    %l1 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<8xf32>  {Id = "1", KernelName = "k0"}
    %b0 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "2", KernelName = "k0"}
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %0 = affine.load %l0[%i] : memref<8xf32>
        %1 = affine.load %l1[%i] : memref<8xf32>
        %2 = arith.addf %0, %1 : f32
        affine.store %2, %b0[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "k0"}
    // Terminal store: consumes the kernel token, writes output arg2, and no
    // later op reads arg2 -> consumes but does NOT produce a token.
    ADORA.BlockStore %b0, %arg2 [0] : memref<8xf32> -> memref<?xf32>  {Id = "2", KernelName = "k0"}
    return
  }
}

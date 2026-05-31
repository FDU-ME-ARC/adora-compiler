// Test: cgra-mapper --enable-async drives the schedule-tasks +
// assign-streams + lower-async-tokens path before emit, so the
// schedule-derived token dependencies surface in the generated code.
//
// Same kernel as mvt_small_emit.mlir, but here we assert the async-token
// effects that are ABSENT in the non-async baseline:
//   C:      BlockLoad/BlockStore carry `async`, and the store dependency
//           lowers to an actual execute(...) call with EX_DEP_ST_LAST_TASK.
//   pytest: imports the CGRA-Cocotb-Sim runtime (test_runif) and threads the
//           token as a depend_type=2 argument on the H2D copy.
//
// CHECK-DAG is used throughout because the mapper's placement search does not
// fix the emission order across runs.
//
// RUN: rm -rf %t && mkdir -p %t
//
// -- EmitCGRACall with async tokens (--output-type=c --enable-async) --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=c --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/emit_async_c.c
// RUN: test -s %t/emit_async_c.c
// RUN: %FileCheck %s --check-prefix=CHECK-ASYNC-C --input-file=%t/emit_async_c.c
//
// -- EmitPytest with async tokens (--output-type=pytest --enable-async) --
// RUN: %cgra-mapper --enable-async \
// RUN:   --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=pytest --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/emit_async_pytest.py
// RUN: test -s %t/emit_async_pytest.py
// RUN: %FileCheck %s --check-prefix=CHECK-ASYNC-PY --input-file=%t/emit_async_pytest.py
//
// Loads carry an async token; the store waits on a kernel token; the token
// dependency lowers to an actual execute() call carrying the store dep flag.
// CHECK-ASYNC-C-DAG: ADORA.BlockLoad async [] %arg0
// CHECK-ASYNC-C-DAG: ADORA.BlockLoad async [] %arg1
// CHECK-ASYNC-C-DAG: ADORA.BlockStore async [%{{.*}}]
// CHECK-ASYNC-C-DAG: execute({{.*}}, _task_id, EX_DEP_ST_LAST_TASK)
//
// Pytest targets the simulator runtime and threads the token as depend_type.
// CHECK-ASYNC-PY-DAG: from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
// CHECK-ASYNC-PY-DAG: memcpyHostToDevice({{.*}}depend_type=2)
// CHECK-ASYNC-PY-DAG: ADORA.BlockStore async [%{{.*}}]

module {
  func.func @kernel_scale_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                               %arg2: memref<?xf32>, %arg3: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {

    // Plain loop before kernel: initialize arg3 = arg0 + arg1.
    affine.for %i = 0 to 8 {
      %a = affine.load %arg0[%i] : memref<?xf32>
      %b = affine.load %arg1[%i] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %arg3[%i] : memref<?xf32>
    }

    // CGRA-mapped kernel: fused mulf+addf into two LocalMemAlloc buffers.
    %result   = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<8xf32>  {Id = "0", KernelName = "kernel_scale_add"}
    %result_0 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<8xf32>  {Id = "1", KernelName = "kernel_scale_add"}
    %out0 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "2", KernelName = "kernel_scale_add"}
    %out1 = ADORA.LocalMemAlloc memref<8xf32>  {Id = "3", KernelName = "kernel_scale_add"}
    ADORA.kernel {
      affine.for %arg4 = 0 to 8 {
        %0 = affine.load %result[%arg4]   : memref<8xf32>
        %1 = affine.load %result_0[%arg4] : memref<8xf32>
        %2 = arith.mulf %0, %1 : f32
        %3 = arith.addf %0, %2 : f32
        affine.store %3, %out0[%arg4] : memref<8xf32>
        %4 = arith.mulf %1, %2 : f32
        affine.store %4, %out1[%arg4] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_scale_add"}
    ADORA.BlockStore %out0, %arg2 [0] : memref<8xf32> -> memref<?xf32>  {Id = "2", KernelName = "kernel_scale_add"}
    ADORA.BlockStore %out1, %arg2 [0] : memref<8xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_scale_add"}

    // Plain loop after kernel: accumulate kernel output into arg3.
    affine.for %i = 0 to 8 {
      %a = affine.load %arg2[%i] : memref<?xf32>
      %b = affine.load %arg3[%i] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %arg3[%i] : memref<?xf32>
    }

    return
  }
}

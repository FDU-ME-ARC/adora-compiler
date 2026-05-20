// Test: cgra-mapper full pipeline (schedule + map + emit) with array AffineStoreOp,
// covering both ADORA.kernel-internal stores and plain-loop stores outside the kernel.
//
// Structure:
//   [plain loop 1]  arg3[i] = arg0[i] + arg1[i]           — non-kernel array store
//   ADORA.BlockLoad %arg0, ADORA.BlockLoad %arg1
//   ADORA.LocalMemAlloc (x2)
//   ADORA.kernel { fused mulf+addf, 2 stores to LocalMemAlloc } — kernel array stores
//   ADORA.BlockStore (x2) -> %arg2, %arg2 (output)
//   [plain loop 2]  arg3[i] = arg2[i] + arg3[i]           — non-kernel array store
//
// Bug: before fix, --output-type=c/sdk/pytest crashed (SIGABRT) on
//      affine.store to any array memref (shape.size() != 0); assert replaced by
//      if/else emitting arr[idx] = val; (EmitCGRACall.cpp, EmitVitisSDK.cpp,
//      EmitPytest.cpp — already fixed as reference).
//
// RUN: rm -rf %t && mkdir -p %t
//
// -- EmitCGRACall (--output-type=c) --
// RUN: %cgra-mapper --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=c --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/emit_c.c
// RUN: test -s %t/emit_c.c
// RUN: %FileCheck %s --check-prefix=CHECK-CGRA --input-file=%t/emit_c.c
//
// -- EmitVitisSDK (--output-type=sdk) --
// RUN: %cgra-mapper --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=sdk --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/emit_sdk.c
// RUN: test -s %t/emit_sdk.c
// RUN: %FileCheck %s --check-prefix=CHECK-SDK --input-file=%t/emit_sdk.c
//
// -- EmitPytest (--output-type=pytest) --
// RUN: %cgra-mapper --adg=%S/../../spec/cgra_fp32/cgra_adg_fp32.json \
// RUN:   --op-file=%S/../../spec/cgra_fp32/operations_fp32.json \
// RUN:   --output-type=pytest --obj-opt=true --max-iters=1 \
// RUN:   %s --output=%t/emit_pytest.py
// RUN: test -s %t/emit_pytest.py
// RUN: %FileCheck %s --check-prefix=CHECK-PYTEST --input-file=%t/emit_pytest.py
//
// Verify array-indexed stores (arr[idx] = val) appear in all three backends.
// The emitter uses underscored names (arg_3 for %arg3).
// CHECK-CGRA: kernel_scale_add
// CHECK-CGRA: arg_3[
//
// CHECK-SDK: void kernel_scale_add
// CHECK-SDK: HostToDeviceTransfer
// CHECK-SDK: DeviceToHostTransfer
// CHECK-SDK: arg_3[
//
// CHECK-PYTEST: async def kernel_scale_add
// CHECK-PYTEST: arg_3[

module {
  func.func @kernel_scale_add(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                               %arg2: memref<?xf32>, %arg3: memref<?xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {

    // Plain loop before kernel (no ADORA.kernel): initialize arg3 = arg0 + arg1.
    // Tests non-kernel array AffineStoreOp emit: arg3[i] = ...
    affine.for %i = 0 to 8 {
      %a = affine.load %arg0[%i] : memref<?xf32>
      %b = affine.load %arg1[%i] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %arg3[%i] : memref<?xf32>
    }

    // CGRA-mapped kernel: fused mulf+addf, reads BlockLoad results,
    // writes to two LocalMemAlloc buffers (1D array stores inside ADORA.kernel).
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

    // Plain loop after kernel (no ADORA.kernel): accumulate kernel output into arg3.
    // Tests a second non-kernel array AffineStoreOp emit: arg3[i] = ...
    affine.for %i = 0 to 8 {
      %a = affine.load %arg2[%i] : memref<?xf32>
      %b = affine.load %arg3[%i] : memref<?xf32>
      %c = arith.addf %a, %b : f32
      affine.store %c, %arg3[%i] : memref<?xf32>
    }

    return
  }
}

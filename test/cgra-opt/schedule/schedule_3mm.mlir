// Verify adora-schedule-tasks correctly threads !ADORA.token for inter-kernel
// RAW dependencies in the 3mm polybench kernel.
//
// Dep graph:
//   kernel_3mm_0 (E=A×B, writes %arg0)  ─┐
//                                          ├─▶ kernel_3mm_2 (out=E×G)
//   kernel_3mm_1 (G=D×E, writes %arg3)  ─┘
//
// kernel_3mm_0 and kernel_3mm_1 are independent root tasks.
// kernel_3mm_2 reads %arg0 and %arg3 → RAW deps on both stores.
//
// RUN: cgra-opt %s --adora-schedule-tasks="emit-token=true" | FileCheck %s

// Root task stores produce tokens:
// CHECK: ADORA.BlockStore {{.*}}, %arg0 {{.*}} -> !ADORA.token
// CHECK: ADORA.BlockStore {{.*}}, %arg3 {{.*}} -> !ADORA.token

// kernel_3mm_2 loads consume the tokens (RAW dep):
// CHECK: ADORA.BlockLoad async [%{{[0-9]+}}] %arg0
// CHECK: ADORA.BlockLoad async [%{{[0-9]+}}] %arg3

// Root task loads have NO async deps:
// CHECK-NOT: ADORA.BlockLoad async {{.*}} %arg1
// CHECK-NOT: ADORA.BlockLoad async {{.*}} %arg4

module {
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>,
                         %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>,
                         %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>,
                         %arg6: memref<?x22xf32>) {
    %cst = arith.constant 0.000000e+00 : f32

    // task_0: E = A × B  (writes %arg0)
    %0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0"}
    %1 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0"}
    %2 = ADORA.LocalMemAlloc memref<16x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    ADORA.kernel {
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 18 {
          %r = affine.for %k = 0 to 20 iter_args(%acc = %cst) -> (f32) {
            %a = affine.load %0[%i, %k] : memref<16x20xf32>
            %b = affine.load %1[%k, %j] : memref<20x18xf32>
            %p = arith.mulf %a, %b : f32
            %s = arith.addf %acc, %p : f32
            affine.yield %s : f32
          }
          affine.store %r, %2[%i, %j] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0"}
    ADORA.BlockStore %2, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}

    // task_1: G = D × E  (writes %arg3, independent of task_0)
    %3 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1"}
    %4 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1"}
    %5 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    ADORA.kernel {
      affine.for %i = 0 to 18 {
        affine.for %j = 0 to 22 {
          %r = affine.for %k = 0 to 24 iter_args(%acc = %cst) -> (f32) {
            %a = affine.load %3[%i, %k] : memref<18x24xf32>
            %b = affine.load %4[%k, %j] : memref<24x22xf32>
            %p = arith.mulf %a, %b : f32
            %s = arith.addf %acc, %p : f32
            affine.yield %s : f32
          }
          affine.store %r, %5[%i, %j] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1"}
    ADORA.BlockStore %5, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}

    // task_2: out = E × G  (RAW: reads %arg0 from task_0, %arg3 from task_1)
    %6 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x18xf32> -> memref<16x18xf32>  {Id = "0", KernelName = "kernel_3mm_2"}
    %7 = ADORA.BlockLoad %arg3 [0, 0] : memref<?x22xf32> -> memref<18x22xf32>  {Id = "1", KernelName = "kernel_3mm_2"}
    %8 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    ADORA.kernel {
      affine.for %i = 0 to 16 {
        affine.for %j = 0 to 22 {
          %r = affine.for %k = 0 to 18 iter_args(%acc = %cst) -> (f32) {
            %a = affine.load %6[%i, %k] : memref<16x18xf32>
            %b = affine.load %7[%k, %j] : memref<18x22xf32>
            %p = arith.mulf %a, %b : f32
            %s = arith.addf %acc, %p : f32
            affine.yield %s : f32
          }
          affine.store %r, %8[%i, %j] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2"}
    ADORA.BlockStore %8, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    return
  }
}

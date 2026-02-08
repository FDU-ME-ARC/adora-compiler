// Test that adoracc skips adora-extract-affine-for-to-kernel when input already has ADORA.kernel.
// Same as mvt.mlir, but the first loop nest is already in ADORA.kernel form; second loop stays affine.for.
//
// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t -o %t/result.mlir
// RUN: FileCheck %s --input-file=%t/result.mlir
//
// CHECK: module {
// CHECK: func.func @kernel_mvt(
// CHECK: ADORA.BlockLoad
// CHECK: ADORA.BlockLoad
// CHECK: ADORA.BlockLoad
// CHECK: ADORA.LocalMemAlloc
// CHECK: ADORA.kernel {
// CHECK: affine.for
// CHECK: affine.for
// CHECK: ADORA.terminator
// CHECK: } {KernelName = "kernel_mvt_0"}
// CHECK: ADORA.BlockStore
// CHECK: affine.for
// CHECK: affine.for
// CHECK: return
// CHECK: }

module attributes {} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32> {Id = "0", KernelName = "kernel_mvt_0"}
    %1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32> {Id = "1", KernelName = "kernel_mvt_0"}
    %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32> {Id = "2", KernelName = "kernel_mvt_0"}
    %3 = ADORA.LocalMemAlloc memref<40xf32> {Id = "3", KernelName = "kernel_mvt_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %4 = affine.load %0[%arg5] : memref<40xf32>
        %5 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %4) -> (f32) {
          %6 = affine.load %1[%arg5, %arg6] : memref<40x40xf32>
          %7 = affine.load %2[%arg6] : memref<40xf32>
          %8 = arith.mulf %6, %7 : f32
          %9 = arith.addf %arg7, %8 : f32
          affine.yield %9 : f32
        }
        affine.store %5, %3[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.BlockStore %3, %arg0 [0] : memref<40xf32> -> memref<?xf32> {Id = "3", KernelName = "kernel_mvt_0"}
    affine.for %arg5 = 0 to 40 {
      affine.for %arg6 = 0 to 40 {
        %10 = affine.load %arg1[%arg5] : memref<?xf32>
        %11 = affine.load %arg4[%arg6, %arg5] : memref<?x40xf32>
        %12 = affine.load %arg3[%arg6] : memref<?xf32>
        %13 = arith.mulf %11, %12 : f32
        %14 = arith.addf %10, %13 : f32
        affine.store %14, %arg1[%arg5] : memref<?xf32>
      }
    }
    return
  }
}
module attributes {} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          %0 = affine.load %arg0[%arg5] : memref<?xf32>
          %1 = affine.load %arg4[%arg5, %arg6] : memref<?x40xf32>
          %2 = affine.load %arg2[%arg6] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %0, %3 : f32
          affine.store %4, %arg0[%arg5] : memref<?xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    affine.for %arg5 = 0 to 40 {
      affine.for %arg6 = 0 to 40 {
        %0 = affine.load %arg1[%arg5] : memref<?xf32>
        %1 = affine.load %arg4[%arg6, %arg5] : memref<?x40xf32>
        %2 = affine.load %arg3[%arg6] : memref<?xf32>
        %3 = arith.mulf %1, %2 : f32
        %4 = arith.addf %0, %3 : f32
        affine.store %4, %arg1[%arg5] : memref<?xf32>
      }
    }
    return
  }
}
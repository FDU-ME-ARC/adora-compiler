// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t -o %t/result.mlir
// RUN: FileCheck %s --input-file=%t/result.mlir
//
// This test checks that sibling loop nests (two loop nests under the same outer
// time-step loop) are extracted as two separate kernels.
//
// CHECK: module
// CHECK: func.func @jacobi_2d(
// CHECK: affine.for %{{.*}} = 0 to 10
// CHECK: ADORA.kernel
// CHECK: } {KernelName = "jacobi_2d_0"}
// CHECK: ADORA.kernel
// CHECK: } {KernelName = "jacobi_2d_1"}
module attributes {} {
  func.func @jacobi_2d(%arg0: memref<?x30xi32>, %arg1: memref<?x30xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 1 to 29 {
        affine.for %arg4 = 1 to 29 {
          %0 = affine.load %arg0[%arg3, %arg4] : memref<?x30xi32>
          %1 = affine.load %arg0[%arg3, %arg4 - 1] : memref<?x30xi32>
          %2 = arith.addi %0, %1 : i32
          %3 = affine.load %arg0[%arg3, %arg4 + 1] : memref<?x30xi32>
          %4 = arith.addi %2, %3 : i32
          %5 = affine.load %arg0[%arg3 + 1, %arg4] : memref<?x30xi32>
          %6 = arith.addi %4, %5 : i32
          %7 = affine.load %arg0[%arg3 - 1, %arg4] : memref<?x30xi32>
          %8 = arith.addi %6, %7 : i32
          %9 = arith.divsi %8, %c5_i32 : i32
          affine.store %9, %arg1[%arg3, %arg4] : memref<?x30xi32>
        }
      }
      affine.for %arg3 = 1 to 29 {
        affine.for %arg4 = 1 to 29 {
          %0 = affine.load %arg1[%arg3, %arg4] : memref<?x30xi32>
          %1 = affine.load %arg1[%arg3, %arg4 - 1] : memref<?x30xi32>
          %2 = arith.addi %0, %1 : i32
          %3 = affine.load %arg1[%arg3, %arg4 + 1] : memref<?x30xi32>
          %4 = arith.addi %2, %3 : i32
          %5 = affine.load %arg1[%arg3 + 1, %arg4] : memref<?x30xi32>
          %6 = arith.addi %4, %5 : i32
          %7 = affine.load %arg1[%arg3 - 1, %arg4] : memref<?x30xi32>
          %8 = arith.addi %6, %7 : i32
          %9 = arith.divsi %8, %c5_i32 : i32
          affine.store %9, %arg0[%arg3, %arg4] : memref<?x30xi32>
        }
      }
    }
    return
  }
}

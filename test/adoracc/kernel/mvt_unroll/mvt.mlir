// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t --enable-unroll --adg-path %S/../../../spec/cgra_fp32/cgra_adg_fp32.json  -o %t/result.mlir
// RUN: FileCheck %s --input-file=%t/result.mlir
//
// CHECK: module
// CHECK: func.func @kernel_mvt(
// CHECK: %[[A:.*]], %{{.*}} = ADORA.BlockLoad async {{.*}} %arg0 [0] : memref<?xf32> -> memref<40xf32>
// CHECK: %[[M:.*]], %{{.*}} = ADORA.BlockLoad async {{.*}} %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>
// CHECK: %[[X:.*]], %{{.*}} = ADORA.BlockLoad async {{.*}} %arg2 [0] : memref<?xf32> -> memref<40xf32>
// CHECK: %[[OUT0:.*]] = ADORA.LocalMemAlloc memref<40xf32>
// CHECK: ADORA.kernel
// CHECK: affine.for %[[I:.*]] = 0 to 40 {
// CHECK: %[[A_I:.*]] = affine.load %[[A]]{{\[}}%[[I]]{{\]}} : memref<40xf32>
// CHECK: %[[RED:.*]] = affine.for %[[J:.*]] = 0 to 40 step 5 iter_args(%[[ACC:.*]] = %[[A_I]]) -> (f32) {
// CHECK: affine.load %[[M]]{{\[}}%[[I]], %[[J]]{{\]}} : memref<40x40xf32>
// CHECK: affine.load %[[X]]{{\[}}%[[J]]{{\]}} : memref<40xf32>
// CHECK: arith.mulf
// CHECK: arith.addf {{.*}}%[[ACC]]
// CHECK: affine.load %[[M]]
// CHECK: affine.load %[[X]]
// CHECK: arith.mulf
// CHECK: arith.addf
// CHECK: affine.yield {{.*}} : f32
// CHECK: }
// CHECK: affine.store %[[RED]], %[[OUT0]]{{\[}}%[[I]]{{\]}} : memref<40xf32>
// CHECK: }
// CHECK: ADORA.terminator
// CHECK: } {KernelName = "kernel_mvt_0"
// CHECK: ADORA.BlockStore {{.*}}%[[OUT0]], %arg0 [0] : memref<40xf32> -> memref<?xf32>
// CHECK: }

module attributes {} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
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

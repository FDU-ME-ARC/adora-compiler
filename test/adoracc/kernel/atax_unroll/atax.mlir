// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t --enable-unroll --adg-path %S/../../../spec/cgra_adg_fp32.json  -o %t/result.mlir
// RUN: FileCheck %s --input-file=%t/result.mlir
// RUN: rm -rf %t && mkdir -p %t
//
// Unrolled inner reduction (row dot) uses step 6 for trip count 24; second kernel
// keeps a plain inner loop over 24.
//
// CHECK: module {
// CHECK: func.func @atax(
// CHECK: %[[R:.*]] = ADORA.BlockLoad %arg3 {{\[}}%{{.*}}] : memref<?xi32> -> memref<1xi32>
// CHECK: %[[Arow:.*]] = ADORA.BlockLoad %arg0 {{\[}}%{{.*}}, 0] : memref<?x24xi32> -> memref<1x24xi32>
// CHECK: %[[X:.*]] = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<24xi32>
// CHECK: %[[Tmp:.*]] = ADORA.LocalMemAlloc memref<1xi32>
// CHECK: ADORA.kernel {
// CHECK: %[[SEED:.*]] = affine.load %[[R]][0] : memref<1xi32>
// CHECK: %[[RED:.*]] = affine.for %{{.*}} = 0 to 24 step 6 iter_args(%{{.*}} = %[[SEED]]) -> (i32) {
// CHECK: affine.load %[[Arow]]
// CHECK: affine.load %[[X]]
// CHECK: arith.muli
// CHECK: arith.addi {{.*}}%{{.*}}
// CHECK: affine.load %[[Arow]]
// CHECK: affine.load %[[X]]
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.yield {{.*}} : i32
// CHECK: }
// CHECK: affine.store %[[RED]], %[[Tmp]][0] : memref<1xi32>
// CHECK: ADORA.terminator
// CHECK: } {KernelName = "atax_0"}
// CHECK: ADORA.BlockStore %[[Tmp]], %arg3 {{\[}}%{{.*}}] : memref<1xi32> -> memref<?xi32>
// CHECK: %[[Y:.*]] = ADORA.BlockLoad %arg2 [0] : memref<?xi32> -> memref<24xi32>
// CHECK: %[[Arow2:.*]] = ADORA.BlockLoad %arg0 {{\[}}%{{.*}}, 0] : memref<?x24xi32> -> memref<1x24xi32>
// CHECK: %[[T:.*]] = ADORA.BlockLoad %arg3 {{\[}}%{{.*}}] : memref<?xi32> -> memref<1xi32>
// CHECK: %[[Out:.*]] = ADORA.LocalMemAlloc memref<24xi32>
// CHECK: ADORA.kernel {
// CHECK: affine.for %{{.*}} = 0 to 24 {
// CHECK: affine.load %[[Y]]
// CHECK: affine.load %[[Arow2]]
// CHECK: affine.load %[[T]][0] : memref<1xi32>
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: affine.store %{{.*}}, %[[Out]]
// CHECK: }
// CHECK: ADORA.terminator
// CHECK: } {KernelName = "atax_1"}
// CHECK: ADORA.BlockStore %[[Out]], %arg2 [0] : memref<24xi32> -> memref<?xi32>
// CHECK: }
// CHECK: return
// CHECK: }
// CHECK: }

module attributes {} {
  func.func @atax(%arg0: memref<?x24xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 24 {
      affine.store %c0_i32, %arg2[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to 24 {
      affine.store %c0_i32, %arg3[%arg4] : memref<?xi32>
      affine.for %arg5 = 0 to 24 {
        %0 = affine.load %arg3[%arg4] : memref<?xi32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<?x24xi32>
        %2 = affine.load %arg1[%arg5] : memref<?xi32>
        %3 = arith.muli %1, %2 : i32
        %4 = arith.addi %0, %3 : i32
        affine.store %4, %arg3[%arg4] : memref<?xi32>
      }
      affine.for %arg5 = 0 to 24 {
        %0 = affine.load %arg2[%arg5] : memref<?xi32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<?x24xi32>
        %2 = affine.load %arg3[%arg4] : memref<?xi32>
        %3 = arith.muli %1, %2 : i32
        %4 = arith.addi %0, %3 : i32
        affine.store %4, %arg2[%arg5] : memref<?xi32>
      }
    }
    return
  }
}

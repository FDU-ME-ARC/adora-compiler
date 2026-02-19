// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t -o %t/opt.mlir
// RUN: %cgra-mapper --adg=%S/../../spec/cgra_adg_fp32.json --op-file=%S/../../spec/operations_fp32.json --output-type=sdk --obj-opt=true --max-iters=1 %t/opt.mlir --output=%t/FPVecAdd_cgra.c
// RUN: test -s %t/FPVecAdd_cgra.c
// RUN: %FileCheck %s --check-prefix=CHECK-SDK --input-file=%t/FPVecAdd_cgra.c
//
// Verify cgra-mapper mapping flow: adoracc -> opt.mlir -> cgra-mapper -> SDK C output.
//
// CHECK-SDK: void
// CHECK-SDK: f32VecAddMul
module attributes {} {
  func.func @f32VecAddMul(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %argr: memref<?xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
      affine.for %arg3 = 0 to 20 {
        %3 = affine.load %arg0[%arg3] : memref<?xf32>
        %4 = affine.load %arg1[%arg3] : memref<?xf32>
        %5 = arith.addf %3, %4 : f32
        affine.store %5, %arg2[%arg3] : memref<?xf32>
        %6 = arith.mulf %3, %4 : f32
        affine.store %6, %argr[%arg3] : memref<?xf32>
      }
    return
  }
}

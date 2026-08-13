// RUN: rm -f cf_reject_scf_store_if_CDFG.dot
// RUN: not %cgra-opt --adora-kernel-dfg-gen --mlir-print-ir-after-failure %s 2>&1 | %FileCheck %s
// RUN: test ! -e cf_reject_scf_store_if_CDFG.dot

// CHECK: error: store-bearing scf.if nested under scf.for is unsupported by CDFG generation; use affine.for
// CHECK: IR Dump After ADORALoopCdfgGen Failed
// CHECK-LABEL: func.func @reject_scf_store_if
// CHECK: scf.for
// CHECK-NEXT: scf.if
// CHECK-NEXT: memref.store
// CHECK-NOT: ADORA.cond_store

module {
  func.func @reject_scf_store_if(%cond: i1, %output: memref<8xi32>,
                                 %value: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    ADORA.kernel {
      scf.for %i = %c0 to %c8 step %c1 {
        scf.if %cond {
          memref.store %value, %output[%i] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_reject_scf_store_if"}
    return
  }
}

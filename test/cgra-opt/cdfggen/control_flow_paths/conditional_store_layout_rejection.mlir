// RUN: rm -f cf_reject_strided_layout_CDFG.dot
// RUN: not %cgra-opt --adora-kernel-dfg-gen --mlir-print-ir-after-failure %s 2>&1 | %FileCheck %s
// RUN: test ! -e cf_reject_strided_layout_CDFG.dot

// CHECK: error: conditional-store lowering requires an identity-layout memref
// CHECK: IR Dump After ADORALoopCdfgGen Failed
// CHECK-LABEL: func.func @reject_strided_layout
// CHECK: scf.if
// CHECK-NEXT: memref.store
// CHECK-NOT: ADORA.cond_store

module {
  func.func @reject_strided_layout(
      %cond: i1, %value: i32,
      %output: memref<8xi32, strided<[2], offset: 1>>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      scf.if %cond {
        memref.store %value, %output[%c0] : memref<8xi32, strided<[2], offset: 1>>
      }
      ADORA.terminator
    } {KernelName = "cf_reject_strided_layout"}
    return
  }
}

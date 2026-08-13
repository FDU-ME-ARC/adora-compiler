// RUN: rm -f cf_reject_trunc_transaction_CDFG.dot
// RUN: not %cgra-opt --adora-kernel-dfg-gen --mlir-print-ir-after-failure %s 2>&1 | %FileCheck %s
// RUN: test ! -e cf_reject_trunc_transaction_CDFG.dot

// CHECK: error: unsupported operation in scf.if branch for conditional-store lowering: 'memref.load'
// CHECK: IR Dump After ADORALoopCdfgGen Failed
// CHECK-LABEL: func.func @reject_trunc_transaction
// CHECK: %[[WIDE:.*]] = arith.constant 2.000000e+00 : f64
// CHECK-NEXT: %[[NARROW:.*]] = arith.truncf %[[WIDE]] : f64 to f32
// CHECK-NOT: arith.constant 2.000000e+00 : f32
// CHECK: scf.if
// CHECK-NEXT: %{{.*}} = memref.load

module {
  func.func @reject_trunc_transaction(%cond: i1, %input: memref<8xf32>) {
    ADORA.kernel {
      %c0 = arith.constant 0 : index
      %wide = arith.constant 2.0 : f64
      %narrow = arith.truncf %wide : f64 to f32
      scf.if %cond {
        %loaded = memref.load %input[%c0] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "cf_reject_trunc_transaction"}
    return
  }
}

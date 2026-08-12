// RUN: rm -f cf_reject_load_CDFG.dot cf_reject_result_load_CDFG.dot cf_reject_copy_CDFG.dot cf_reject_loop_CDFG.dot cf_reject_scf_for_CDFG.dot
// RUN: not %cgra-opt --adora-kernel-dfg-gen --mlir-print-ir-after-failure -split-input-file %s 2>&1 | %FileCheck %s
// RUN: test ! -e cf_reject_load_CDFG.dot
// RUN: test ! -e cf_reject_result_load_CDFG.dot
// RUN: test ! -e cf_reject_copy_CDFG.dot
// RUN: test ! -e cf_reject_loop_CDFG.dot
// RUN: test ! -e cf_reject_scf_for_CDFG.dot

// CHECK: error: unsupported operation in scf.if branch for conditional-store lowering: 'memref.load'
// CHECK: IR Dump After ADORALoopCdfgGen Failed
// CHECK: scf.if
// CHECK-NEXT: memref.store
// CHECK-NEXT: %{{[0-9]+}} = memref.load
// CHECK-NEXT: memref.store
// CHECK: error: unsupported operation in scf.if branch for conditional-store lowering: 'memref.load'
// CHECK: error: unsupported operation in scf.if branch for conditional-store lowering: 'memref.copy'
// CHECK: error: unsupported operation in scf.if branch for conditional-store lowering: 'affine.for'
// CHECK: error: ADORA.cond_store nested under scf.for is unsupported by CDFG generation

module {
  func.func @reject_store_load_store(%cond: i1, %first: i32,
                                     %output: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ADORA.kernel {
      scf.if %cond {
        memref.store %first, %output[%c0] : memref<8xi32>
        %loaded = memref.load %output[%c0] : memref<8xi32>
        memref.store %loaded, %output[%c1] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_reject_load"}
    return
  }
}

// -----

module {
  func.func @reject_result_load(%cond: i1, %input: memref<8xi32>,
                                %other: i32) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      %selected = scf.if %cond -> (i32) {
        %loaded = memref.load %input[%c0] : memref<8xi32>
        scf.yield %loaded : i32
      } else {
        scf.yield %other : i32
      }
      %used = arith.addi %selected, %other : i32
      ADORA.terminator
    } {KernelName = "cf_reject_result_load"}
    return
  }
}

// -----

module {
  func.func @reject_false_copy(%cond: i1, %input: memref<8xi32>,
                               %output: memref<8xi32>, %value: i32) {
    ADORA.kernel {
      scf.if %cond {
        affine.store %value, %output[0] : memref<8xi32>
      } else {
        memref.copy %input, %output : memref<8xi32> to memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_reject_copy"}
    return
  }
}

// -----

module {
  func.func @reject_false_store_loop(%cond: i1, %output: memref<8xi32>,
                                     %value: i32) {
    ADORA.kernel {
      scf.if %cond {
        affine.store %value, %output[0] : memref<8xi32>
      } else {
        affine.for %i = 0 to 8 {
          affine.store %value, %output[%i] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_reject_loop"}
    return
  }
}

// -----

module {
  func.func @reject_scf_for(%cond: i1, %output: memref<8xi32>, %value: i32) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    ADORA.kernel {
      scf.for %i = %c0 to %c8 step %c1 {
        ADORA.cond_store %value, %output[%i] if %cond : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_reject_scf_for"}
    return
  }
}

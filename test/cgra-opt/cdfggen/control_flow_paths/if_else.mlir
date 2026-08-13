// RUN: rm -f cf_if_else_CDFG.dot cf_if_else_repeat_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s > %t.out 2> %t.err
// RUN: %FileCheck %s --input-file=%t.out
// RUN: %FileCheck %s --check-prefix=ERR --input-file=%t.err
// RUN: test -s cf_if_else_CDFG.dot
// RUN: test "$(grep -c 'opcode = \"SEL\"' cf_if_else_CDFG.dot)" -eq 1
// RUN: %FileCheck %s --check-prefix=DOT --input-file=cf_if_else_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: test -s cf_if_else_repeat_CDFG.dot
// RUN: test "$(grep -c 'ref_name=\"cf_if_else_repeat:arg1\"' cf_if_else_repeat_CDFG.dot)" -eq 1
// RUN: %FileCheck %s --check-prefix=REPEAT --input-file=cf_if_else_repeat_CDFG.dot
// RUN: rm -f cf_if_else_CDFG.dot cf_if_else_repeat_CDFG.dot

// CHECK-LABEL: func.func @if_else
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-1: arith.select
// CHECK-NOT: scf.if
// CHECK: ADORA.terminator
// CHECK-LABEL: func.func @if_else_repeated_source
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-1: arith.select
// CHECK-NOT: scf.if
// CHECK: ADORA.terminator

// ERR-NOT: outputlist already has

// DOT: Digraph G {
// DOT-DAG: Input[[COND:[0-9]+]][opcode = "Input", ref_name="cf_if_else:arg0", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[FALSE:[0-9]+]][opcode = "Input", ref_name="cf_if_else:arg1", size="4", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[TRUE:[0-9]+]][opcode = "Input", ref_name="cf_if_else:arg2", size="4", offset="0,0", pattern="0,1"
// DOT-DAG: SEL[[SEL:[0-9]+]][opcode = "SEL"
// DOT-DAG: Input[[FALSE]] -> SEL[[SEL]]{{[^]]*}}operand = 0, label = "Op=0"
// DOT-DAG: Input[[TRUE]] -> SEL[[SEL]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT-DAG: Input[[COND]] -> SEL[[SEL]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT: }

// REPEAT: Digraph G {
// REPEAT-DAG: Input[[REUSE_COND:[0-9]+]][opcode = "Input", ref_name="cf_if_else_repeat:arg0", size="1", offset="0,0", pattern="0,1"
// REPEAT-DAG: Input[[REUSE_VALUE:[0-9]+]][opcode = "Input", ref_name="cf_if_else_repeat:arg1", size="4", offset="0,0", pattern="0,1"
// REPEAT-DAG: SEL[[REUSE_SEL:[0-9]+]][opcode = "SEL"
// REPEAT-DAG: FADD32[[REUSE_ADD:[0-9]+]][opcode = "FADD32"
// REPEAT-DAG: Input[[REUSE_VALUE]] -> SEL[[REUSE_SEL]]{{[^]]*}}operand = 0, label = "Op=0"
// REPEAT-DAG: Input[[REUSE_VALUE]] -> SEL[[REUSE_SEL]]{{[^]]*}}operand = 1, label = "Op=1"
// REPEAT-DAG: Input[[REUSE_COND]] -> SEL[[REUSE_SEL]]{{[^]]*}}operand = 2, label = "Op=2"
// REPEAT-DAG: Input[[REUSE_VALUE]] -> FADD32[[REUSE_ADD]]{{[^]]*}}operand = 0, label = "Op=0"
// REPEAT-DAG: Input[[REUSE_VALUE]] -> FADD32[[REUSE_ADD]]{{[^]]*}}operand = 1, label = "Op=1"
// REPEAT: }

module {
  func.func @if_else(%cond: i1, %false_value: f32, %true_value: f32, %output: memref<8xf32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %selected = scf.if %cond -> (f32) {
          scf.yield %true_value : f32
        } else {
          scf.yield %false_value : f32
        }
        affine.store %selected, %output[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "cf_if_else"}
    return
  }

  func.func @if_else_repeated_source(%cond: i1, %value: f32, %selected_output: memref<8xf32>, %probe_output: memref<8xf32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %selected = scf.if %cond -> (f32) {
          scf.yield %value : f32
        } else {
          scf.yield %value : f32
        }
        %probe = arith.addf %value, %value : f32
        affine.store %selected, %selected_output[%i] : memref<8xf32>
        affine.store %probe, %probe_output[%i] : memref<8xf32>
      }
      ADORA.terminator
    } {KernelName = "cf_if_else_repeat"}
    return
  }
}

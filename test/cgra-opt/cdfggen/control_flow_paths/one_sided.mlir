// RUN: rm -f cf_one_sided_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test -s cf_one_sided_CDFG.dot
// RUN: test "$(grep -c 'opcode = \"SEL\"' cf_one_sided_CDFG.dot)" -eq 1
// RUN: %FileCheck %s --check-prefix=DOT --input-file=cf_one_sided_CDFG.dot
// RUN: rm -f cf_one_sided_CDFG.dot

// CHECK-LABEL: func.func @one_sided
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-1: arith.select
// CHECK-NOT: scf.if
// CHECK: ADORA.terminator

// DOT: Digraph G {
// DOT-DAG: Input[[COND:[0-9]+]][opcode = "Input", ref_name="cf_one_sided:arg0", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[VALUE:[0-9]+]][opcode = "Input", ref_name="cf_one_sided:arg1", size="4", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[OLD:[0-9]+]][opcode = "Input"
// DOT-DAG: SEL[[SEL:[0-9]+]][opcode = "SEL"
// DOT-DAG: Input[[OLD]] -> SEL[[SEL]]{{[^]]*}}operand = 0, label = "Op=0"
// DOT-DAG: Input[[VALUE]] -> SEL[[SEL]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT-DAG: Input[[COND]] -> SEL[[SEL]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-NOT: opcode = "undefined"
// DOT-NOT: opcode = "CTRL
// DOT: }

module {
  func.func @one_sided(%cond: i1, %value: i32, %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %old = affine.load %output[%i] : memref<8xi32>
        scf.if %cond {
          affine.store %value, %output[%i] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_one_sided"}
    return
  }
}

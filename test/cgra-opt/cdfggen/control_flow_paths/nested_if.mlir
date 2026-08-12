// RUN: rm -f cf_nested_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test -s cf_nested_CDFG.dot
// RUN: test "$(grep -c 'opcode = \"SEL\"' cf_nested_CDFG.dot)" -eq 2
// RUN: %FileCheck %s --check-prefix=DOT --input-file=cf_nested_CDFG.dot
// RUN: rm -f cf_nested_CDFG.dot

// CHECK-LABEL: func.func @nested_if
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-2: arith.select
// CHECK-NOT: scf.if
// CHECK: ADORA.terminator

// DOT: Digraph G {
// DOT-DAG: Input[[A:[0-9]+]][opcode = "Input", ref_name="cf_nested:arg0", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[B:[0-9]+]][opcode = "Input", ref_name="cf_nested:arg1", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: SEL[[INNER:[0-9]+]][opcode = "SEL"
// DOT-DAG: SEL[[OUTER:[0-9]+]][opcode = "SEL"
// DOT-DAG: Input[[B]] -> SEL[[INNER]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-DAG: Input[[A]] -> SEL[[OUTER]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-DAG: SEL[[INNER]] -> SEL[[OUTER]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT-NOT: opcode = "undefined"
// DOT-NOT: opcode = "CTRL
// DOT: }

module {
  func.func @nested_if(%a: i1, %b: i1, %output: memref<8xi32>) {
    %c11 = arith.constant 11 : i32
    %c22 = arith.constant 22 : i32
    %c33 = arith.constant 33 : i32
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %selected = scf.if %a -> (i32) {
          %nested = scf.if %b -> (i32) {
            scf.yield %c11 : i32
          } else {
            scf.yield %c22 : i32
          }
          scf.yield %nested : i32
        } else {
          scf.yield %c33 : i32
        }
        affine.store %selected, %output[%i] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_nested"}
    return
  }
}

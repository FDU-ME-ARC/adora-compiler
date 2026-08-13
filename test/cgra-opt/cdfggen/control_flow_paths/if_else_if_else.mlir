// RUN: rm -f cf_else_if_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test -s cf_else_if_CDFG.dot
// RUN: test "$(grep -c 'opcode = \"SEL\"' cf_else_if_CDFG.dot)" -eq 2
// RUN: %FileCheck %s --check-prefix=DOT --input-file=cf_else_if_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: rm -f cf_else_if_CDFG.dot

// CHECK-LABEL: func.func @if_else_if_else
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-2: arith.select
// CHECK-NOT: scf.if
// CHECK: ADORA.terminator

// DOT: Digraph G {
// DOT-DAG: Input[[A:[0-9]+]][opcode = "Input", ref_name="cf_else_if:arg0", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[B:[0-9]+]][opcode = "Input", ref_name="cf_else_if:arg1", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: SEL[[INNER:[0-9]+]][opcode = "SEL"
// DOT-DAG: SEL[[OUTER:[0-9]+]][opcode = "SEL"
// DOT-DAG: CONST[[INNER_TRUE:[0-9]+]][opcode = "CONST", value="0x00000016"
// DOT-DAG: CONST[[INNER_FALSE:[0-9]+]][opcode = "CONST", value="0x00000021"
// DOT-DAG: CONST[[OUTER_TRUE:[0-9]+]][opcode = "CONST", value="0x0000000B"
// DOT-DAG: Input[[B]] -> SEL[[INNER]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-DAG: CONST[[INNER_FALSE]] -> SEL[[INNER]]{{[^]]*}}operand = 0, label = "Op=0"
// DOT-DAG: CONST[[INNER_TRUE]] -> SEL[[INNER]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT-DAG: Input[[A]] -> SEL[[OUTER]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-DAG: SEL[[INNER]] -> SEL[[OUTER]]{{[^]]*}}operand = 0, label = "Op=0"
// DOT-DAG: CONST[[OUTER_TRUE]] -> SEL[[OUTER]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT: }

module {
  func.func @if_else_if_else(%a: i1, %b: i1, %output: memref<8xi32>) {
    %c11 = arith.constant 11 : i32
    %c22 = arith.constant 22 : i32
    %c33 = arith.constant 33 : i32
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %selected = scf.if %a -> (i32) {
          scf.yield %c11 : i32
        } else {
          %fallback = scf.if %b -> (i32) {
            scf.yield %c22 : i32
          } else {
            scf.yield %c33 : i32
          }
          scf.yield %fallback : i32
        }
        affine.store %selected, %output[%i] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_else_if"}
    return
  }
}

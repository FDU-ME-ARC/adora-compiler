// RUN: rm -f cf_one_sided_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test -s cf_one_sided_CDFG.dot
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_one_sided_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"SEL\"' cf_one_sided_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'iterdist =' cf_one_sided_CDFG.dot)" -eq 0
// RUN: %FileCheck %s --check-prefix=DOT --input-file=cf_one_sided_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"' --implicit-check-not='iterdist ='
// RUN: rm -f cf_one_sided_CDFG.dot

// CHECK-LABEL: func.func @one_sided
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-NOT: affine.load
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %{{.*}} : memref<8xi32>
// CHECK-NOT: affine.load
// CHECK: ADORA.terminator

// DOT: Digraph G {
// DOT-DAG: Input[[COND:[0-9]+]][opcode = "Input", ref_name="cf_one_sided:arg0", size="1", offset="0,0", pattern="0,1"
// DOT-DAG: Input[[VALUE:[0-9]+]][opcode = "Input", ref_name="cf_one_sided:arg1", size="4", offset="0,0", pattern="0,1"
// DOT-DAG: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE"
// DOT-DAG: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// DOT-DAG: MUL[[ADDR:[0-9]+]][opcode = "MUL"
// DOT-DAG: Input[[VALUE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// DOT-DAG: MUL[[ADDR]] -> CSTORE[[STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// DOT-DAG: Input[[COND]] -> CSTORE[[STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// DOT-DAG: CONST[[FOUR]] -> MUL[[ADDR]]
// DOT: }

module {
  func.func @one_sided(%cond: i1, %value: i32, %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          affine.store %value, %output[%i] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_one_sided"}
    return
  }
}

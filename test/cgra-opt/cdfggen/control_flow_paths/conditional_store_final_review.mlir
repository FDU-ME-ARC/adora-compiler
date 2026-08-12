// RUN: rm -f cf_memory_before_CDFG.dot cf_memory_after_loop_CDFG.dot cf_affine_apply_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: %FileCheck %s --check-prefix=BEFORE-DOT --input-file=cf_memory_before_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=AFTER-DOT --input-file=cf_memory_after_loop_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=APPLY-DOT --input-file=cf_affine_apply_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: rm -f cf_memory_before_CDFG.dot cf_memory_after_loop_CDFG.dot cf_affine_apply_CDFG.dot

// CHECK-LABEL: func.func @memory_before_cstore
// CHECK: ADORA.kernel
// CHECK: affine.store
// CHECK-NEXT: affine.store
// CHECK-NEXT: ADORA.cond_store
// CHECK: ADORA.terminator

// BEFORE-DOT: Digraph G {
// BEFORE-DOT: Output[[FIRST:[0-9]+]][opcode = "Output"
// BEFORE-DOT: Output[[MIDDLE:[0-9]+]][opcode = "Output"
// BEFORE-DOT: CSTORE[[LAST:[0-9]+]][opcode = "CSTORE"
// BEFORE-DOT-DAG: Output[[FIRST]] -> Output[[MIDDLE]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BEFORE-DOT-DAG: Output[[MIDDLE]] -> CSTORE[[LAST]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BEFORE-DOT-NOT: CSTORE[[LAST]] -> Output[[FIRST]]
// BEFORE-DOT: }

// CHECK-LABEL: func.func @memory_after_cstore_in_loop
// CHECK: affine.for
// CHECK: ADORA.cond_store
// CHECK-NEXT: affine.store
// CHECK-NEXT: %{{.*}} = affine.load
// CHECK: ADORA.terminator

// AFTER-DOT: Digraph G {
// AFTER-DOT: CSTORE[[FIRST:[0-9]+]][opcode = "CSTORE"
// AFTER-DOT: Output[[MIDDLE:[0-9]+]][opcode = "Output"
// AFTER-DOT: Input[[LAST:[0-9]+]][opcode = "Input"
// AFTER-DOT-DAG: CSTORE[[FIRST]] -> Output[[MIDDLE]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// AFTER-DOT-DAG: Output[[MIDDLE]] -> Input[[LAST]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// AFTER-DOT-NOT: Input[[LAST]] -> CSTORE[[FIRST]]
// AFTER-DOT: }

// CHECK-LABEL: func.func @composed_affine_apply
// CHECK: affine.for
// CHECK-NOT: scf.if
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%[[ADDRESS:.*]]] if %{{.*}} : memref<32xi32>
// CHECK: ADORA.terminator

// APPLY-DOT: Digraph G {
// APPLY-DOT: ADD[[COMPOSED:[0-9]+]][opcode = "ADD"
// APPLY-DOT: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_affine_apply:arg2", size="128", offset="0,0", pattern="0,1"
// APPLY-DOT: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// APPLY-DOT-NEXT: MUL[[BYTE_ADDRESS:[0-9]+]][opcode = "MUL"
// APPLY-DOT-DAG: {{.*}} -> CSTORE[[STORE]]{{.*}}operand = 0, label = "Op=0"
// APPLY-DOT-DAG: ADD[[COMPOSED]] -> MUL[[BYTE_ADDRESS]]{{.*}}operand = 0, label = "Op=0"
// APPLY-DOT-DAG: CONST[[FOUR]] -> MUL[[BYTE_ADDRESS]]{{.*}}operand = 1, label = "Op=1"
// APPLY-DOT-DAG: MUL[[BYTE_ADDRESS]] -> CSTORE[[STORE]]{{.*}}operand = 1, label = "Op=1"
// APPLY-DOT-DAG: {{.*}} -> CSTORE[[STORE]]{{.*}}operand = 2, label = "Op=2"
// APPLY-DOT: }

#plus_one = affine_map<(d0) -> (d0 + 1)>
#twice = affine_map<(d0) -> (d0 * 2)>

module {
  func.func @memory_before_cstore(%cond: i1, %first: i32, %middle: i32,
                                  %last: i32, %a: memref<8xi32>,
                                  %b: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      affine.store %first, %a[0] : memref<8xi32>
      affine.store %middle, %b[0] : memref<8xi32>
      ADORA.cond_store %last, %a[%c0] if %cond : memref<8xi32>
      ADORA.terminator
    } {KernelName = "cf_memory_before"}
    return
  }

  func.func @memory_after_cstore_in_loop(%cond: i1, %first: i32,
                                         %middle: i32, %a: memref<8xi32>,
                                         %b: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        ADORA.cond_store %first, %a[%i] if %cond : memref<8xi32>
        affine.store %middle, %b[%i] : memref<8xi32>
        %loaded = affine.load %a[%i] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_memory_after_loop"}
    return
  }

  func.func @composed_affine_apply(%cond: i1, %value: i32,
                                   %output: memref<32xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %plus_one = affine.apply #plus_one(%i)
        %twice = affine.apply #twice(%plus_one)
        scf.if %cond {
          affine.store %value, %output[%twice] : memref<32xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_affine_apply"}
    return
  }
}

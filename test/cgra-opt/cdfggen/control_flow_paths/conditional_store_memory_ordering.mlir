// RUN: rm -f cf_memory_before_CDFG.dot cf_memory_after_loop_CDFG.dot cf_affine_apply_CDFG.dot cf_memory_loop_boundary_CDFG.dot cf_cstore_hoist_barrier_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: %FileCheck %s --check-prefix=BEFORE-DOT --input-file=cf_memory_before_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=AFTER-DOT --input-file=cf_memory_after_loop_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=APPLY-DOT --input-file=cf_affine_apply_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=BOUNDARY-DOT --input-file=cf_memory_loop_boundary_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=HOIST-DOT --input-file=cf_cstore_hoist_barrier_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: rm -f cf_memory_before_CDFG.dot cf_memory_after_loop_CDFG.dot cf_affine_apply_CDFG.dot cf_memory_loop_boundary_CDFG.dot cf_cstore_hoist_barrier_CDFG.dot

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
// APPLY-DOT: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_affine_apply:arg2", size="128", offset="0,0", pattern="0,8"
// APPLY-DOT: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// APPLY-DOT-NEXT: MUL[[BYTE_ADDRESS:[0-9]+]][opcode = "MUL"
// APPLY-DOT-DAG: {{.*}} -> CSTORE[[STORE]]{{.*}}operand = 0, label = "Op=0"
// APPLY-DOT-DAG: ADD[[COMPOSED]] -> MUL[[BYTE_ADDRESS]]{{.*}}operand = 0, label = "Op=0"
// APPLY-DOT-DAG: CONST[[FOUR]] -> MUL[[BYTE_ADDRESS]]{{.*}}operand = 1, label = "Op=1"
// APPLY-DOT-DAG: MUL[[BYTE_ADDRESS]] -> CSTORE[[STORE]]{{.*}}operand = 1, label = "Op=1"
// APPLY-DOT-DAG: {{.*}} -> CSTORE[[STORE]]{{.*}}operand = 2, label = "Op=2"
// APPLY-DOT: }

// CHECK-LABEL: func.func @memory_across_nested_affine_loop
// CHECK: affine.store
// CHECK-NEXT: %{{.*}} = affine.load
// CHECK-NEXT: affine.for
// CHECK: affine.for
// CHECK: ADORA.cond_store
// CHECK: %{{.*}} = affine.load
// CHECK-NEXT: affine.store

// BOUNDARY-DOT: Digraph G {
// BOUNDARY-DOT-DAG: Output[[BEFORE_STORE:[0-9]+]][opcode = "Output"
// BOUNDARY-DOT-DAG: Input[[BEFORE_LOAD:[0-9]+]][opcode = "Input", ref_name="cf_memory_loop_boundary:arg3", size="32"
// BOUNDARY-DOT-DAG: CSTORE[[INNER:[0-9]+]][opcode = "CSTORE", ref_name="cf_memory_loop_boundary:arg4", size="32"
// BOUNDARY-DOT-DAG: Input[[AFTER_LOAD:[0-9]+]][opcode = "Input", ref_name="cf_memory_loop_boundary:arg5", size="32"
// BOUNDARY-DOT-DAG: Output[[AFTER_STORE:[0-9]+]][opcode = "Output"
// BOUNDARY-DOT-DAG: Output[[BEFORE_STORE]] -> Input[[BEFORE_LOAD]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BOUNDARY-DOT-DAG: Input[[BEFORE_LOAD]] -> CSTORE[[INNER]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BOUNDARY-DOT-DAG: CSTORE[[INNER]] -> Input[[AFTER_LOAD]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BOUNDARY-DOT-DAG: Input[[AFTER_LOAD]] -> Output[[AFTER_STORE]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// BOUNDARY-DOT-NOT: Output[[AFTER_STORE]] -> Output[[BEFORE_STORE]]
// BOUNDARY-DOT: }

// CHECK-LABEL: func.func @cstore_blocks_load_store_hoist
// CHECK: affine.for
// CHECK-NEXT: %[[INITIAL:.*]] = affine.load %{{.*}}[0]
// CHECK-NEXT: %[[UPDATED:.*]] = arith.addi %[[INITIAL]], %{{.*}}
// CHECK-NEXT: ADORA.cond_store
// CHECK-NEXT: affine.store %[[UPDATED]], %{{.*}}[0]
// CHECK-NEXT: %[[OBSERVED:.*]] = affine.load %{{.*}}[0]
// CHECK-NEXT: affine.store %[[OBSERVED]], %{{.*}}[%{{.*}}]

// HOIST-DOT: Digraph G {
// HOIST-DOT-DAG: Input[[INITIAL_LOAD:[0-9]+]][opcode = "Input", ref_name="cf_cstore_hoist_barrier:arg2", size="4"
// HOIST-DOT-DAG: CSTORE[[CONDITIONAL:[0-9]+]][opcode = "CSTORE", ref_name="cf_cstore_hoist_barrier:arg3", size="32"
// HOIST-DOT-DAG: Output[[MATCHED_STORE:[0-9]+]][opcode = "Output"
// HOIST-DOT-DAG: Input[[OBSERVED_LOAD:[0-9]+]][opcode = "Input", ref_name="cf_cstore_hoist_barrier:arg2", size="4"
// HOIST-DOT-DAG: Output[[OBSERVED_STORE:[0-9]+]][opcode = "Output"
// HOIST-DOT-DAG: Input[[INITIAL_LOAD]] -> CSTORE[[CONDITIONAL]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// HOIST-DOT-DAG: CSTORE[[CONDITIONAL]] -> Output[[MATCHED_STORE]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// HOIST-DOT-DAG: Output[[MATCHED_STORE]] -> Input[[OBSERVED_LOAD]][color = blue{{.*}}operand = -1, label = "Op=-1, DepDist = 0"
// HOIST-DOT-DAG: Input[[OBSERVED_LOAD]] -> Output[[OBSERVED_STORE]][color = black{{.*}}operand = 0, label = "Op=0"
// HOIST-DOT-NOT: Input[[OBSERVED_LOAD]] -> Output[[MATCHED_STORE]]
// HOIST-DOT: }

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
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        ADORA.cond_store %first, %a[%c0] if %cond : memref<8xi32>
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

  func.func @memory_across_nested_affine_loop(
      %cond: i1, %value: i32, %before_store: memref<8xi32>,
      %before_load: memref<8xi32>, %loop_target: memref<8xi32>,
      %after_load: memref<8xi32>, %after_store: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      affine.store %value, %before_store[0] : memref<8xi32>
      %before = affine.load %before_load[0] : memref<8xi32>
      affine.for %i = 0 to 2 {
        affine.for %j = 0 to 8 {
          ADORA.cond_store %value, %loop_target[%c0] if %cond : memref<8xi32>
        }
      }
      %after = affine.load %after_load[0] : memref<8xi32>
      affine.store %value, %after_store[0] : memref<8xi32>
      ADORA.terminator
    } {KernelName = "cf_memory_loop_boundary"}
    return
  }

  func.func @cstore_blocks_load_store_hoist(
      %cond: i1, %delta: i32, %state: memref<1xi32>,
      %conditional: memref<8xi32>, %observed: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        %initial = affine.load %state[0] : memref<1xi32>
        %updated = arith.addi %initial, %delta : i32
        scf.if %cond {
          affine.store %delta, %conditional[%c0] : memref<8xi32>
        }
        affine.store %updated, %state[0] : memref<1xi32>
        %observed_value = affine.load %state[0] : memref<1xi32>
        affine.store %observed_value, %observed[%i] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_cstore_hoist_barrier"}
    return
  }
}

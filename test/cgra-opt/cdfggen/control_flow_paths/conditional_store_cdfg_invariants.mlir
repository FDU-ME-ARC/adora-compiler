// RUN: rm -f cf_ordered_store_CDFG.dot cf_paired_values_CDFG.dot cf_mapped_producers_CDFG.dot cf_direct_affine_baseline_CDFG.dot cf_unrelated_memory_CDFG.dot cf_mixed_constant_CDFG.dot cf_fallback_rank_two_CDFG.dot cf_crossed_order_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_ordered_store_CDFG.dot)" -eq 2
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_mapped_producers_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_mixed_constant_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_crossed_order_CDFG.dot)" -eq 2
// RUN: %FileCheck %s --check-prefix=ORDER-DOT --input-file=cf_ordered_store_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=PRODUCER-DOT --input-file=cf_mapped_producers_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=BASELINE-DOT --input-file=cf_direct_affine_baseline_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=UNRELATED-DOT --input-file=cf_unrelated_memory_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"' --implicit-check-not='opcode = "load"' --implicit-check-not='opcode = "store"'
// RUN: %FileCheck %s --check-prefix=CONSTANT-DOT --input-file=cf_mixed_constant_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=FALLBACK-DOT --input-file=cf_fallback_rank_two_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=CROSSED-DOT --input-file=cf_crossed_order_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: rm -f cf_ordered_store_CDFG.dot cf_paired_values_CDFG.dot cf_mapped_producers_CDFG.dot cf_direct_affine_baseline_CDFG.dot cf_unrelated_memory_CDFG.dot cf_mixed_constant_CDFG.dot cf_fallback_rank_two_CDFG.dot cf_crossed_order_CDFG.dot

// CHECK-LABEL: func.func @ordered_nested_then_direct(
// CHECK-SAME: %[[OUTER:[a-zA-Z0-9]+]]: i1, %[[INNER:[a-zA-Z0-9]+]]: i1, %[[FIRST:[a-zA-Z0-9]+]]: i32, %[[SECOND:[a-zA-Z0-9]+]]: i32
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-NOT: affine.load
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[PATH:.*]] = arith.select %[[OUTER]], %[[INNER]], %[[FALSE]] : i1
// CHECK: ADORA.cond_store %[[FIRST]], %{{.*}}[%{{.*}}] if %[[PATH]] : memref<8xi32>
// CHECK: ADORA.cond_store %[[SECOND]], %{{.*}}[%{{.*}}] if %[[OUTER]] : memref<8xi32>
// CHECK: ADORA.terminator

// ORDER-DOT: Digraph G {
// ORDER-DOT-COUNT-2: opcode = "CSTORE"
// ORDER-DOT: }

// CHECK-LABEL: func.func @paired_branch_values(
// CHECK: affine.for
// CHECK: %[[THEN:.*]] = arith.addi
// CHECK-NEXT: %[[ELSE:.*]] = arith.subi
// CHECK-NEXT: %[[SELECTED:.*]] = arith.select %{{.*}}, %[[THEN]], %[[ELSE]] : i32
// CHECK-NEXT: affine.store %[[SELECTED]], %{{.*}}[0] : memref<8xi32>

// CHECK-LABEL: func.func @mapped_producers
// CHECK: %[[INDEX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[VALUE:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK: %[[PREDICATE:.*]] = arith.xori %{{.*}}, %{{.*}} : i1
// CHECK: ADORA.kernel
// CHECK: ADORA.cond_store %[[VALUE]], %{{.*}}[%[[INDEX]]] if %[[PREDICATE]] : memref<16xi32>
// CHECK: ADORA.terminator

// PRODUCER-DOT: Digraph G {
// PRODUCER-DOT-DAG: Input[[RAW:[0-9]+]][opcode = "Input", ref_name="cf_mapped_producers:arg0"
// PRODUCER-DOT-DAG: Input[[VALUE_INPUT:[0-9]+]][opcode = "Input", ref_name="cf_mapped_producers:arg1"
// PRODUCER-DOT-DAG: Input[[COND:[0-9]+]][opcode = "Input", ref_name="cf_mapped_producers:arg2"
// PRODUCER-DOT-DAG: XOR[[VALUE:[0-9]+]][opcode = "XOR"
// PRODUCER-DOT-DAG: XOR[[PREDICATE:[0-9]+]][opcode = "XOR"
// PRODUCER-DOT-DAG: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// PRODUCER-DOT-DAG: MUL[[BYTE_ADDR:[0-9]+]][opcode = "MUL"
// PRODUCER-DOT-DAG: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE"
// PRODUCER-DOT-DAG: Input[[VALUE_INPUT]] -> XOR[[VALUE]]
// PRODUCER-DOT-DAG: XOR[[VALUE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// PRODUCER-DOT-DAG: Input[[RAW]] -> MUL[[BYTE_ADDR]]
// PRODUCER-DOT-DAG: CONST[[FOUR]] -> MUL[[BYTE_ADDR]]
// PRODUCER-DOT-DAG: MUL[[BYTE_ADDR]] -> CSTORE[[STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// PRODUCER-DOT-DAG: Input[[COND]] -> XOR[[PREDICATE]]
// PRODUCER-DOT-DAG: XOR[[PREDICATE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// PRODUCER-DOT: }

// CHECK-LABEL: func.func @direct_affine_baseline
// CHECK: ADORA.kernel
// CHECK: %[[LOADED:.*]] = affine.load
// CHECK: %[[SUM:.*]] = arith.addi %[[LOADED]],
// CHECK: %[[DIFFERENCE:.*]] = arith.subi %[[SUM]],
// CHECK: affine.store %[[DIFFERENCE]],
// CHECK: ADORA.terminator

// BASELINE-DOT: Digraph G {
// BASELINE-DOT: Input[[LOAD:[0-9]+]][opcode = "Input"
// BASELINE-DOT: ADD[[SUM:[0-9]+]][opcode = "ADD"
// BASELINE-DOT: SUB[[DIFFERENCE:[0-9]+]][opcode = "SUB"
// BASELINE-DOT: Output[[STORE:[0-9]+]][opcode = "Output"
// BASELINE-DOT: Input[[RHS:[0-9]+]][opcode = "Input", ref_name="cf_direct_affine_baseline:arg2"
// BASELINE-DOT-DAG: Input[[LOAD]] -> ADD[[SUM]]{{[^]]*}}operand = 0, label = "Op=0"
// BASELINE-DOT-DAG: Input[[RHS]] -> ADD[[SUM]]{{[^]]*}}operand = 1, label = "Op=1"
// BASELINE-DOT-DAG: ADD[[SUM]] -> SUB[[DIFFERENCE]]{{[^]]*}}operand = 0, label = "Op=0"
// BASELINE-DOT-DAG: Input[[RHS]] -> SUB[[DIFFERENCE]]{{[^]]*}}operand = 1, label = "Op=1"
// BASELINE-DOT-DAG: SUB[[DIFFERENCE]] -> Output[[STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// BASELINE-DOT: }

// CHECK-LABEL: func.func @unrelated_direct_memory
// CHECK: ADORA.kernel
// CHECK: memref.load
// CHECK: memref.store
// CHECK: ADORA.terminator

// UNRELATED-DOT: Digraph G {
// UNRELATED-DOT: opcode = "CSTORE"
// UNRELATED-DOT: }

// CHECK-LABEL: func.func @mixed_constant_address
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-NOT: ADORA.cond_store
// CHECK: %[[SELECTED:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: affine.store %[[SELECTED]], %{{.*}}[3] : memref<8xi32>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @fallback_rank_two
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-NOT: ADORA.cond_store
// CHECK-NOT: arith.constant false
// CHECK-NOT: arith.constant true
// CHECK: affine.load
// CHECK: arith.select
// CHECK: affine.store
// CHECK: ADORA.terminator

// FALLBACK-DOT: Digraph G {
// FALLBACK-DOT: opcode = "Input"
// FALLBACK-DOT: opcode = "Output"
// FALLBACK-DOT-NOT: opcode = "CSTORE"
// FALLBACK-DOT: }

// CHECK-LABEL: func.func @crossed_store_order(
// CHECK-SAME: %[[COND:[a-zA-Z0-9]+]]: i1, %[[THEN_A:[a-zA-Z0-9]+]]: i32, %[[THEN_B:[a-zA-Z0-9]+]]: i32, %[[ELSE_B:[a-zA-Z0-9]+]]: i32, %[[ELSE_A:[a-zA-Z0-9]+]]: i32
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK: ADORA.cond_store %[[THEN_A]], %{{.*}}[%{{.*}}] if %[[COND]] : memref<16xi32>
// CHECK: %[[B_VALUE:.*]] = arith.select %[[COND]], %[[THEN_B]], %[[ELSE_B]] : i32
// CHECK: memref.store %[[B_VALUE]], %{{.*}}[%{{.*}}] : memref<16xi32>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: %[[TRUE:.*]] = arith.constant true
// CHECK: %[[ELSE_PATH:.*]] = arith.select %[[COND]], %[[FALSE]], %[[TRUE]] : i1
// CHECK: ADORA.cond_store %[[ELSE_A]], %{{.*}}[%{{.*}}] if %[[ELSE_PATH]] : memref<16xi32>
// CHECK: ADORA.terminator

// CONSTANT-DOT: Digraph G {
// CONSTANT-DOT: opcode = "SEL"
// CONSTANT-DOT-NOT: opcode = "CSTORE"
// CONSTANT-DOT: }

// CROSSED-DOT: Digraph G {
// CROSSED-DOT-DAG: opcode = "CSTORE"
// CROSSED-DOT-DAG: opcode = "SEL"
// CROSSED-DOT: }

module {
  func.func @ordered_nested_then_direct(%outer: i1, %inner: i1,
                                         %first: i32, %second: i32,
                                         %output: memref<8xi32>) {
    ADORA.kernel {
      scf.if %outer {
        scf.if %inner {
          affine.store %first, %output[0] : memref<8xi32>
        }
        affine.store %second, %output[0] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_ordered_store"}
    return
  }

  func.func @paired_branch_values(%cond: i1, %lhs: i32, %rhs: i32,
                                  %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          %then_value = arith.addi %lhs, %rhs : i32
          affine.store %then_value, %output[0] : memref<8xi32>
        } else {
          %else_value = arith.subi %lhs, %rhs : i32
          affine.store %else_value, %output[0] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_paired_values"}
    return
  }

  func.func @mapped_producers(%raw_index: i32, %value: i32, %cond: i1,
                              %output: memref<16xi32>) {
    %index = arith.index_cast %raw_index : i32 to index
    %c1 = arith.constant 1 : i32
    %stored_value = arith.xori %value, %c1 : i32
    %true = arith.constant true
    %predicate = arith.xori %cond, %true : i1
    ADORA.kernel {
      ADORA.cond_store %stored_value, %output[%index] if %predicate : memref<16xi32>
      ADORA.terminator
    } {KernelName = "cf_mapped_producers"}
    return
  }

  func.func @direct_affine_baseline(%input: memref<4xi32>,
                                    %output: memref<4xi32>, %rhs: i32) {
    ADORA.kernel {
      %loaded = affine.load %input[0] : memref<4xi32>
      %sum = arith.addi %loaded, %rhs : i32
      %difference = arith.subi %sum, %rhs : i32
      affine.store %difference, %output[1] : memref<4xi32>
      ADORA.terminator
    } {KernelName = "cf_direct_affine_baseline"}
    return
  }

  func.func @unrelated_direct_memory(%input: memref<8xi32>,
                                     %output: memref<8xi32>, %index: index,
                                     %value: i32, %cond: i1) {
    ADORA.kernel {
      ADORA.cond_store %value, %output[%index] if %cond : memref<8xi32>
      %loaded = memref.load %input[%index] : memref<8xi32>
      memref.store %loaded, %output[%index] : memref<8xi32>
      ADORA.terminator
    } {KernelName = "cf_unrelated_memory"}
    return
  }

  func.func @mixed_constant_address(%cond: i1, %then_value: i32,
                                    %else_value: i32,
                                    %output: memref<8xi32>) {
    %c3 = arith.constant 3 : index
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          affine.store %then_value, %output[3] : memref<8xi32>
        } else {
          memref.store %else_value, %output[%c3] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_mixed_constant"}
    return
  }

  func.func @fallback_rank_two(%cond: i1, %value: i32,
                               %output: memref<4x4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ADORA.kernel {
      scf.if %cond {
      } else {
        affine.store %value, %output[0, 1] : memref<4x4xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_fallback_rank_two"}
    return
  }

  func.func @crossed_store_order(%cond: i1, %then_a: i32, %then_b: i32,
                                 %else_b: i32, %else_a: i32,
                                 %output: memref<16xi32>, %i: index,
                                 %j: index) {
    ADORA.kernel {
      scf.if %cond {
        memref.store %then_a, %output[%i] : memref<16xi32>
        memref.store %then_b, %output[%j] : memref<16xi32>
      } else {
        memref.store %else_b, %output[%j] : memref<16xi32>
        memref.store %else_a, %output[%i] : memref<16xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_crossed_order"}
    return
  }
}

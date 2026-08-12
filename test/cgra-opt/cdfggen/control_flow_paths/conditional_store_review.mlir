// RUN: rm -f cf_ordered_store_CDFG.dot cf_mapped_producers_CDFG.dot cf_mixed_constant_CDFG.dot cf_fallback_rank_two_CDFG.dot cf_crossed_order_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_ordered_store_CDFG.dot)" -eq 2
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_mapped_producers_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_mixed_constant_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_crossed_order_CDFG.dot)" -eq 2
// RUN: %FileCheck %s --check-prefix=ORDER-DOT --input-file=cf_ordered_store_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=PRODUCER-DOT --input-file=cf_mapped_producers_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=CONSTANT-DOT --input-file=cf_mixed_constant_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=CROSSED-DOT --input-file=cf_crossed_order_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: rm -f cf_ordered_store_CDFG.dot cf_mapped_producers_CDFG.dot cf_mixed_constant_CDFG.dot cf_fallback_rank_two_CDFG.dot cf_crossed_order_CDFG.dot

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

// CHECK-LABEL: func.func @mapped_producers
// CHECK: ADORA.kernel
// CHECK: %[[INDEX:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: %[[VALUE:.*]] = arith.xori %{{.*}}, %{{.*}} : i32
// CHECK: ADORA.cond_store %[[VALUE]], %{{.*}}[%[[INDEX]]] if %{{.*}} : memref<16xi32>
// CHECK: ADORA.terminator

// PRODUCER-DOT: Digraph G {
// PRODUCER-DOT-DAG: Input[[RAW:[0-9]+]][opcode = "Input", ref_name="cf_mapped_producers:arg0"
// PRODUCER-DOT-DAG: Input[[COND:[0-9]+]][opcode = "Input", ref_name="cf_mapped_producers:arg3"
// PRODUCER-DOT-DAG: XOR[[VALUE:[0-9]+]][opcode = "XOR"
// PRODUCER-DOT-DAG: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// PRODUCER-DOT-DAG: MUL[[BYTE_ADDR:[0-9]+]][opcode = "MUL"
// PRODUCER-DOT-DAG: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE"
// PRODUCER-DOT-DAG: XOR[[VALUE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// PRODUCER-DOT-DAG: Input[[RAW]] -> MUL[[BYTE_ADDR]]
// PRODUCER-DOT-DAG: CONST[[FOUR]] -> MUL[[BYTE_ADDR]]
// PRODUCER-DOT-DAG: MUL[[BYTE_ADDR]] -> CSTORE[[STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// PRODUCER-DOT-DAG: Input[[COND]] -> CSTORE[[STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// PRODUCER-DOT: }

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
// CHECK: memref.load
// CHECK: arith.select
// CHECK: memref.store
// CHECK: ADORA.terminator

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

  func.func @mapped_producers(%raw_index: i32, %value: i32,
                              %output: memref<16xi32>, %cond: i1) {
    ADORA.kernel {
      %index = arith.index_cast %raw_index : i32 to index
      %c1 = arith.constant 1 : i32
      %stored_value = arith.xori %value, %c1 : i32
      ADORA.cond_store %stored_value, %output[%index] if %cond : memref<16xi32>
      ADORA.terminator
    } {KernelName = "cf_mapped_producers"}
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
        memref.store %value, %output[%c0, %c1] : memref<4x4xi32>
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

// RUN: rm -f cf_else_only_CDFG.dot cf_same_address_CDFG.dot cf_different_address_CDFG.dot cf_store_else_if_CDFG.dot cf_store_nested_CDFG.dot cf_direct_i8_CDFG.dot cf_distinct_memrefs_CDFG.dot cf_result_store_CDFG.dot cf_mixed_address_CDFG.dot cf_direct_dynamic_CDFG.dot cf_outside_affine_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_else_only_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_same_address_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_different_address_CDFG.dot)" -eq 2
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_store_else_if_CDFG.dot)" -eq 3
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_store_nested_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_result_store_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_direct_i8_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"MUL\"' cf_direct_i8_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_distinct_memrefs_CDFG.dot)" -eq 2
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_mixed_address_CDFG.dot)" -eq 0
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_direct_dynamic_CDFG.dot)" -eq 1
// RUN: test "$(grep -c 'opcode = \"CSTORE\"' cf_outside_affine_CDFG.dot)" -eq 1
// RUN: %FileCheck %s --check-prefix=ELSE-DOT --input-file=cf_else_only_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=SAME-DOT --input-file=cf_same_address_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=DIFF-DOT --input-file=cf_different_address_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=ELSEIF-DOT --input-file=cf_store_else_if_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=NESTED-DOT --input-file=cf_store_nested_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=I8-DOT --input-file=cf_direct_i8_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL'
// RUN: %FileCheck %s --check-prefix=METADATA-DOT --input-file=cf_distinct_memrefs_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=RESULT-DOT --input-file=cf_result_store_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=DYNAMIC-DOT --input-file=cf_direct_dynamic_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: %FileCheck %s --check-prefix=AFFINE-DOT --input-file=cf_outside_affine_CDFG.dot --implicit-check-not='opcode = "undefined"' --implicit-check-not='opcode = "CTRL"'
// RUN: rm -f cf_else_only_CDFG.dot cf_same_address_CDFG.dot cf_different_address_CDFG.dot cf_store_else_if_CDFG.dot cf_store_nested_CDFG.dot cf_direct_i8_CDFG.dot cf_distinct_memrefs_CDFG.dot cf_result_store_CDFG.dot cf_mixed_address_CDFG.dot cf_direct_dynamic_CDFG.dot cf_outside_affine_CDFG.dot

// CHECK-LABEL: func.func @else_only(
// CHECK-SAME: %[[ELSE_PARENT:[a-zA-Z0-9]+]]: i1
// CHECK: ADORA.kernel
// CHECK-NOT: memref.load
// CHECK-NOT: scf.if
// CHECK: %[[ELSE_FALSE:.*]] = arith.constant false
// CHECK: %[[ELSE_TRUE:.*]] = arith.constant true
// CHECK: %[[NOT:.*]] = arith.select %[[ELSE_PARENT]], %[[ELSE_FALSE]], %[[ELSE_TRUE]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[NOT]] : memref<8xi32>
// CHECK-NOT: memref.load
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @same_address
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-NOT: ADORA.cond_store
// CHECK: %[[VALUE:.*]] = arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: affine.store %[[VALUE]], %{{.*}}[0] : memref<8xi32>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @different_address
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK-COUNT-2: ADORA.cond_store
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @store_else_if(
// CHECK-SAME: %[[ELSEIF_OUTER:[a-zA-Z0-9]+]]: i1, %[[ELSEIF_INNER:[a-zA-Z0-9]+]]: i1
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK: %[[INNER_FALSE:.*]] = arith.constant false
// CHECK: %[[INNER_TRUE:.*]] = arith.constant true
// CHECK: %[[NOT_INNER:.*]] = arith.select %[[ELSEIF_INNER]], %[[INNER_FALSE]], %[[INNER_TRUE]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[ELSEIF_OUTER]] : memref<8xi32>
// CHECK: %[[ELSEIF_FALSE:.*]] = arith.constant false
// CHECK: %[[ELSE_PATH:.*]] = arith.select %[[ELSEIF_OUTER]], %[[ELSEIF_FALSE]], %[[ELSEIF_INNER]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[ELSE_PATH]] : memref<8xi32>
// CHECK: %[[FINAL_PATH:.*]] = arith.select %[[ELSEIF_OUTER]], %[[ELSEIF_FALSE]], %[[NOT_INNER]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[FINAL_PATH]] : memref<8xi32>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @store_nested(
// CHECK-SAME: %[[NESTED_OUTER:[a-zA-Z0-9]+]]: i1, %[[NESTED_INNER:[a-zA-Z0-9]+]]: i1
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK: %[[NESTED_FALSE:.*]] = arith.constant false
// CHECK: %[[NESTED_PATH:.*]] = arith.select %[[NESTED_OUTER]], %[[NESTED_INNER]], %[[NESTED_FALSE]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[NESTED_PATH]] : memref<8xi32>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @direct_i8
// CHECK: ADORA.kernel
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %{{.*}} : memref<8xi8>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @result_with_store(
// CHECK-SAME: %[[RESULT_OUTER:[a-zA-Z0-9]+]]: i1, %[[RESULT_INNER:[a-zA-Z0-9]+]]: i1
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK: %[[RESULT_FALSE:.*]] = arith.constant false
// CHECK: %[[STORE_PATH:.*]] = arith.select %[[RESULT_OUTER]], %[[RESULT_INNER]], %[[RESULT_FALSE]] : i1
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %[[STORE_PATH]] : memref<8xi32>
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @mixed_address
// CHECK: ADORA.kernel
// CHECK-NOT: ADORA.cond_store
// CHECK: arith.select %{{.*}}, %{{.*}}, %{{.*}} : i32
// CHECK: affine.store
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @direct_dynamic
// CHECK: ADORA.kernel
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %{{.*}} : memref<8xi32>
// CHECK: ADORA.terminator

// CHECK-LABEL: func.func @outside_affine
// CHECK: ADORA.kernel
// CHECK-NOT: scf.if
// CHECK: arith.muli
// CHECK: arith.addi
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %{{.*}} : memref<32xi32>
// CHECK: ADORA.terminator

// ELSE-DOT: Digraph G {
// ELSE-DOT-DAG: Input[[PARENT:[0-9]+]][opcode = "Input", ref_name="cf_else_only:arg0", size="1", offset="0,0", pattern="0,1"
// ELSE-DOT-DAG: Input[[VALUE:[0-9]+]][opcode = "Input", ref_name="cf_else_only:arg1", size="4", offset="0,0", pattern="0,1"
// ELSE-DOT-DAG: CONST[[FALSE:[0-9]+]][opcode = "CONST", value="0x00"
// ELSE-DOT-DAG: CONST[[TRUE:[0-9]+]][opcode = "CONST", value="0x01"
// ELSE-DOT-DAG: CSTORE[[STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_else_only:arg2", size="32", offset="0,0", pattern="0,1"
// ELSE-DOT-DAG: CONST[[FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// ELSE-DOT-DAG: MUL[[ADDR:[0-9]+]][opcode = "MUL"
// ELSE-DOT-DAG: SEL[[ENABLE:[0-9]+]][opcode = "SEL"
// ELSE-DOT-DAG: Input[[VALUE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// ELSE-DOT-DAG: MUL[[ADDR]] -> CSTORE[[STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// ELSE-DOT-DAG: SEL[[ENABLE]] -> CSTORE[[STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// ELSE-DOT-DAG: CONST[[TRUE]] -> SEL[[ENABLE]]{{[^]]*}}operand = 0, label = "Op=0"
// ELSE-DOT-DAG: CONST[[FALSE]] -> SEL[[ENABLE]]{{[^]]*}}operand = 1, label = "Op=1"
// ELSE-DOT-DAG: Input[[PARENT]] -> SEL[[ENABLE]]{{[^]]*}}operand = 2, label = "Op=2"
// ELSE-DOT-DAG: CONST[[FOUR]] -> MUL[[ADDR]]
// ELSE-DOT: }

// SAME-DOT: Digraph G {
// SAME-DOT-DAG: SEL{{[0-9]+}}[opcode = "SEL"
// SAME-DOT-DAG: Output{{[0-9]+}}[opcode = "Output"
// SAME-DOT-NOT: opcode = "CSTORE"
// SAME-DOT: }

// DIFF-DOT: Digraph G {
// DIFF-DOT-COUNT-2: opcode = "CSTORE"
// DIFF-DOT: }

// ELSEIF-DOT: Digraph G {
// ELSEIF-DOT-DAG: SEL{{[0-9]+}}[opcode = "SEL"
// ELSEIF-DOT-COUNT-3: opcode = "CSTORE"
// ELSEIF-DOT-COUNT-3: -> CSTORE{{[0-9]+}}{{[^]]*}}operand = 2, label = "Op=2"
// ELSEIF-DOT: }

// NESTED-DOT: Digraph G {
// NESTED-DOT-DAG: CSTORE[[NESTED_STORE:[0-9]+]][opcode = "CSTORE"
// NESTED-DOT-DAG: SEL[[NESTED_ENABLE:[0-9]+]][opcode = "SEL"
// NESTED-DOT-DAG: SEL[[NESTED_ENABLE]] -> CSTORE[[NESTED_STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// NESTED-DOT: }

// I8-DOT: Digraph G {
// I8-DOT-DAG: Input[[I8_VALUE:[0-9]+]][opcode = "Input", ref_name="cf_direct_i8:arg0", size="1", offset="0,0", pattern="0,1"
// I8-DOT-DAG: Input[[I8_COND:[0-9]+]][opcode = "Input", ref_name="cf_direct_i8:arg2", size="1", offset="0,0", pattern="0,1"
// I8-DOT-DAG: CONST[[I8_ADDR:[0-9]+]][opcode = "CONST"
// I8-DOT-DAG: CSTORE[[I8_STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_direct_i8:arg1", size="8", offset="0,0", pattern="0,1"
// I8-DOT-DAG: Input[[I8_VALUE]] -> CSTORE[[I8_STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// I8-DOT-DAG: CONST[[I8_ADDR]] -> CSTORE[[I8_STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// I8-DOT-DAG: Input[[I8_COND]] -> CSTORE[[I8_STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// I8-DOT-NOT: opcode = "MUL"
// I8-DOT: }

// METADATA-DOT: Digraph G {
// METADATA-DOT-DAG: CSTORE[[FIRST_STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_distinct_memrefs:arg2", size="32", offset="0,0", pattern="0,1"
// METADATA-DOT-DAG: CSTORE[[SECOND_STORE:[0-9]+]][opcode = "CSTORE", ref_name="cf_distinct_memrefs:arg3", size="64", offset="0,0", pattern="0,1"
// METADATA-DOT: }

// RESULT-DOT: Digraph G {
// RESULT-DOT: CSTORE{{[0-9]+}}[opcode = "CSTORE"
// RESULT-DOT: }

// DYNAMIC-DOT: Digraph G {
// DYNAMIC-DOT-DAG: Input[[DYN_VALUE:[0-9]+]][opcode = "Input", ref_name="cf_direct_dynamic:arg0"
// DYNAMIC-DOT-DAG: Input[[DYN_INDEX:[0-9]+]][opcode = "Input", ref_name="cf_direct_dynamic:arg2"
// DYNAMIC-DOT-DAG: Input[[DYN_COND:[0-9]+]][opcode = "Input", ref_name="cf_direct_dynamic:arg3"
// DYNAMIC-DOT-DAG: CONST[[DYN_FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// DYNAMIC-DOT-DAG: MUL[[DYN_ADDR:[0-9]+]][opcode = "MUL"
// DYNAMIC-DOT-DAG: CSTORE[[DYN_STORE:[0-9]+]][opcode = "CSTORE"
// DYNAMIC-DOT-DAG: Input[[DYN_VALUE]] -> CSTORE[[DYN_STORE]]{{[^]]*}}operand = 0, label = "Op=0"
// DYNAMIC-DOT-DAG: Input[[DYN_INDEX]] -> MUL[[DYN_ADDR]]
// DYNAMIC-DOT-DAG: CONST[[DYN_FOUR]] -> MUL[[DYN_ADDR]]
// DYNAMIC-DOT-DAG: MUL[[DYN_ADDR]] -> CSTORE[[DYN_STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// DYNAMIC-DOT-DAG: Input[[DYN_COND]] -> CSTORE[[DYN_STORE]]{{[^]]*}}operand = 2, label = "Op=2"
// DYNAMIC-DOT: }

// AFFINE-DOT: Digraph G {
// AFFINE-DOT-DAG: Input[[AFF_INDEX:[0-9]+]][opcode = "Input", ref_name="cf_outside_affine:arg2"
// AFFINE-DOT-DAG: CONST[[AFF_TWO:[0-9]+]][opcode = "CONST", value="0x00000002"
// AFFINE-DOT-DAG: CONST[[AFF_ONE:[0-9]+]][opcode = "CONST", value="0x00000001"
// AFFINE-DOT-DAG: CONST[[AFF_FOUR:[0-9]+]][opcode = "CONST", value="0x00000004"
// AFFINE-DOT-DAG: MUL[[AFF_EXPR_MUL:[0-9]+]][opcode = "MUL"
// AFFINE-DOT-DAG: ADD[[AFF_EXPR_ADD:[0-9]+]][opcode = "ADD"
// AFFINE-DOT-DAG: MUL[[AFF_BYTE_MUL:[0-9]+]][opcode = "MUL"
// AFFINE-DOT-DAG: CSTORE[[AFF_STORE:[0-9]+]][opcode = "CSTORE"
// AFFINE-DOT-DAG: Input[[AFF_INDEX]] -> MUL[[AFF_EXPR_MUL]]
// AFFINE-DOT-DAG: CONST[[AFF_TWO]] -> MUL[[AFF_EXPR_MUL]]
// AFFINE-DOT-DAG: MUL[[AFF_EXPR_MUL]] -> ADD[[AFF_EXPR_ADD]]
// AFFINE-DOT-DAG: CONST[[AFF_ONE]] -> ADD[[AFF_EXPR_ADD]]
// AFFINE-DOT-DAG: ADD[[AFF_EXPR_ADD]] -> MUL[[AFF_BYTE_MUL]]
// AFFINE-DOT-DAG: CONST[[AFF_FOUR]] -> MUL[[AFF_BYTE_MUL]]
// AFFINE-DOT-DAG: MUL[[AFF_BYTE_MUL]] -> CSTORE[[AFF_STORE]]{{[^]]*}}operand = 1, label = "Op=1"
// AFFINE-DOT: }

module {
  func.func @else_only(%cond: i1, %value: i32, %output: memref<8xi32>) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
        } else {
          memref.store %value, %output[%c0] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_else_only"}
    return
  }

  func.func @same_address(%cond: i1, %then_value: i32, %else_value: i32,
                          %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          affine.store %then_value, %output[0] : memref<8xi32>
        } else {
          affine.store %else_value, %output[0] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_same_address"}
    return
  }

  func.func @different_address(%cond: i1, %then_value: i32, %else_value: i32,
                               %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          affine.store %then_value, %output[0] : memref<8xi32>
        } else {
          affine.store %else_value, %output[1] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_different_address"}
    return
  }

  func.func @store_else_if(%a: i1, %b: i1, %outer_value: i32,
                           %inner_value: i32, %else_value: i32,
                           %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %a {
          affine.store %outer_value, %output[0] : memref<8xi32>
        } else {
          scf.if %b {
            affine.store %inner_value, %output[1] : memref<8xi32>
          } else {
            affine.store %else_value, %output[2] : memref<8xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "cf_store_else_if"}
    return
  }

  func.func @store_nested(%a: i1, %b: i1, %value: i32,
                          %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %a {
          scf.if %b {
            affine.store %value, %output[0] : memref<8xi32>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "cf_store_nested"}
    return
  }

  func.func @direct_i8(%value: i8, %output: memref<8xi8>, %cond: i1) {
    %c0 = arith.constant 0 : index
    ADORA.kernel {
      ADORA.cond_store %value, %output[%c0] if %cond : memref<8xi8>
      ADORA.terminator
    } {KernelName = "cf_direct_i8"}
    return
  }

  func.func @direct_distinct_memrefs(%cond: i1, %value: i32,
                                     %first: memref<8xi32>,
                                     %second: memref<16xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    ADORA.kernel {
      ADORA.cond_store %value, %first[%c0] if %cond : memref<8xi32>
      ADORA.cond_store %value, %second[%c1] if %cond : memref<16xi32>
      ADORA.terminator
    } {KernelName = "cf_distinct_memrefs"}
    return
  }

  func.func @result_with_store(%outer: i1, %inner: i1, %value: i32,
                               %other: i32, %output: memref<8xi32>) {
    ADORA.kernel {
      %selected = scf.if %outer -> (i32) {
        scf.if %inner {
          affine.store %value, %output[0] : memref<8xi32>
        }
        scf.yield %value : i32
      } else {
        scf.yield %other : i32
      }
      %used = arith.addi %selected, %value : i32
      ADORA.terminator
    } {KernelName = "cf_result_store"}
    return
  }

  func.func @mixed_address(%cond: i1, %then_value: i32, %else_value: i32,
                           %output: memref<8xi32>) {
    ADORA.kernel {
      affine.for %i = 0 to 8 {
        scf.if %cond {
          affine.store %then_value, %output[%i] : memref<8xi32>
        } else {
          memref.store %else_value, %output[%i] : memref<8xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "cf_mixed_address"}
    return
  }

  func.func @direct_dynamic(%value: i32, %output: memref<8xi32>, %index: index,
                            %cond: i1) {
    ADORA.kernel {
      ADORA.cond_store %value, %output[%index] if %cond : memref<8xi32>
      ADORA.terminator
    } {KernelName = "cf_direct_dynamic"}
    return
  }

  func.func @outside_affine(%cond: i1, %value: i32, %index: index,
                            %output: memref<32xi32>) {
    ADORA.kernel {
      scf.if %cond {
        affine.store %value, %output[2 * %index + 1] : memref<32xi32>
      }
      ADORA.terminator
    } {KernelName = "cf_outside_affine"}
    return
  }
}

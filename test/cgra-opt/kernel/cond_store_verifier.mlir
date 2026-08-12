// RUN: not %cgra-opt %s -split-input-file 2>&1 | %FileCheck %s

// CHECK: error: 'ADORA.cond_store' op failed to verify that type of 'value' matches element type of 'memref'
func.func @value_type_mismatch(%value: f32, %buffer: memref<16xi32>,
                               %index: index, %condition: i1) {
  "ADORA.cond_store"(%value, %buffer, %index, %condition) : (f32, memref<16xi32>, index, i1) -> ()
  return
}

// -----

// CHECK: error: 'ADORA.cond_store' op requires a rank-1 memref
func.func @rank_mismatch(%value: i32, %buffer: memref<4x4xi32>,
                         %index: index, %condition: i1) {
  ADORA.cond_store %value, %buffer[%index] if %condition : memref<4x4xi32>
  return
}

// -----

// CHECK: error: 'ADORA.cond_store' op requires exactly one index
func.func @too_many_indices(%value: i32, %buffer: memref<16xi32>,
                            %index0: index, %index1: index, %condition: i1) {
  "ADORA.cond_store"(%value, %buffer, %index0, %index1, %condition) : (i32, memref<16xi32>, index, index, i1) -> ()
  return
}

// -----

// CHECK: error: 'ADORA.cond_store' op requires exactly one index
func.func @zero_indices(%value: i32, %buffer: memref<16xi32>,
                        %condition: i1) {
  "ADORA.cond_store"(%value, %buffer, %condition) : (i32, memref<16xi32>, i1) -> ()
  return
}

// -----

// CHECK: error: 'ADORA.cond_store' op operand #3 must be 1-bit signless integer, but got 'i32'
func.func @condition_must_be_i1(%value: i32, %buffer: memref<16xi32>,
                                %index: index, %condition: i32) {
  "ADORA.cond_store"(%value, %buffer, %index, %condition) : (i32, memref<16xi32>, index, i32) -> ()
  return
}

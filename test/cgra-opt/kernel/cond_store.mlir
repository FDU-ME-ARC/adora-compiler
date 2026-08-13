// RUN: %cgra-opt %s | %cgra-opt | %FileCheck %s

// CHECK-LABEL: func.func @cond_store_roundtrip
// CHECK: ADORA.cond_store %{{.*}}, %{{.*}}[%{{.*}}] if %{{.*}} : memref<16xi32>
func.func @cond_store_roundtrip(%value: i32, %buffer: memref<16xi32>,
                               %index: index, %condition: i1) {
  ADORA.cond_store %value, %buffer[%index] if %condition : memref<16xi32>
  return
}

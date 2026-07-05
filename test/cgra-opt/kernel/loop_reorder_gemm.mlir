// RUN: %cgra-opt --adora-loop-reorder %s | %FileCheck %s

// The reuse-group memory-access cost model sinks the loop with the cheapest
// innermost cost to the innermost position. For this i,j,k gemm the original
// %j loop is cheapest as innermost (B[k,j] and C[i,j] become unit-stride),
// so the nest is reordered i,j,k -> i,k,j.

// CHECK-LABEL: func.func @gemm
// CHECK:         affine.for %[[I:.+]] = 0 to 64 {
// CHECK-NEXT:      affine.for %[[K:.+]] = 0 to 64 {
// CHECK-NEXT:        affine.for %[[J:.+]] = 0 to 64 {
// CHECK-DAG:           affine.load %arg0[%[[I]], %[[K]]]
// CHECK-DAG:           affine.load %arg1[%[[K]], %[[J]]]
// CHECK-DAG:           affine.load %arg2[%[[I]], %[[J]]]
// CHECK:               affine.store %{{.+}}, %arg2[%[[I]], %[[J]]]

func.func @gemm(%A: memref<64x64xi32>, %B: memref<64x64xi32>, %C: memref<64x64xi32>) {
  affine.for %i = 0 to 64 {
    affine.for %j = 0 to 64 {
      affine.for %k = 0 to 64 {
        %a = affine.load %A[%i, %k] : memref<64x64xi32>
        %b = affine.load %B[%k, %j] : memref<64x64xi32>
        %c = affine.load %C[%i, %j] : memref<64x64xi32>
        %m = arith.muli %a, %b : i32
        %s = arith.addi %c, %m : i32
        affine.store %s, %C[%i, %j] : memref<64x64xi32>
      }
    }
  }
  return
}

// FileCheck expectations for 01_gemm after --adora-loop-reorder.
// Reordered nest is i,k,j (outer -> inner). Loads become:
//   A[i,k]  B[k,j]  C[i,j]   with %arg3=i, %arg4=k, %arg5=j.

// CHECK-LABEL: func.func @gemm
// CHECK:         affine.for %[[I:.+]] = 0 to 64 {
// CHECK-NEXT:      affine.for %[[K:.+]] = 0 to 64 {
// CHECK-NEXT:        affine.for %[[J:.+]] = 0 to 64 {
// CHECK-DAG:           affine.load %arg0[%[[I]], %[[K]]]
// CHECK-DAG:           affine.load %arg1[%[[K]], %[[J]]]
// CHECK-DAG:           affine.load %arg2[%[[I]], %[[J]]]
// CHECK:               affine.store %{{.+}}, %arg2[%[[I]], %[[J]]]

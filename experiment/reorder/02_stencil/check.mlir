// FileCheck expectations for 02_stencil after --adora-loop-reorder.
// Nest is reordered j,i -> i,j so the unit-stride loop (j) is innermost.
// After reorder: %arg2 = i (outer), %arg3 = j (inner).

// CHECK-LABEL: func.func @row_copy
// CHECK:         affine.for %[[I:.+]] = 0 to 128 {
// CHECK-NEXT:      affine.for %[[J:.+]] = 0 to 128 {
// CHECK:             affine.load %arg0[%[[I]], %[[J]]]
// CHECK:             affine.store %{{.+}}, %arg1[%[[I]], %[[J]]]

// RUN: %cgra-opt --adora-simplify-affine-loop-levels %s | %FileCheck %s

// CHECK-LABEL: func.func @single_trip_const
// CHECK-NOT: affine.for
// CHECK: %[[V:.+]] = affine.load %arg0[%{{.+}}] : memref<4xf32>
// CHECK: %[[SUM:.+]] = arith.addf %{{.+}}, %[[V]] : f32
// CHECK: return %[[SUM]]
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (d0 + 1)>

func.func @single_trip_const(%arg0: memref<4xf32>) -> f32 {
  %c0 = arith.constant 0 : index
  %cst = arith.constant 0.000000e+00 : f32
  %0 = affine.for %arg1 = 0 to 1 iter_args(%arg2 = %cst) -> (f32) {
    %1 = affine.load %arg0[%c0] : memref<4xf32>
    %2 = arith.addf %arg2, %1 : f32
    affine.yield %2 : f32
  }
  return %0 : f32
}

// CHECK-LABEL: func.func @single_trip_dynamic_lb
// CHECK-NOT: affine.for
// CHECK: %[[IV:.+]] = affine.apply #map
// CHECK: %[[V:.+]] = affine.load %arg1[%[[IV]]] : memref<16xf32>
// CHECK: return %{{.+}} : f32
func.func @single_trip_dynamic_lb(%arg0: index, %arg1: memref<16xf32>) -> f32 {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = affine.for %arg2 = #map(%arg0) to #map1(%arg0) iter_args(%arg3 = %cst) -> (f32) {
    %1 = affine.load %arg1[%arg2] : memref<16xf32>
    %2 = arith.addf %arg3, %1 : f32
    affine.yield %2 : f32
  }
  return %0 : f32
}

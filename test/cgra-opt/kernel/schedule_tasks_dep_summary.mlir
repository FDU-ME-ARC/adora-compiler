// End-to-end test for P1.0 + P4.0:
//   1. Run the full pre-schedule pipeline to produce ADORA.BlockLoad/Store.
//   2. Run --adora-schedule-tasks which fills analyzeDependencyInGraph and
//      attaches `adora.dep_summary` to the host FuncOp.
//   3. FileCheck that the grouped attribute schema is present, with at least
//      one RAR, one WAR, and one WAW edge — mirroring what the three new
//      checkers (checkDependencyBetweenBlockLoadAndBlockLoad / ...StoreAndBlockStore
//      / ...LoadAndBlockStore) are supposed to detect for gemm.
//
// RUN: %cgra-opt \
// RUN:   --adora-extract-affine-for-to-kernel \
// RUN:   --adora-simplify-loadstore \
// RUN:   --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
// RUN:   --adora-schedule-tasks \
// RUN:   %s 2>/dev/null | %FileCheck %s

// CHECK: adora.scheduled
// CHECK: adora.dep_summary = [{block_idx = 0 : i64, edges = [

// CHECK-DAG: kind = "RAR"
// CHECK-DAG: kind = "WAR"
// CHECK-DAG: kind = "WAW"

// CHECK-SAME: ]}]

module {
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) {
    affine.for %i = 0 to 20 {
      affine.for %j = 0 to 25 {
        %v = affine.load %arg0[%i, %j] : memref<?x25xf32>
        %c = arith.constant 1.200000e+00 : f32
        %m = arith.mulf %v, %c : f32
        affine.store %m, %arg0[%i, %j] : memref<?x25xf32>
      }
    }
    affine.for %i = 0 to 20 {
      affine.for %j = 0 to 25 {
        %0 = affine.load %arg0[%i, %j] : memref<?x25xf32>
        %acc = affine.for %k = 0 to 30 iter_args(%s = %0) -> (f32) {
          %a = affine.load %arg1[%i, %k] : memref<?x30xf32>
          %b = affine.load %arg2[%k, %j] : memref<?x25xf32>
          %p = arith.mulf %a, %b : f32
          %ns = arith.addf %s, %p : f32
          affine.yield %ns : f32
        }
        affine.store %acc, %arg0[%i, %j] : memref<?x25xf32>
      }
    }
    return
  }
}

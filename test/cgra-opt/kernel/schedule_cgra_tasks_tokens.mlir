// PR2 commit B — verify that --adora-schedule-tasks threads SSA
// !ADORA.token values when emit-token=true, and that emit-token=false
// preserves the PR1 adora.dep_summary only path for byte-level baseline
// compatibility (without tokens downstream defaults to serial execution,
// so the sync form remains semantically safe).
//
// RUN: %cgra-opt \
// RUN:   --adora-extract-affine-for-to-kernel \
// RUN:   --adora-simplify-loadstore \
// RUN:   --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
// RUN:   --adora-schedule-tasks="emit-token=true" \
// RUN:   %s 2>/dev/null | %FileCheck %s --check-prefix=TOKEN

// RUN: %cgra-opt \
// RUN:   --adora-extract-affine-for-to-kernel \
// RUN:   --adora-simplify-loadstore \
// RUN:   --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
// RUN:   --adora-schedule-tasks="emit-token=false" \
// RUN:   %s 2>/dev/null | %FileCheck %s --check-prefix=NOTOKEN

// With emit-token=true we expect !ADORA.token to surface on at least one
// op as an SSA value connecting a producer to a consumer; the dep_summary
// attribute is still emitted because emit-summary defaults to true.
// TOKEN: adora.scheduled
// TOKEN: adora.dep_summary
// TOKEN: !ADORA.token

// With emit-token=false the IR must be free of any !ADORA.token mention;
// only the PR1 string-based dep_summary carries ordering information.
// NOTOKEN-NOT: !ADORA.token
// NOTOKEN: adora.scheduled
// NOTOKEN: adora.dep_summary

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

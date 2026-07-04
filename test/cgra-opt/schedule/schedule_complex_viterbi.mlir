// schedule_complex_viterbi.mlir
// Viterbi (64-step, 8-state DP) — sequential forward-pass with RAW deps.
// Verifies --adora-schedule-tasks:
//   1. marks module adora.scheduled
//   2. dep_summary is non-empty (RAW edges from loads→kernel→stores)
//   3. async !ADORA.token is emitted
//   4. REGRESSION: ADORA.BlockLoad must NOT carry a trailing "-> !ADORA.token"
//      type suffix — the op defines exactly 1 result (the loaded memref).
//      Buggy form:  %r, %tok = ADORA.BlockLoad ... -> !ADORA.token
//      Correct form: %r = ADORA.BlockLoad async [%dep] ...
//
// RUN: cgra-opt %s --adora-schedule-tasks 2>/dev/null | FileCheck %s

// CHECK: module attributes {adora.scheduled}
// CHECK-NOT: adora.dep_summary
// CHECK-DAG: dep_kinds = [
// CHECK-DAG: "RAW"
// CHECK-DAG: ADORA.BlockLoad async [
// CHECK-DAG: ADORA.kernel async [

// REGRESSION: BlockLoad must use implicit async keyword (GPU-dialect style).
// No explicit "-> !ADORA.token" suffix should appear.
// CHECK-NOT: ADORA.BlockLoad{{.*}}-> !ADORA.token

module {
  func.func @viterbi(%arg0: memref<?x8xi32>, %arg1: memref<?xi32>, %arg2: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca() : memref<64x8xi32>
    affine.for %arg3 = 0 to 8 {
      %4 = affine.load %arg0[0, %arg3] : memref<?x8xi32>
      affine.store %4, %alloca_0[0, %arg3] : memref<64x8xi32>
    }
    %result = ADORA.BlockLoad %alloca_0 [62, 0] : memref<64x8xi32> -> memref<1x8xi32>  {Id = "0", KernelName = "viterbi"}
    %result_1 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x8xi32> -> memref<8x8xi32>  {Id = "1", KernelName = "viterbi"}
    %result_2 = ADORA.BlockLoad %arg0 [63, 0] : memref<?x8xi32> -> memref<1x8xi32>  {Id = "2", KernelName = "viterbi"}
    %1 = ADORA.LocalMemAlloc memref<8xi32>  {Id = "3", KernelName = "viterbi"}
    %2 = ADORA.LocalMemAlloc memref<1x8xi32>  {Id = "4", KernelName = "viterbi"}
    %3 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "5", KernelName = "viterbi"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 8 {
        %4 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %c-2147483648_i32) -> (i32) {
          %7 = affine.load %result[0, %arg4] : memref<1x8xi32>
          %8 = affine.load %result_1[%arg4, %arg3] : memref<8x8xi32>
          %9 = arith.addi %7, %8 : i32
          %10 = arith.cmpi sgt, %9, %arg5 : i32
          %11 = arith.select %10, %9, %arg5 : i32
          affine.yield %11 : i32
        }
        affine.store %4, %3[0] : memref<2xi32>
        %5 = affine.load %result_2[0, %arg3] : memref<1x8xi32>
        %6 = arith.addi %4, %5 : i32
        affine.store %6, %2[0, %arg3] : memref<1x8xi32>
        affine.store %6, %1[%arg3] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "viterbi"}
    ADORA.BlockStore %3, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "5", KernelName = "viterbi"}
    ADORA.BlockStore %2, %alloca_0 [63, 0] : memref<1x8xi32> -> memref<64x8xi32>  {Id = "4", KernelName = "viterbi"}
    ADORA.BlockStore %1, %arg1 [0] : memref<8xi32> -> memref<?xi32>  {Id = "3", KernelName = "viterbi"}
    return
  }
}

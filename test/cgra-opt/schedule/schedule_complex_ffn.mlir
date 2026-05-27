// schedule_complex_ffn.mlir
// FFN: two-layer FC chain  ffn_fc1 --(RAW)--> ffn_fc2  (also tiled @ffn).
// Verifies --adora-schedule-tasks:
//   1. marks module adora.scheduled
//   2. dep_summary contains RAW edges for all three functions
//   3. ADORA.kernel async [...] threads tokens between BlockLoad and kernel
//   4. REGRESSION: ADORA.BlockLoad must NOT carry trailing "-> !ADORA.token"
//      type suffix.  The op defines exactly 1 result (the loaded memref).
//      Buggy:   %r, %tok = ADORA.BlockLoad ... -> !ADORA.token
//      Correct: %r = ADORA.BlockLoad async [%dep] ...
//
// RUN: cgra-opt %s --adora-schedule-tasks 2>/dev/null | FileCheck %s

// CHECK: module attributes {adora.scheduled}
// CHECK: adora.dep_summary
// CHECK-NOT: adora.dep_summary = []
// CHECK: kind = "RAW"
// CHECK: ADORA.BlockLoad async [
// CHECK: ADORA.kernel async [

// REGRESSION: BlockLoad must use implicit async keyword (GPU-dialect style).
// No explicit "-> !ADORA.token" suffix should appear.
// CHECK-NOT: ADORA.BlockLoad{{.*}}-> !ADORA.token

module {
  func.func @ffn_fc1(%arg0: memref<?x32xi32>, %arg1: memref<?x64xi32>, %arg2: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<?x32xi32> -> memref<16x32xi32>  {Id = "0", KernelName = "ffn_fc1"}
    %result_0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x64xi32> -> memref<32x64xi32>  {Id = "1", KernelName = "ffn_fc1"}
    %1 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "ffn_fc1"}
    %2 = ADORA.LocalMemAlloc memref<16x64xi32>  {Id = "3", KernelName = "ffn_fc1"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 64 {
          %3 = affine.for %arg5 = 0 to 32 iter_args(%arg6 = %c0_i32) -> (i32) {
            %6 = affine.load %result[%arg3, %arg5] : memref<16x32xi32>
            %7 = affine.load %result_0[%arg5, %arg4] : memref<32x64xi32>
            %8 = arith.muli %6, %7 : i32
            %9 = arith.addi %arg6, %8 : i32
            affine.yield %9 : i32
          }
          affine.store %3, %1[0] : memref<2xi32>
          %4 = arith.cmpi sgt, %3, %c0_i32 : i32
          %5 = arith.select %4, %3, %c0_i32 : i32
          affine.store %5, %2[%arg3, %arg4] : memref<16x64xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "ffn_fc1"}
    ADORA.BlockStore %2, %arg2 [0, 0] : memref<16x64xi32> -> memref<?x64xi32>  {Id = "3", KernelName = "ffn_fc1"}
    ADORA.BlockStore %1, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "ffn_fc1"}
    return
  }
  func.func @ffn_fc2(%arg0: memref<?x64xi32>, %arg1: memref<?x32xi32>, %arg2: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<?x64xi32> -> memref<16x64xi32>  {Id = "0", KernelName = "ffn_fc2"}
    %result_0 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x32xi32> -> memref<64x32xi32>  {Id = "1", KernelName = "ffn_fc2"}
    %1 = ADORA.LocalMemAlloc memref<16x32xi32>  {Id = "2", KernelName = "ffn_fc2"}
    %2 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "3", KernelName = "ffn_fc2"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 32 {
          %3 = affine.for %arg5 = 0 to 64 iter_args(%arg6 = %c0_i32) -> (i32) {
            %4 = affine.load %result[%arg3, %arg5] : memref<16x64xi32>
            %5 = affine.load %result_0[%arg5, %arg4] : memref<64x32xi32>
            %6 = arith.muli %4, %5 : i32
            %7 = arith.addi %arg6, %6 : i32
            affine.yield %7 : i32
          }
          affine.store %3, %2[0] : memref<2xi32>
          affine.store %3, %1[%arg3, %arg4] : memref<16x32xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "ffn_fc2"}
    ADORA.BlockStore %2, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "3", KernelName = "ffn_fc2"}
    ADORA.BlockStore %1, %arg2 [0, 0] : memref<16x32xi32> -> memref<?x32xi32>  {Id = "2", KernelName = "ffn_fc2"}
    return
  }
  func.func @ffn(%arg0: memref<?x32xi32>, %arg1: memref<?x64xi32>, %arg2: memref<?x32xi32>, %arg3: memref<?x64xi32>, %arg4: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca() : memref<i32>
    affine.store %0, %alloca_0[] : memref<i32>
    affine.for %arg5 = 0 to 16 {
      %result = ADORA.BlockLoad %arg0 [%arg5, 0] : memref<?x32xi32> -> memref<1x32xi32>  {Id = "0", KernelName = "ffn_0"}
      %result_1 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x64xi32> -> memref<32x64xi32>  {Id = "1", KernelName = "ffn_0"}
      %1 = ADORA.LocalMemAlloc memref<1x64xi32>  {Id = "2", KernelName = "ffn_0"}
      %2 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "3", KernelName = "ffn_0"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 64 {
          %5 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %c0_i32) -> (i32) {
            %8 = affine.load %result[0, %arg7] : memref<1x32xi32>
            %9 = affine.load %result_1[%arg7, %arg6] : memref<32x64xi32>
            %10 = arith.muli %8, %9 : i32
            %11 = arith.addi %arg8, %10 : i32
            affine.yield %11 : i32
          }
          affine.store %5, %2[0] : memref<2xi32>
          %6 = arith.cmpi sgt, %5, %c0_i32 : i32
          %7 = arith.select %6, %5, %c0_i32 : i32
          affine.store %7, %1[0, %arg6] : memref<1x64xi32>
        }
        ADORA.terminator
      } {KernelName = "ffn_0"}
      ADORA.BlockStore %2, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "3", KernelName = "ffn_0"}
      ADORA.BlockStore %1, %arg3 [%arg5, 0] : memref<1x64xi32> -> memref<?x64xi32>  {Id = "2", KernelName = "ffn_0"}
      %result_2 = ADORA.BlockLoad %arg3 [%arg5, 0] : memref<?x64xi32> -> memref<1x64xi32>  {Id = "0", KernelName = "ffn_1"}
      %result_3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x32xi32> -> memref<64x32xi32>  {Id = "1", KernelName = "ffn_1"}
      %3 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "ffn_1"}
      %4 = ADORA.LocalMemAlloc memref<1x32xi32>  {Id = "3", KernelName = "ffn_1"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 32 {
          %5 = affine.for %arg7 = 0 to 64 iter_args(%arg8 = %c0_i32) -> (i32) {
            %6 = affine.load %result_2[0, %arg7] : memref<1x64xi32>
            %7 = affine.load %result_3[%arg7, %arg6] : memref<64x32xi32>
            %8 = arith.muli %6, %7 : i32
            %9 = arith.addi %arg8, %8 : i32
            affine.yield %9 : i32
          }
          affine.store %5, %3[0] : memref<2xi32>
          affine.store %5, %4[0, %arg6] : memref<1x32xi32>
        }
        ADORA.terminator
      } {KernelName = "ffn_1"}
      ADORA.BlockStore %4, %arg4 [%arg5, 0] : memref<1x32xi32> -> memref<?x32xi32>  {Id = "3", KernelName = "ffn_1"}
      ADORA.BlockStore %3, %alloca_0 [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "ffn_1"}
    }
    return
  }
}

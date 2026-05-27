// schedule_complex_sobel.mlir
// Sobel edge detection: diamond dependency  img -> {Gx, Gy} -> G (AND-join).
// Verifies --adora-schedule-tasks:
//   1. marks module adora.scheduled
//   2. dep_summary contains RAW + RAR edges across three kernel blocks
//   3. ADORA.kernel async [...] threads tokens
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
// CHECK: kind = "RAR"
// CHECK: ADORA.BlockLoad async [
// CHECK: ADORA.kernel async [

// REGRESSION: BlockLoad must use implicit async keyword (GPU-dialect style).
// No explicit "-> !ADORA.token" suffix should appear.
// CHECK-NOT: ADORA.BlockLoad{{.*}}-> !ADORA.token

module {
  func.func @sobel(%arg0: memref<?x64xi32>, %arg1: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<64x64xi32>
    %alloca_0 = memref.alloca() : memref<64x64xi32>
    affine.for %arg2 = 0 to 64 {
      affine.for %arg3 = 0 to 64 {
        affine.store %c0_i32, %alloca_0[%arg2, %arg3] : memref<64x64xi32>
        affine.store %c0_i32, %alloca[%arg2, %arg3] : memref<64x64xi32>
      }
    }
    affine.for %arg2 = 0 to 62 {
      %result = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "0", KernelName = "sobel_0"}
      %result_1 = ADORA.BlockLoad %arg0 [%arg2, 2] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "1", KernelName = "sobel_0"}
      %result_2 = ADORA.BlockLoad %arg0 [%arg2 + 1, 0] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "2", KernelName = "sobel_0"}
      %result_3 = ADORA.BlockLoad %arg0 [%arg2 + 1, 2] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "3", KernelName = "sobel_0"}
      %result_4 = ADORA.BlockLoad %arg0 [%arg2 + 2, 0] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "4", KernelName = "sobel_0"}
      %result_5 = ADORA.BlockLoad %arg0 [%arg2 + 2, 2] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "5", KernelName = "sobel_0"}
      %0 = ADORA.LocalMemAlloc memref<1x64xi32>  {Id = "6", KernelName = "sobel_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 62 {
          %1 = affine.load %result[0, %arg3] : memref<1x62xi32>
          %2 = affine.load %result_1[0, %arg3] : memref<1x62xi32>
          %3 = arith.subi %2, %1 : i32
          %4 = affine.load %result_2[1, %arg3] : memref<1x62xi32>
          %5 = arith.muli %4, %c2_i32 : i32
          %6 = arith.subi %3, %5 : i32
          %7 = affine.load %result_3[1, %arg3] : memref<1x62xi32>
          %8 = arith.muli %7, %c2_i32 : i32
          %9 = arith.addi %6, %8 : i32
          %10 = affine.load %result_4[2, %arg3] : memref<1x62xi32>
          %11 = arith.subi %9, %10 : i32
          %12 = affine.load %result_5[2, %arg3] : memref<1x62xi32>
          %13 = arith.addi %11, %12 : i32
          affine.store %13, %0[1, %arg3] : memref<1x64xi32>
        }
        ADORA.terminator
      } {KernelName = "sobel_0"}
      ADORA.BlockStore %0, %alloca_0 [%arg2 + 1, 1] : memref<1x64xi32> -> memref<64x64xi32>  {Id = "6", KernelName = "sobel_0"}
    }
    affine.for %arg2 = 0 to 62 {
      %result = ADORA.BlockLoad %arg0 [%arg2, 0] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "0", KernelName = "sobel_1"}
      %result_1 = ADORA.BlockLoad %arg0 [%arg2, 1] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "1", KernelName = "sobel_1"}
      %result_2 = ADORA.BlockLoad %arg0 [%arg2, 2] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "2", KernelName = "sobel_1"}
      %result_3 = ADORA.BlockLoad %arg0 [%arg2 + 2, 0] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "3", KernelName = "sobel_1"}
      %result_4 = ADORA.BlockLoad %arg0 [%arg2 + 2, 1] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "4", KernelName = "sobel_1"}
      %result_5 = ADORA.BlockLoad %arg0 [%arg2 + 2, 2] : memref<?x64xi32> -> memref<1x62xi32>  {Id = "5", KernelName = "sobel_1"}
      %0 = ADORA.LocalMemAlloc memref<1x64xi32>  {Id = "6", KernelName = "sobel_1"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 62 {
          %1 = affine.load %result[0, %arg3] : memref<1x62xi32>
          %2 = arith.muli %1, %c-1_i32 : i32
          %3 = affine.load %result_1[0, %arg3] : memref<1x62xi32>
          %4 = arith.muli %3, %c2_i32 : i32
          %5 = arith.subi %2, %4 : i32
          %6 = affine.load %result_2[0, %arg3] : memref<1x62xi32>
          %7 = arith.subi %5, %6 : i32
          %8 = affine.load %result_3[2, %arg3] : memref<1x62xi32>
          %9 = arith.addi %7, %8 : i32
          %10 = affine.load %result_4[2, %arg3] : memref<1x62xi32>
          %11 = arith.muli %10, %c2_i32 : i32
          %12 = arith.addi %9, %11 : i32
          %13 = affine.load %result_5[2, %arg3] : memref<1x62xi32>
          %14 = arith.addi %12, %13 : i32
          affine.store %14, %0[1, %arg3] : memref<1x64xi32>
        }
        ADORA.terminator
      } {KernelName = "sobel_1"}
      ADORA.BlockStore %0, %alloca [%arg2 + 1, 1] : memref<1x64xi32> -> memref<64x64xi32>  {Id = "6", KernelName = "sobel_1"}
    }
    affine.for %arg2 = 0 to 62 step 31 {
      %result = ADORA.BlockLoad %alloca_0 [%arg2 + 1, 1] : memref<64x64xi32> -> memref<31x64xi32>  {Id = "0", KernelName = "sobel_2"}
      %result_1 = ADORA.BlockLoad %alloca [%arg2 + 1, 1] : memref<64x64xi32> -> memref<31x64xi32>  {Id = "1", KernelName = "sobel_2"}
      %0 = ADORA.LocalMemAlloc memref<31x62xi32>  {Id = "2", KernelName = "sobel_2"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 31 {
          affine.for %arg4 = 0 to 62 {
            %1 = affine.load %result[%arg3 + 1, %arg4] : memref<31x64xi32>
            %2 = arith.cmpi slt, %1, %c0_i32 : i32
            %3 = scf.if %2 -> (i32) {
              %8 = arith.subi %c0_i32, %1 : i32
              scf.yield %8 : i32
            } else {
              scf.yield %1 : i32
            }
            %4 = affine.load %result_1[%arg3 + 1, %arg4] : memref<31x64xi32>
            %5 = arith.cmpi slt, %4, %c0_i32 : i32
            %6 = scf.if %5 -> (i32) {
              %8 = arith.subi %c0_i32, %4 : i32
              scf.yield %8 : i32
            } else {
              scf.yield %4 : i32
            }
            %7 = arith.addi %3, %6 : i32
            affine.store %7, %0[%arg3 + 1, %arg4] : memref<31x62xi32>
          }
        }
        ADORA.terminator
      } {KernelName = "sobel_2"}
      ADORA.BlockStore %0, %arg1 [%arg2 + 1, 1] : memref<31x62xi32> -> memref<?x64xi32>  {Id = "2", KernelName = "sobel_2"}
    }
    return
  }
}

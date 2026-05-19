// RUN: %cgra-opt \
// RUN:   --adora-extract-affine-for-to-kernel \
// RUN:   --adora-simplify-loadstore \
// RUN:   --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
// RUN:   %s | %FileCheck %s
//
// CHECK: module {
// CHECK:   func.func @gemm_opt
// CHECK:   %cst = arith.constant 1.200000e+00 : f32
// CHECK:   %cst_0 = arith.constant 1.500000e+00 : f32
// CHECK:   %result = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_0"}
// CHECK:   %0 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
// CHECK:   ADORA.kernel {
// CHECK:     affine.for %arg3 = 0 to 20 {
// CHECK:       affine.for %arg4 = 0 to 25 {
// CHECK:         %2 = affine.load %result[%arg3, %arg4] : memref<20x25xf32>
// CHECK:         %3 = arith.mulf %2, %cst : f32
// CHECK:         affine.store %3, %0[%arg3, %arg4] : memref<20x25xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     ADORA.terminator
// CHECK:   } {KernelName = "gemm_opt_0"}
// CHECK:   ADORA.BlockStore %0, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
// CHECK:   %result_1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_1"}
// CHECK:   %result_2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x30xf32> -> memref<20x30xf32>  {Id = "1", KernelName = "gemm_opt_1"}
// CHECK:   %result_3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "gemm_opt_1"}
// CHECK:   %1 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
// CHECK:   ADORA.kernel {
// CHECK:     affine.for %arg3 = 0 to 20 {
// CHECK:       affine.for %arg4 = 0 to 25 {
// CHECK:         %2 = affine.load %result_1[%arg3, %arg4] : memref<20x25xf32>
// CHECK:         %3 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %2) -> (f32) {
// CHECK:           %4 = affine.load %result_2[%arg3, %arg5] : memref<20x30xf32>
// CHECK:           %5 = arith.mulf %4, %cst_0 : f32
// CHECK:           %6 = affine.load %result_3[%arg5, %arg4] : memref<30x25xf32>
// CHECK:           %7 = arith.mulf %5, %6 : f32
// CHECK:           %8 = arith.addf %arg6, %7 : f32
// CHECK:           affine.yield %8 : f32
// CHECK:         }
// CHECK:         affine.store %3, %1[%arg3, %arg4] : memref<20x25xf32>
// CHECK:       }
// CHECK:     }
// CHECK:     ADORA.terminator
// CHECK:   } {KernelName = "gemm_opt_1"}
// CHECK:   ADORA.BlockStore %1, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
// CHECK:   return
// CHECK: }
module attributes {} {
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 {
      affine.for %arg4 = 0 to 25 {
        %0 = affine.load %arg0[%arg3, %arg4] : memref<?x25xf32>
        %1 = arith.mulf %0, %cst : f32
        affine.store %1, %arg0[%arg3, %arg4] : memref<?x25xf32>
      }
    }
    affine.for %arg3 = 0 to 20 {
      affine.for %arg4 = 0 to 25 {
        affine.for %arg5 = 0 to 30 {
          %0 = affine.load %arg1[%arg3, %arg5] : memref<?x30xf32>
          %1 = arith.mulf %0, %cst_0 : f32
          %2 = affine.load %arg2[%arg5, %arg4] : memref<?x25xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = affine.load %arg0[%arg3, %arg4] : memref<?x25xf32>
          %5 = arith.addf %4, %3 : f32
          affine.store %5, %arg0[%arg3, %arg4] : memref<?x25xf32>
        }
      }
    }
    return
  }
}

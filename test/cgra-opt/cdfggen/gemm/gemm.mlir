// RUN: rm -f gemm_opt_0_CDFG.dot gemm_opt_1_CDFG.dot
// RUN: %cgra-opt --adora-kernel-dfg-gen %s | %FileCheck %s
// RUN: test -s gemm_opt_0_CDFG.dot
// RUN: test -s gemm_opt_1_CDFG.dot
// RUN: %FileCheck %s --check-prefix=DOT0 --input-file=gemm_opt_0_CDFG.dot
// RUN: %FileCheck %s --check-prefix=DOT1 --input-file=gemm_opt_1_CDFG.dot
//
// DOT0: Digraph G
// DOT0-DAG: Input{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT0-DAG: CONST{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT0-DAG: FMUL32{{[0-9]+}} -> Output{{[0-9]+}}
// DOT0: }
//
// DOT1: Digraph G {
// DOT1-DAG: Input{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT1-DAG: CONST{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT1-DAG: FMUL32{{[0-9]+}} -> FMUL32{{[0-9]+}}
// DOT1-DAG: FMUL32{{[0-9]+}} -> FACC32{{[0-9]+}}
// DOT1-DAG: Input{{[0-9]+}} -> FADD32{{[0-9]+}}
// DOT1-DAG: FADD32{{[0-9]+}} -> Output{{[0-9]+}}
// DOT1-DAG: FACC32{{[0-9]+}} -> FADD32{{[0-9]+}}
// DOT1: }
//
// CHECK: module {
// CHECK:   func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK:     %cst = arith.constant 1.200000e+00 : f32
// CHECK:     %cst_0 = arith.constant 1.500000e+00 : f32
// CHECK:     %result = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_0"}
// CHECK:     %0 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
// CHECK:     ADORA.kernel {
// CHECK:       affine.for %arg3 = 0 to 20 {
// CHECK:         affine.for %arg4 = 0 to 25 {
// CHECK:           %2 = affine.load %result[%arg3, %arg4] : memref<20x25xf32>
// CHECK:           %3 = arith.mulf %2, %cst : f32
// CHECK:           affine.store %3, %0[%arg3, %arg4] : memref<20x25xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       ADORA.terminator
// CHECK:     } {KernelName = "gemm_opt_0"}
// CHECK:     ADORA.BlockStore %0, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
// CHECK:     %result_1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_1"}
// CHECK:     %result_2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x30xf32> -> memref<20x30xf32>  {Id = "1", KernelName = "gemm_opt_1"}
// CHECK:     %result_3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "gemm_opt_1"}
// CHECK:     %1 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
// CHECK:     ADORA.kernel {
// CHECK:       affine.for %arg3 = 0 to 20 {
// CHECK:         affine.for %arg4 = 0 to 25 {
// CHECK:           %2 = affine.load %result_1[%arg3, %arg4] : memref<20x25xf32>
// CHECK:           %cst_4 = arith.constant 0.000000e+00 : f32
// CHECK:           %3 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %cst_4) -> (f32) {
// CHECK:             %5 = affine.load %result_2[%arg3, %arg5] : memref<20x30xf32>
// CHECK:             %6 = arith.mulf %5, %cst_0 : f32
// CHECK:             %7 = affine.load %result_3[%arg5, %arg4] : memref<30x25xf32>
// CHECK:             %8 = arith.mulf %6, %7 : f32
// CHECK:             %9 = arith.addf %8, %arg6 : f32
// CHECK:             affine.yield %9 : f32
// CHECK:           }
// CHECK:           %4 = arith.addf %3, %2 : f32
// CHECK:           affine.store %4, %1[%arg3, %arg4] : memref<20x25xf32>
// CHECK:         }
// CHECK:       }
// CHECK:       ADORA.terminator
// CHECK:     } {KernelName = "gemm_opt_1"}
// CHECK:     ADORA.BlockStore %1, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
// CHECK:     return
// CHECK:   }
// CHECK: }
// 
module {
  func.func @gemm_opt(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %0 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_0"}
    %1 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 20 {
        affine.for %arg4 = 0 to 25 {
          %6 = affine.load %0[%arg3, %arg4] : memref<20x25xf32>
          %7 = arith.mulf %6, %cst : f32
          affine.store %7, %1[%arg3, %arg4] : memref<20x25xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "gemm_opt_0"}
    ADORA.BlockStore %1, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
    %2 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_1"}
    %3 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x30xf32> -> memref<20x30xf32>  {Id = "1", KernelName = "gemm_opt_1"}
    %4 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "gemm_opt_1"}
    %5 = ADORA.LocalMemAlloc memref<20x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 20 {
        affine.for %arg4 = 0 to 25 {
          %6 = affine.load %2[%arg3, %arg4] : memref<20x25xf32>
          %7 = affine.for %arg5 = 0 to 30 iter_args(%arg6 = %6) -> (f32) {
            %8 = affine.load %3[%arg3, %arg5] : memref<20x30xf32>
            %9 = arith.mulf %8, %cst_0 : f32
            %10 = affine.load %4[%arg5, %arg4] : memref<30x25xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg6, %11 : f32
            affine.yield %12 : f32
          }
          affine.store %7, %5[%arg3, %arg4] : memref<20x25xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "gemm_opt_1"}
    ADORA.BlockStore %5, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
    return
  }
}
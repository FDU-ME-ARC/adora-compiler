module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>, llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "polygeist.target-cpu" = "x86-64", "polygeist.target-features" = "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87", "polygeist.tune-cpu" = "generic"} {
  func.func @gemm(%arg0: memref<?x25xf32>, %arg1: memref<?x30xf32>, %arg2: memref<?x25xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 1.200000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    affine.for %arg3 = 0 to 20 {
      %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
      %1 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %6 = affine.load %0[0, %arg4] : memref<1x25xf32>
          %7 = arith.mulf %6, %cst : f32
          affine.store %7, %1[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_0"}
      ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
      %2 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_1"}
      %3 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "1", KernelName = "kernel_gemm_1"}
      %4 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
      %5 = ADORA.LocalMemAlloc memref<1x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
      ADORA.kernel {
        affine.for %arg4 = 0 to 25 {
          %6 = affine.load %2[0, %arg4] : memref<1x25xf32>
          %7 = affine.for %arg5 = 0 to 30 step 10 iter_args(%arg6 = %6) -> (f32) {
            %8 = affine.load %3[0, %arg5] : memref<1x30xf32>
            %9 = arith.mulf %8, %cst_0 : f32
            %10 = affine.load %4[%arg5, %arg4] : memref<30x25xf32>
            %11 = arith.mulf %9, %10 : f32
            %12 = arith.addf %arg6, %11 : f32
            %13 = affine.load %3[0, %arg5 + 1] : memref<1x30xf32>
            %14 = arith.mulf %13, %cst_0 : f32
            %15 = affine.load %4[%arg5 + 1, %arg4] : memref<30x25xf32>
            %16 = arith.mulf %14, %15 : f32
            %17 = arith.addf %12, %16 : f32
            %18 = affine.load %3[0, %arg5 + 2] : memref<1x30xf32>
            %19 = arith.mulf %18, %cst_0 : f32
            %20 = affine.load %4[%arg5 + 2, %arg4] : memref<30x25xf32>
            %21 = arith.mulf %19, %20 : f32
            %22 = arith.addf %17, %21 : f32
            %23 = affine.load %3[0, %arg5 + 3] : memref<1x30xf32>
            %24 = arith.mulf %23, %cst_0 : f32
            %25 = affine.load %4[%arg5 + 3, %arg4] : memref<30x25xf32>
            %26 = arith.mulf %24, %25 : f32
            %27 = arith.addf %22, %26 : f32
            %28 = affine.load %3[0, %arg5 + 4] : memref<1x30xf32>
            %29 = arith.mulf %28, %cst_0 : f32
            %30 = affine.load %4[%arg5 + 4, %arg4] : memref<30x25xf32>
            %31 = arith.mulf %29, %30 : f32
            %32 = arith.addf %27, %31 : f32
            %33 = affine.load %3[0, %arg5 + 5] : memref<1x30xf32>
            %34 = arith.mulf %33, %cst_0 : f32
            %35 = affine.load %4[%arg5 + 5, %arg4] : memref<30x25xf32>
            %36 = arith.mulf %34, %35 : f32
            %37 = arith.addf %32, %36 : f32
            %38 = affine.load %3[0, %arg5 + 6] : memref<1x30xf32>
            %39 = arith.mulf %38, %cst_0 : f32
            %40 = affine.load %4[%arg5 + 6, %arg4] : memref<30x25xf32>
            %41 = arith.mulf %39, %40 : f32
            %42 = arith.addf %37, %41 : f32
            %43 = affine.load %3[0, %arg5 + 7] : memref<1x30xf32>
            %44 = arith.mulf %43, %cst_0 : f32
            %45 = affine.load %4[%arg5 + 7, %arg4] : memref<30x25xf32>
            %46 = arith.mulf %44, %45 : f32
            %47 = arith.addf %42, %46 : f32
            %48 = affine.load %3[0, %arg5 + 8] : memref<1x30xf32>
            %49 = arith.mulf %48, %cst_0 : f32
            %50 = affine.load %4[%arg5 + 8, %arg4] : memref<30x25xf32>
            %51 = arith.mulf %49, %50 : f32
            %52 = arith.addf %47, %51 : f32
            %53 = affine.load %3[0, %arg5 + 9] : memref<1x30xf32>
            %54 = arith.mulf %53, %cst_0 : f32
            %55 = affine.load %4[%arg5 + 9, %arg4] : memref<30x25xf32>
            %56 = arith.mulf %54, %55 : f32
            %57 = arith.addf %52, %56 : f32
            affine.yield %57 : f32
          }
          affine.store %7, %5[0, %arg4] : memref<1x25xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_gemm_1"}
      ADORA.BlockStore %5, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    }
    return
  }
}

module {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"}
    %1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"}
    %2 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"}
    %3 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %0[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 step 20 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %1[%arg5, %arg6] : memref<40x40xf32>
          %11 = affine.load %2[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          %14 = affine.load %1[%arg5, %arg6 + 1] : memref<40x40xf32>
          %15 = affine.load %2[%arg6 + 1] : memref<40xf32>
          %16 = arith.mulf %14, %15 : f32
          %17 = arith.addf %13, %16 : f32
          %18 = affine.load %1[%arg5, %arg6 + 2] : memref<40x40xf32>
          %19 = affine.load %2[%arg6 + 2] : memref<40xf32>
          %20 = arith.mulf %18, %19 : f32
          %21 = arith.addf %17, %20 : f32
          %22 = affine.load %1[%arg5, %arg6 + 3] : memref<40x40xf32>
          %23 = affine.load %2[%arg6 + 3] : memref<40xf32>
          %24 = arith.mulf %22, %23 : f32
          %25 = arith.addf %21, %24 : f32
          %26 = affine.load %1[%arg5, %arg6 + 4] : memref<40x40xf32>
          %27 = affine.load %2[%arg6 + 4] : memref<40xf32>
          %28 = arith.mulf %26, %27 : f32
          %29 = arith.addf %25, %28 : f32
          %30 = affine.load %1[%arg5, %arg6 + 5] : memref<40x40xf32>
          %31 = affine.load %2[%arg6 + 5] : memref<40xf32>
          %32 = arith.mulf %30, %31 : f32
          %33 = arith.addf %29, %32 : f32
          %34 = affine.load %1[%arg5, %arg6 + 6] : memref<40x40xf32>
          %35 = affine.load %2[%arg6 + 6] : memref<40xf32>
          %36 = arith.mulf %34, %35 : f32
          %37 = arith.addf %33, %36 : f32
          %38 = affine.load %1[%arg5, %arg6 + 7] : memref<40x40xf32>
          %39 = affine.load %2[%arg6 + 7] : memref<40xf32>
          %40 = arith.mulf %38, %39 : f32
          %41 = arith.addf %37, %40 : f32
          %42 = affine.load %1[%arg5, %arg6 + 8] : memref<40x40xf32>
          %43 = affine.load %2[%arg6 + 8] : memref<40xf32>
          %44 = arith.mulf %42, %43 : f32
          %45 = arith.addf %41, %44 : f32
          %46 = affine.load %1[%arg5, %arg6 + 9] : memref<40x40xf32>
          %47 = affine.load %2[%arg6 + 9] : memref<40xf32>
          %48 = arith.mulf %46, %47 : f32
          %49 = arith.addf %45, %48 : f32
          %50 = affine.load %1[%arg5, %arg6 + 10] : memref<40x40xf32>
          %51 = affine.load %2[%arg6 + 10] : memref<40xf32>
          %52 = arith.mulf %50, %51 : f32
          %53 = arith.addf %49, %52 : f32
          %54 = affine.load %1[%arg5, %arg6 + 11] : memref<40x40xf32>
          %55 = affine.load %2[%arg6 + 11] : memref<40xf32>
          %56 = arith.mulf %54, %55 : f32
          %57 = arith.addf %53, %56 : f32
          %58 = affine.load %1[%arg5, %arg6 + 12] : memref<40x40xf32>
          %59 = affine.load %2[%arg6 + 12] : memref<40xf32>
          %60 = arith.mulf %58, %59 : f32
          %61 = arith.addf %57, %60 : f32
          %62 = affine.load %1[%arg5, %arg6 + 13] : memref<40x40xf32>
          %63 = affine.load %2[%arg6 + 13] : memref<40xf32>
          %64 = arith.mulf %62, %63 : f32
          %65 = arith.addf %61, %64 : f32
          %66 = affine.load %1[%arg5, %arg6 + 14] : memref<40x40xf32>
          %67 = affine.load %2[%arg6 + 14] : memref<40xf32>
          %68 = arith.mulf %66, %67 : f32
          %69 = arith.addf %65, %68 : f32
          %70 = affine.load %1[%arg5, %arg6 + 15] : memref<40x40xf32>
          %71 = affine.load %2[%arg6 + 15] : memref<40xf32>
          %72 = arith.mulf %70, %71 : f32
          %73 = arith.addf %69, %72 : f32
          %74 = affine.load %1[%arg5, %arg6 + 16] : memref<40x40xf32>
          %75 = affine.load %2[%arg6 + 16] : memref<40xf32>
          %76 = arith.mulf %74, %75 : f32
          %77 = arith.addf %73, %76 : f32
          %78 = affine.load %1[%arg5, %arg6 + 17] : memref<40x40xf32>
          %79 = affine.load %2[%arg6 + 17] : memref<40xf32>
          %80 = arith.mulf %78, %79 : f32
          %81 = arith.addf %77, %80 : f32
          %82 = affine.load %1[%arg5, %arg6 + 18] : memref<40x40xf32>
          %83 = affine.load %2[%arg6 + 18] : memref<40xf32>
          %84 = arith.mulf %82, %83 : f32
          %85 = arith.addf %81, %84 : f32
          %86 = affine.load %1[%arg5, %arg6 + 19] : memref<40x40xf32>
          %87 = affine.load %2[%arg6 + 19] : memref<40xf32>
          %88 = arith.mulf %86, %87 : f32
          %89 = arith.addf %85, %88 : f32
          affine.yield %89 : f32
        }
        affine.store %9, %3[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_0"}
    ADORA.BlockStore %3, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
    %4 = ADORA.BlockLoad %arg1 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_1"}
    %5 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_1"}
    %6 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_1"}
    %7 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    ADORA.kernel {
      affine.for %arg5 = 0 to 40 {
        %8 = affine.load %4[%arg5] : memref<40xf32>
        %9 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %8) -> (f32) {
          %10 = affine.load %5[%arg6, %arg5] : memref<40x40xf32>
          %11 = affine.load %6[%arg6] : memref<40xf32>
          %12 = arith.mulf %10, %11 : f32
          %13 = arith.addf %arg7, %12 : f32
          affine.yield %13 : f32
        }
        affine.store %9, %7[%arg5] : memref<40xf32>
      }
      ADORA.terminator
    } {KernelName = "kernel_mvt_1"}
    ADORA.BlockStore %7, %arg1 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_1"}
    return
  }
}

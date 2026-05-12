module {
  module {
    func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
      %result = ADORA.BlockLoad %arg0 [0] : memref<?xf32> -> memref<40xf32>  {Id = "0", KernelName = "kernel_mvt_0"}
      %result_0 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x40xf32> -> memref<40x40xf32>  {Id = "1", KernelName = "kernel_mvt_0"}
      %result_1 = ADORA.BlockLoad %arg2 [0] : memref<?xf32> -> memref<40xf32>  {Id = "2", KernelName = "kernel_mvt_0"}
      %0 = ADORA.LocalMemAlloc memref<40xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
      ADORA.kernel {
        affine.for %arg5 = 0 to 40 {
          %1 = affine.load %result[%arg5] : memref<40xf32>
          %2 = affine.for %arg6 = 0 to 40 iter_args(%arg7 = %1) -> (f32) {
            %3 = affine.load %result_0[%arg5, %arg6] : memref<40x40xf32>
            %4 = affine.load %result_1[%arg6] : memref<40xf32>
            %5 = arith.mulf %3, %4 : f32
            %6 = arith.addf %arg7, %5 : f32
            affine.yield %6 : f32
          }
          affine.store %2, %0[%arg5] : memref<40xf32>
        }
        ADORA.terminator
      } {KernelName = "kernel_mvt_0"}
      ADORA.BlockStore %0, %arg0 [0] : memref<40xf32> -> memref<?xf32>  {Id = "3", KernelName = "kernel_mvt_0"}
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          %1 = affine.load %arg1[%arg5] : memref<?xf32>
          %2 = affine.load %arg4[%arg6, %arg5] : memref<?x40xf32>
          %3 = affine.load %arg3[%arg6] : memref<?xf32>
          %4 = arith.mulf %2, %3 : f32
          %5 = arith.addf %1, %4 : f32
          affine.store %5, %arg1[%arg5] : memref<?xf32>
        }
      }
      return
    }
  }
  module {
    func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %arg2: memref<?xf32>, %arg3: memref<?xf32>, %arg4: memref<?x40xf32>) attributes {llvm.linkage = #llvm.linkage<external>} {
      ADORA.kernel {
        affine.for %arg5 = 0 to 40 {
          affine.for %arg6 = 0 to 40 {
            %0 = affine.load %arg0[%arg5] : memref<?xf32>
            %1 = affine.load %arg4[%arg5, %arg6] : memref<?x40xf32>
            %2 = affine.load %arg2[%arg6] : memref<?xf32>
            %3 = arith.mulf %1, %2 : f32
            %4 = arith.addf %0, %3 : f32
            affine.store %4, %arg0[%arg5] : memref<?xf32>
          }
        }
        ADORA.terminator
      } {KernelName = "kernel_mvt_0"}
      affine.for %arg5 = 0 to 40 {
        affine.for %arg6 = 0 to 40 {
          %0 = affine.load %arg1[%arg5] : memref<?xf32>
          %1 = affine.load %arg4[%arg6, %arg5] : memref<?x40xf32>
          %2 = affine.load %arg3[%arg6] : memref<?xf32>
          %3 = arith.mulf %1, %2 : f32
          %4 = arith.addf %0, %3 : f32
          affine.store %4, %arg1[%arg5] : memref<?xf32>
        }
      }
      return
    }
  }
}


module {
  func.func @jacobi_2d(%arg0: memref<?x30xi32>, %arg1: memref<?x30xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    affine.for %arg2 = 0 to 10 {
      %0 = ADORA.BlockLoad %arg0 [1, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "0", KernelName = "jacobi_2d_0"}
      %1 = ADORA.BlockLoad %arg0 [1, 0] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "1", KernelName = "jacobi_2d_0"}
      %2 = ADORA.BlockLoad %arg0 [1, 2] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "2", KernelName = "jacobi_2d_0"}
      %3 = ADORA.BlockLoad %arg0 [2, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "3", KernelName = "jacobi_2d_0"}
      %4 = ADORA.BlockLoad %arg0 [0, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "4", KernelName = "jacobi_2d_0"}
      %5 = ADORA.LocalMemAlloc memref<28x28xi32>  {Id = "5", KernelName = "jacobi_2d_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %12 = affine.load %0[%arg3, %arg4] : memref<28x28xi32>
            %13 = affine.load %1[%arg3, %arg4] : memref<28x28xi32>
            %14 = arith.addi %12, %13 : i32
            %15 = affine.load %2[%arg3, %arg4] : memref<28x28xi32>
            %16 = arith.addi %14, %15 : i32
            %17 = affine.load %3[%arg3, %arg4] : memref<28x28xi32>
            %18 = arith.addi %16, %17 : i32
            %19 = affine.load %4[%arg3, %arg4] : memref<28x28xi32>
            %20 = arith.addi %18, %19 : i32
            %21 = arith.divsi %20, %c5_i32 : i32
            affine.store %21, %5[%arg3, %arg4] : memref<28x28xi32>
          }
        }
        ADORA.terminator
      } {KernelName = "jacobi_2d_0"}
      ADORA.BlockStore %5, %arg1 [1, 1] : memref<28x28xi32> -> memref<?x30xi32>  {Id = "5", KernelName = "jacobi_2d_0"}
      %6 = ADORA.BlockLoad %arg1 [1, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "0", KernelName = "jacobi_2d_1"}
      %7 = ADORA.BlockLoad %arg1 [1, 0] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "1", KernelName = "jacobi_2d_1"}
      %8 = ADORA.BlockLoad %arg1 [1, 2] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "2", KernelName = "jacobi_2d_1"}
      %9 = ADORA.BlockLoad %arg1 [2, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "3", KernelName = "jacobi_2d_1"}
      %10 = ADORA.BlockLoad %arg1 [0, 1] : memref<?x30xi32> -> memref<28x28xi32>  {Id = "4", KernelName = "jacobi_2d_1"}
      %11 = ADORA.LocalMemAlloc memref<28x28xi32>  {Id = "5", KernelName = "jacobi_2d_1"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %12 = affine.load %6[%arg3, %arg4] : memref<28x28xi32>
            %13 = affine.load %7[%arg3, %arg4] : memref<28x28xi32>
            %14 = arith.addi %12, %13 : i32
            %15 = affine.load %8[%arg3, %arg4] : memref<28x28xi32>
            %16 = arith.addi %14, %15 : i32
            %17 = affine.load %9[%arg3, %arg4] : memref<28x28xi32>
            %18 = arith.addi %16, %17 : i32
            %19 = affine.load %10[%arg3, %arg4] : memref<28x28xi32>
            %20 = arith.addi %18, %19 : i32
            %21 = arith.divsi %20, %c5_i32 : i32
            affine.store %21, %11[%arg3, %arg4] : memref<28x28xi32>
          }
        }
        ADORA.terminator
      } {KernelName = "jacobi_2d_1"}
      ADORA.BlockStore %11, %arg0 [1, 1] : memref<28x28xi32> -> memref<?x30xi32>  {Id = "5", KernelName = "jacobi_2d_1"}
    }
    return
  }
}


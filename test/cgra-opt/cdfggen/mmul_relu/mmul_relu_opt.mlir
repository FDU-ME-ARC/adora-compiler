module {
  func.func @mmul_relu(%arg0: memref<?x25xi32>, %arg1: memref<?x25xi32>, %arg2: memref<?x25xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %1 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xi32> -> memref<25x25xi32>  {Id = "0", KernelName = "mmul_relu"}
    %2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x25xi32> -> memref<25x25xi32>  {Id = "1", KernelName = "mmul_relu"}
    %3 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "mmul_relu"}
    %4 = ADORA.LocalMemAlloc memref<25x25xi32>  {Id = "3", KernelName = "mmul_relu"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 25 {
        affine.for %arg4 = 0 to 25 {
          %5 = affine.for %arg5 = 0 to 25 iter_args(%arg6 = %c0_i32) -> (i32) {
            %8 = affine.load %1[%arg3, %arg5] : memref<25x25xi32>
            %9 = affine.load %2[%arg5, %arg4] : memref<25x25xi32>
            %10 = arith.muli %8, %9 : i32
            %11 = arith.addi %arg6, %10 : i32
            affine.yield %11 : i32
          }
          affine.store %5, %3[0] : memref<2xi32>
          %6 = arith.cmpi slt, %5, %c0_i32 : i32
          %7 = arith.select %6, %c0_i32, %5 : i32
          // scf.if %6 {
          //   affine.store %c0_i32, %3[0] : memref<2xi32>
          // }
          affine.store %7, %4[%arg3, %arg4] : memref<25x25xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "mmul_relu"}
    ADORA.BlockStore %4, %arg2 [0, 0] : memref<25x25xi32> -> memref<?x25xi32>  {Id = "3", KernelName = "mmul_relu"}
    ADORA.BlockStore %3, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "mmul_relu"}
    return
  }
}


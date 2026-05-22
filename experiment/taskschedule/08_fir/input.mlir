module {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> memref<i32> {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %0 = ADORA.BlockLoad %arg1 [0] : memref<100xi32> -> memref<100xi32>  {Id = "0", KernelName = "kernel_fir"}
    %1 = ADORA.BlockLoad %arg0 [0] : memref<100xi32> -> memref<100xi32>  {Id = "1", KernelName = "kernel_fir"}
    %2 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "kernel_fir"}
    ADORA.kernel {
      %3 = affine.for %arg2 = 0 to 100 iter_args(%arg3 = %c0_i32) -> (i32) {
        %4 = affine.load %0[%arg2] : memref<100xi32>
        %5 = affine.load %1[-%arg2 + 99] : memref<100xi32>
        %6 = arith.muli %4, %5 : i32
        %7 = arith.addi %arg3, %6 : i32
        affine.yield %7 : i32
      }
      affine.store %3, %2[0] : memref<2xi32>
      ADORA.terminator
    } {KernelName = "kernel_fir"}
    ADORA.BlockStore %2, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "kernel_fir"}
    return %alloca : memref<i32>
  }
}


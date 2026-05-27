module {
  func.func @viterbi(%arg0: memref<?x8xi32>, %arg1: memref<?xi32>, %arg2: memref<?x8xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-2147483648_i32 = arith.constant -2147483648 : i32
    %alloca = memref.alloca() : memref<64x8xi32>
    affine.for %arg3 = 0 to 8 {
      %2 = affine.load %arg0[0, %arg3] : memref<?x8xi32>
      affine.store %2, %alloca[0, %arg3] : memref<64x8xi32>
    }
    %result = ADORA.BlockLoad %alloca [62, 0] : memref<64x8xi32> -> memref<1x8xi32>  {Id = "0", KernelName = "viterbi"}
    %result_0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x8xi32> -> memref<8x8xi32>  {Id = "1", KernelName = "viterbi"}
    %result_1 = ADORA.BlockLoad %arg0 [63, 0] : memref<?x8xi32> -> memref<1x8xi32>  {Id = "2", KernelName = "viterbi"}
    %0 = ADORA.LocalMemAlloc memref<1x8xi32>  {Id = "3", KernelName = "viterbi"}
    %1 = ADORA.LocalMemAlloc memref<8xi32>  {Id = "4", KernelName = "viterbi"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 8 {
        %2 = affine.for %arg4 = 0 to 8 iter_args(%arg5 = %c-2147483648_i32) -> (i32) {
          %5 = affine.load %result[0, %arg4] : memref<1x8xi32>
          %6 = affine.load %result_0[%arg4, %arg3] : memref<8x8xi32>
          %7 = arith.addi %5, %6 : i32
          %8 = arith.cmpi sgt, %7, %arg5 : i32
          %9 = arith.select %8, %7, %arg5 : i32
          affine.yield %9 : i32
        }
        %3 = affine.load %result_1[0, %arg3] : memref<1x8xi32>
        %4 = arith.addi %2, %3 : i32
        affine.store %4, %0[0, %arg3] : memref<1x8xi32>
        affine.store %4, %1[%arg3] : memref<8xi32>
      }
      ADORA.terminator
    } {KernelName = "viterbi"}
    ADORA.BlockStore %1, %arg1 [0] : memref<8xi32> -> memref<?xi32>  {Id = "4", KernelName = "viterbi"}
    ADORA.BlockStore %0, %alloca [63, 0] : memref<1x8xi32> -> memref<64x8xi32>  {Id = "3", KernelName = "viterbi"}
    return
  }
}


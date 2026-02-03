module {
  func.func @merge(%arg0: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c1024_i32 = arith.constant 1024 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = llvm.mlir.undef : i32
    %alloca = memref.alloca() : memref<i32>
    affine.store %0, %alloca[] : memref<i32>
    %alloca_0 = memref.alloca() : memref<i32>
    affine.store %0, %alloca_0[] : memref<i32>
    %alloca_1 = memref.alloca() : memref<2048xi32>
    affine.for %arg1 = 0 to 1025 {
      %6 = arith.index_cast %arg1 : index to i32
      affine.store %6, %alloca_0[] : memref<i32>
      %7 = affine.load %arg0[%arg1] : memref<?xi32>
      affine.store %7, %alloca_1[%arg1] : memref<2048xi32>
    }
    affine.for %arg1 = 1025 to 1025 {
      %6 = arith.index_cast %arg1 : index to i32
      affine.store %6, %alloca[] : memref<i32>
      %7 = affine.load %arg0[%arg1] : memref<?xi32>
      affine.store %7, %alloca_1[-%arg1 + 2049] : memref<2048xi32>
    }
    affine.store %c0_i32, %alloca_0[] : memref<i32>
    affine.store %c1024_i32, %alloca[] : memref<i32>
    %1 = ADORA.BlockLoad %alloca [] : memref<i32> -> memref<2xi32>  {Id = "0", KernelName = ""}
    // %2 = ADORA.BlockLoad %alloca_0 [] : memref<i32> -> memref<2xi32>  {Id = "1", KernelName = ""}
    %3 = ADORA.BlockLoad %alloca_1 [0] : memref<2048xi32> -> memref<2048xi32>  {Id = "2", KernelName = ""}
    %5 = ADORA.LocalMemAlloc memref<1025xi32>  {Id = "4", KernelName = ""}
    ADORA.kernel {
      affine.for %arg1 = 0 to 1025 {
        %6 = affine.load %1[0] : memref<2xi32>
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %3[%7] : memref<2048xi32>
        affine.store %8, %5[%arg1] : memref<1025xi32>
        // %10 = arith.index_cast %9 : i32 to index
        // %11 = memref.load %alloca_1[%10] : memref<2048xi32>
      }
      ADORA.terminator
    }{KernelName = ""}
    ADORA.BlockStore %5, %arg0 [0] : memref<1025xi32> -> memref<?xi32>  {Id = "4", KernelName = ""}
    return
  }
}


module {
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    affine.for %arg2 = 0 to 921600 step 2048 {
      %0 = ADORA.BlockLoad %arg0 [%arg2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray"}
      %1 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 2048 {
          %2 = affine.load %0[%arg3] : memref<2048xi32>
          %3 = arith.shrsi %2, %c24_i32 : i32
          %4 = arith.andi %3, %c255_i32 : i32
          %5 = arith.shli %4, %c24_i32 : i32
          %6 = arith.andi %2, %c255_i32 : i32
          %7 = arith.muli %6, %c77_i32 : i32
          %8 = arith.shrsi %2, %c8_i32 : i32
          %9 = arith.andi %8, %c255_i32 : i32
          %10 = arith.muli %9, %c150_i32 : i32
          %11 = arith.addi %7, %10 : i32
          %12 = arith.shrsi %2, %c16_i32 : i32
          %13 = arith.andi %12, %c255_i32 : i32
          %14 = arith.muli %13, %c29_i32 : i32
          %15 = arith.addi %11, %14 : i32
          %16 = arith.shrsi %15, %c8_i32 : i32
          %17 = arith.shli %16, %c16_i32 : i32
          %18 = arith.ori %5, %17 : i32
          %19 = arith.shli %16, %c8_i32 : i32
          %20 = arith.ori %18, %19 : i32
          %21 = arith.ori %20, %16 : i32
          affine.store %21, %1[%arg3] : memref<2048xi32>
        }
        ADORA.terminator
      } {KernelName = "gray"}
      ADORA.BlockStore %1, %arg1 [%arg2] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray"}
    }
    return
  }
}


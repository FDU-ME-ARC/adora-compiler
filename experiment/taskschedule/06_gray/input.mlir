#map = affine_map<(d0) -> (d0 + 2048)>
#map1 = affine_map<(d0) -> (d0 + 4096)>
module {
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    affine.for %arg2 = 0 to 921600 step 6144 {
      %0 = ADORA.BlockLoad %arg0 [%arg2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray"}
      %1 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      %2 = affine.apply #map(%arg2)
      %3 = ADORA.BlockLoad %arg0 [%2] : memref<?xi32> -> memref<2048xi32>  {Id = "2", KernelName = "gray"}
      %4 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "3", KernelName = "gray"}
      %5 = affine.apply #map1(%arg2)
      %6 = ADORA.BlockLoad %arg0 [%5] : memref<?xi32> -> memref<2048xi32>  {Id = "4", KernelName = "gray"}
      %7 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "5", KernelName = "gray"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 2048 {
          %10 = affine.load %0[%arg3] : memref<2048xi32>
          %11 = arith.shrsi %10, %c24_i32 : i32
          %12 = arith.andi %11, %c255_i32 : i32
          %13 = arith.shli %12, %c24_i32 : i32
          %14 = arith.andi %10, %c255_i32 : i32
          %15 = arith.muli %14, %c77_i32 : i32
          %16 = arith.shrsi %10, %c8_i32 : i32
          %17 = arith.andi %16, %c255_i32 : i32
          %18 = arith.muli %17, %c150_i32 : i32
          %19 = arith.addi %15, %18 : i32
          %20 = arith.shrsi %10, %c16_i32 : i32
          %21 = arith.andi %20, %c255_i32 : i32
          %22 = arith.muli %21, %c29_i32 : i32
          %23 = arith.addi %19, %22 : i32
          %24 = arith.shrsi %23, %c8_i32 : i32
          %25 = arith.shli %24, %c16_i32 : i32
          %26 = arith.ori %13, %25 : i32
          %27 = arith.shli %24, %c8_i32 : i32
          %28 = arith.ori %26, %27 : i32
          %29 = arith.ori %28, %24 : i32
          affine.store %29, %1[%arg3] : memref<2048xi32>
          %30 = affine.load %3[%arg3] : memref<2048xi32>
          %31 = arith.shrsi %30, %c24_i32 : i32
          %32 = arith.andi %31, %c255_i32 : i32
          %33 = arith.shli %32, %c24_i32 : i32
          %34 = arith.andi %30, %c255_i32 : i32
          %35 = arith.muli %34, %c77_i32 : i32
          %36 = arith.shrsi %30, %c8_i32 : i32
          %37 = arith.andi %36, %c255_i32 : i32
          %38 = arith.muli %37, %c150_i32 : i32
          %39 = arith.addi %35, %38 : i32
          %40 = arith.shrsi %30, %c16_i32 : i32
          %41 = arith.andi %40, %c255_i32 : i32
          %42 = arith.muli %41, %c29_i32 : i32
          %43 = arith.addi %39, %42 : i32
          %44 = arith.shrsi %43, %c8_i32 : i32
          %45 = arith.shli %44, %c16_i32 : i32
          %46 = arith.ori %33, %45 : i32
          %47 = arith.shli %44, %c8_i32 : i32
          %48 = arith.ori %46, %47 : i32
          %49 = arith.ori %48, %44 : i32
          affine.store %49, %4[%arg3] : memref<2048xi32>
          %50 = affine.load %6[%arg3] : memref<2048xi32>
          %51 = arith.shrsi %50, %c24_i32 : i32
          %52 = arith.andi %51, %c255_i32 : i32
          %53 = arith.shli %52, %c24_i32 : i32
          %54 = arith.andi %50, %c255_i32 : i32
          %55 = arith.muli %54, %c77_i32 : i32
          %56 = arith.shrsi %50, %c8_i32 : i32
          %57 = arith.andi %56, %c255_i32 : i32
          %58 = arith.muli %57, %c150_i32 : i32
          %59 = arith.addi %55, %58 : i32
          %60 = arith.shrsi %50, %c16_i32 : i32
          %61 = arith.andi %60, %c255_i32 : i32
          %62 = arith.muli %61, %c29_i32 : i32
          %63 = arith.addi %59, %62 : i32
          %64 = arith.shrsi %63, %c8_i32 : i32
          %65 = arith.shli %64, %c16_i32 : i32
          %66 = arith.ori %53, %65 : i32
          %67 = arith.shli %64, %c8_i32 : i32
          %68 = arith.ori %66, %67 : i32
          %69 = arith.ori %68, %64 : i32
          affine.store %69, %7[%arg3] : memref<2048xi32>
        }
        ADORA.terminator
      } {KernelName = "gray"}
      ADORA.BlockStore %1, %arg1 [%arg2] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray"}
      %8 = affine.apply #map(%arg2)
      ADORA.BlockStore %4, %arg1 [%8] : memref<2048xi32> -> memref<?xi32>  {Id = "3", KernelName = "gray"}
      %9 = affine.apply #map1(%arg2)
      ADORA.BlockStore %7, %arg1 [%9] : memref<2048xi32> -> memref<?xi32>  {Id = "5", KernelName = "gray"}
    }
    return
  }
}


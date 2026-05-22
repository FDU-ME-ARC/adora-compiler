module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = false, kind = "LC-RAR", step = 2048 : i64}, {exact = false, kind = "LC-WAW", step = 2048 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    affine.for %arg2 = 0 to 921600 step 2048 {
      %1 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result = ADORA.BlockLoad %arg0 [%arg2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray", stream = 0 : i32}
      %2 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%1, %2) : (!llvm.ptr, i64) -> ()
      %3 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      %4 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %5 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%1, %5) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%1) : (!llvm.ptr) -> ()
      ADORA.kernel {
        affine.for %arg3 = 0 to 2048 {
          %10 = affine.load %result[%arg3] : memref<2048xi32>
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
          affine.store %29, %3[%arg3] : memref<2048xi32>
        }
        ADORA.terminator
      } {KernelName = "gray", stream = 0 : i32}
      %6 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%4, %6) : (!llvm.ptr, i64) -> ()
      %7 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %8 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%4, %8) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%4) : (!llvm.ptr) -> ()
      ADORA.BlockStore %3, %arg1 [%arg2] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray", stream = 0 : i32}
      %9 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%7, %9) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%7) : (!llvm.ptr) -> ()
    }
    return
  }
}


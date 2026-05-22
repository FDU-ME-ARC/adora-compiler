#map = affine_map<(d0) -> (d0 + 2048)>
#map1 = affine_map<(d0) -> (d0 + 4096)>
module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @gray(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 6 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "WAW", overlap = false, src = 7 : i64}, {dst = 8 : i64, kind = "WAW", overlap = false, src = 7 : i64}, {dst = 9 : i64, kind = "WAW", overlap = false, src = 8 : i64}, {dst = 4 : i64, kind = "RAR", overlap = false, src = 2 : i64}, {dst = 4 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 2 : i64, kind = "RAR", overlap = false, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = false, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = false, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = false, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = true, kind = "LC-RAR", step = 6144 : i64}, {exact = false, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}, {exact = false, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}, {exact = false, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}, {exact = true, kind = "LC-WAW", step = 6144 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], llvm.linkage = #llvm.linkage<external>} {
    %c29_i32 = arith.constant 29 : i32
    %c150_i32 = arith.constant 150 : i32
    %c77_i32 = arith.constant 77 : i32
    %c24_i32 = arith.constant 24 : i32
    %c16_i32 = arith.constant 16 : i32
    %c8_i32 = arith.constant 8 : i32
    %c255_i32 = arith.constant 255 : i32
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %1 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %3 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %4 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %5 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %6 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %7 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %8 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    affine.for %arg2 = 0 to 921600 step 6144 {
      %9 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result = ADORA.BlockLoad %arg0 [%arg2] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "gray", stream = 0 : i32}
      %10 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%9, %10) : (!llvm.ptr, i64) -> ()
      %11 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "1", KernelName = "gray"}
      %12 = affine.apply #map(%arg2)
      %13 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %14 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%9, %14) : (!llvm.ptr, i64) -> ()
      %result_0 = ADORA.BlockLoad %arg0 [%12] : memref<?xi32> -> memref<2048xi32>  {Id = "2", KernelName = "gray", stream = 0 : i32}
      %15 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%13, %15) : (!llvm.ptr, i64) -> ()
      %16 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "3", KernelName = "gray"}
      %17 = affine.apply #map1(%arg2)
      %18 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %19 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%13, %19) : (!llvm.ptr, i64) -> ()
      %20 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%9, %20) : (!llvm.ptr, i64) -> ()
      %result_1 = ADORA.BlockLoad %arg0 [%17] : memref<?xi32> -> memref<2048xi32>  {Id = "4", KernelName = "gray", stream = 0 : i32}
      %21 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%18, %21) : (!llvm.ptr, i64) -> ()
      %22 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "5", KernelName = "gray"}
      %23 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %24 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%18, %24) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%18) : (!llvm.ptr) -> ()
      %25 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%13, %25) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%13) : (!llvm.ptr) -> ()
      %26 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%9, %26) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%9) : (!llvm.ptr) -> ()
      ADORA.kernel {
        affine.for %arg3 = 0 to 2048 {
          %42 = affine.load %result[%arg3] : memref<2048xi32>
          %43 = arith.shrsi %42, %c24_i32 : i32
          %44 = arith.andi %43, %c255_i32 : i32
          %45 = arith.shli %44, %c24_i32 : i32
          %46 = arith.andi %42, %c255_i32 : i32
          %47 = arith.muli %46, %c77_i32 : i32
          %48 = arith.shrsi %42, %c8_i32 : i32
          %49 = arith.andi %48, %c255_i32 : i32
          %50 = arith.muli %49, %c150_i32 : i32
          %51 = arith.addi %47, %50 : i32
          %52 = arith.shrsi %42, %c16_i32 : i32
          %53 = arith.andi %52, %c255_i32 : i32
          %54 = arith.muli %53, %c29_i32 : i32
          %55 = arith.addi %51, %54 : i32
          %56 = arith.shrsi %55, %c8_i32 : i32
          %57 = arith.shli %56, %c16_i32 : i32
          %58 = arith.ori %45, %57 : i32
          %59 = arith.shli %56, %c8_i32 : i32
          %60 = arith.ori %58, %59 : i32
          %61 = arith.ori %60, %56 : i32
          affine.store %61, %11[%arg3] : memref<2048xi32>
          %62 = affine.load %result_0[%arg3] : memref<2048xi32>
          %63 = arith.shrsi %62, %c24_i32 : i32
          %64 = arith.andi %63, %c255_i32 : i32
          %65 = arith.shli %64, %c24_i32 : i32
          %66 = arith.andi %62, %c255_i32 : i32
          %67 = arith.muli %66, %c77_i32 : i32
          %68 = arith.shrsi %62, %c8_i32 : i32
          %69 = arith.andi %68, %c255_i32 : i32
          %70 = arith.muli %69, %c150_i32 : i32
          %71 = arith.addi %67, %70 : i32
          %72 = arith.shrsi %62, %c16_i32 : i32
          %73 = arith.andi %72, %c255_i32 : i32
          %74 = arith.muli %73, %c29_i32 : i32
          %75 = arith.addi %71, %74 : i32
          %76 = arith.shrsi %75, %c8_i32 : i32
          %77 = arith.shli %76, %c16_i32 : i32
          %78 = arith.ori %65, %77 : i32
          %79 = arith.shli %76, %c8_i32 : i32
          %80 = arith.ori %78, %79 : i32
          %81 = arith.ori %80, %76 : i32
          affine.store %81, %16[%arg3] : memref<2048xi32>
          %82 = affine.load %result_1[%arg3] : memref<2048xi32>
          %83 = arith.shrsi %82, %c24_i32 : i32
          %84 = arith.andi %83, %c255_i32 : i32
          %85 = arith.shli %84, %c24_i32 : i32
          %86 = arith.andi %82, %c255_i32 : i32
          %87 = arith.muli %86, %c77_i32 : i32
          %88 = arith.shrsi %82, %c8_i32 : i32
          %89 = arith.andi %88, %c255_i32 : i32
          %90 = arith.muli %89, %c150_i32 : i32
          %91 = arith.addi %87, %90 : i32
          %92 = arith.shrsi %82, %c16_i32 : i32
          %93 = arith.andi %92, %c255_i32 : i32
          %94 = arith.muli %93, %c29_i32 : i32
          %95 = arith.addi %91, %94 : i32
          %96 = arith.shrsi %95, %c8_i32 : i32
          %97 = arith.shli %96, %c16_i32 : i32
          %98 = arith.ori %85, %97 : i32
          %99 = arith.shli %96, %c8_i32 : i32
          %100 = arith.ori %98, %99 : i32
          %101 = arith.ori %100, %96 : i32
          affine.store %101, %22[%arg3] : memref<2048xi32>
        }
        ADORA.terminator
      } {KernelName = "gray", stream = 0 : i32}
      %27 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%23, %27) : (!llvm.ptr, i64) -> ()
      %28 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %29 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%23, %29) : (!llvm.ptr, i64) -> ()
      ADORA.BlockStore %11, %arg1 [%arg2] : memref<2048xi32> -> memref<?xi32>  {Id = "1", KernelName = "gray", stream = 0 : i32}
      %30 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%28, %30) : (!llvm.ptr, i64) -> ()
      %31 = affine.apply #map(%arg2)
      %32 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %33 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%23, %33) : (!llvm.ptr, i64) -> ()
      %34 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%28, %34) : (!llvm.ptr, i64) -> ()
      ADORA.BlockStore %16, %arg1 [%31] : memref<2048xi32> -> memref<?xi32>  {Id = "3", KernelName = "gray", stream = 0 : i32}
      %35 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%32, %35) : (!llvm.ptr, i64) -> ()
      %36 = affine.apply #map1(%arg2)
      %37 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %38 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%23, %38) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%23) : (!llvm.ptr) -> ()
      %39 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%28, %39) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%28) : (!llvm.ptr) -> ()
      %40 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%32, %40) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%32) : (!llvm.ptr) -> ()
      ADORA.BlockStore %22, %arg1 [%36] : memref<2048xi32> -> memref<?xi32>  {Id = "5", KernelName = "gray", stream = 0 : i32}
      %41 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%37, %41) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%37) : (!llvm.ptr) -> ()
    }
    return
  }
}


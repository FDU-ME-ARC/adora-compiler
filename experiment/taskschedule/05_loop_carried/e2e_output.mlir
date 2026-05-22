module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @loop_carried_min(%arg0: memref<16xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}]} {
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %1 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    affine.for %arg1 = 0 to 4 {
      %3 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result = ADORA.BlockLoad %arg0 [0] : memref<16xf32> -> memref<16xf32>  {Id = "0", KernelName = "loop_carried_min_kernel", stream = 0 : i32}
      %4 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%3, %4) : (!llvm.ptr, i64) -> ()
      %5 = ADORA.LocalMemAlloc memref<16xf32>  {Id = "1", KernelName = "loop_carried_min_kernel"}
      %6 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %7 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%3, %7) : (!llvm.ptr, i64) -> ()
      ADORA.kernel {
        affine.for %arg2 = 0 to 16 {
          %13 = affine.load %result[%arg2] : memref<16xf32>
          affine.store %13, %5[%arg2] : memref<16xf32>
        }
        ADORA.terminator
      } {KernelName = "loop_carried_min_kernel", stream = 0 : i32}
      %8 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%6, %8) : (!llvm.ptr, i64) -> ()
      %9 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %10 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%6, %10) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%6) : (!llvm.ptr) -> ()
      %11 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%3, %11) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%3) : (!llvm.ptr) -> ()
      ADORA.BlockStore %5, %arg0 [0] : memref<16xf32> -> memref<16xf32>  {Id = "2", KernelName = "loop_carried_min_kernel", stream = 0 : i32}
      %12 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%9, %12) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%9) : (!llvm.ptr) -> ()
    }
    return
  }
}


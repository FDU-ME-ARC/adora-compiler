module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @fanin(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>, %arg2: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}]}]} {
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "fanin_k", stream = 0 : i32}
    %1 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%0, %1) : (!llvm.ptr, i64) -> ()
    %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_0 = ADORA.BlockLoad %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "fanin_k", stream = 1 : i32}
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @adoraEventRecord(%2, %3) : (!llvm.ptr, i64) -> ()
    %4 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "2", KernelName = "fanin_k"}
    %5 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %6 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%2, %6) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%2) : (!llvm.ptr) -> ()
    %7 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%0, %7) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%0) : (!llvm.ptr) -> ()
    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "fanin_k", stream = 0 : i32}
    %8 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%5, %8) : (!llvm.ptr, i64) -> ()
    %9 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%5, %9) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%5) : (!llvm.ptr) -> ()
    ADORA.BlockStore %4, %arg2 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "2", KernelName = "fanin_k", stream = 0 : i32}
    return
  }
}


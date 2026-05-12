module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @linear_chain(%arg0: memref<32x32xf32>, %arg1: memref<32x32xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}]}]} {
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result = ADORA.BlockLoad %arg0 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "0", KernelName = "linear_k", stream = 0 : i32}
    %1 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%0, %1) : (!llvm.ptr, i64) -> ()
    %2 = ADORA.LocalMemAlloc memref<32x32xf32>  {Id = "1", KernelName = "linear_k"}
    %3 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %4 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%0, %4) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%0) : (!llvm.ptr) -> ()
    ADORA.kernel {
      ADORA.terminator
    } {KernelName = "linear_k", stream = 0 : i32}
    %5 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%3, %5) : (!llvm.ptr, i64) -> ()
    %6 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%3, %6) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%3) : (!llvm.ptr) -> ()
    ADORA.BlockStore %2, %arg1 [0, 0] : memref<32x32xf32> -> memref<32x32xf32>  {Id = "1", KernelName = "linear_k", stream = 0 : i32}
    return
  }
}


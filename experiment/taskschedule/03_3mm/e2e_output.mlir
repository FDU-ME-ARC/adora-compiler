module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @kernel_3mm(%arg0: memref<?x18xf32>, %arg1: memref<?x20xf32>, %arg2: memref<?x18xf32>, %arg3: memref<?x22xf32>, %arg4: memref<?x24xf32>, %arg5: memref<?x22xf32>, %arg6: memref<?x22xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 9 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 10 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 11 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 13 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 11 : i64, kind = "RAW", overlap = true, src = 9 : i64}]}]} {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result = ADORA.BlockLoad %arg1 [0, 0] : memref<?x20xf32> -> memref<16x20xf32>  {Id = "0", KernelName = "kernel_3mm_0", stream = 0 : i32}
    %1 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%0, %1) : (!llvm.ptr, i64) -> ()
    %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_0 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x18xf32> -> memref<20x18xf32>  {Id = "1", KernelName = "kernel_3mm_0", stream = 1 : i32}
    %3 = llvm.mlir.constant(1 : i64) : i64
    llvm.call @adoraEventRecord(%2, %3) : (!llvm.ptr, i64) -> ()
    %4 = ADORA.LocalMemAlloc memref<16x18xf32>  {Id = "2", KernelName = "kernel_3mm_0"}
    %5 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %6 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%2, %6) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%2) : (!llvm.ptr) -> ()
    %7 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%0, %7) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%0) : (!llvm.ptr) -> ()
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 18 {
          %24 = affine.for %arg9 = 0 to 20 iter_args(%arg10 = %cst) -> (f32) {
            %25 = affine.load %result[%arg7, %arg9] : memref<16x20xf32>
            %26 = affine.load %result_0[%arg9, %arg8] : memref<20x18xf32>
            %27 = arith.mulf %25, %26 : f32
            %28 = arith.addf %arg10, %27 : f32
            affine.yield %28 : f32
          }
          affine.store %24, %4[%arg7, %arg8] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0", stream = 0 : i32}
    %8 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%5, %8) : (!llvm.ptr, i64) -> ()
    %9 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%5, %9) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%5) : (!llvm.ptr) -> ()
    ADORA.BlockStore %4, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0", stream = 0 : i32}
    %10 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1", stream = 2 : i32}
    %11 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%10, %11) : (!llvm.ptr, i64) -> ()
    %12 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_2 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1", stream = 3 : i32}
    %13 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @adoraEventRecord(%12, %13) : (!llvm.ptr, i64) -> ()
    %14 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %15 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %16 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%12, %16) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%12) : (!llvm.ptr) -> ()
    %17 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%10, %17) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%10) : (!llvm.ptr) -> ()
    ADORA.kernel {
      affine.for %arg7 = 0 to 18 {
        affine.for %arg8 = 0 to 22 {
          %24 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %cst) -> (f32) {
            %25 = affine.load %result_1[%arg7, %arg9] : memref<18x24xf32>
            %26 = affine.load %result_2[%arg9, %arg8] : memref<24x22xf32>
            %27 = arith.mulf %25, %26 : f32
            %28 = arith.addf %arg10, %27 : f32
            affine.yield %28 : f32
          }
          affine.store %24, %14[%arg7, %arg8] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1", stream = 2 : i32}
    %18 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%15, %18) : (!llvm.ptr, i64) -> ()
    %19 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%15, %19) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%15) : (!llvm.ptr) -> ()
    ADORA.BlockStore %14, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1", stream = 2 : i32}
    %20 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    %21 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 22 {
          %24 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %cst) -> (f32) {
            %25 = affine.load %4[%arg7, %arg9] : memref<16x18xf32>
            %26 = affine.load %14[%arg9, %arg8] : memref<18x22xf32>
            %27 = arith.mulf %25, %26 : f32
            %28 = arith.addf %arg10, %27 : f32
            affine.yield %28 : f32
          }
          affine.store %24, %20[%arg7, %arg8] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2", stream = 0 : i32}
    %22 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%21, %22) : (!llvm.ptr, i64) -> ()
    %23 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%21, %23) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%21) : (!llvm.ptr) -> ()
    ADORA.BlockStore %20, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2", stream = 0 : i32}
    return
  }
}


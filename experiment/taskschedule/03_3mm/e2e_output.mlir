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
          %36 = affine.for %arg9 = 0 to 20 iter_args(%arg10 = %cst) -> (f32) {
            %37 = affine.load %result[%arg7, %arg9] : memref<16x20xf32>
            %38 = affine.load %result_0[%arg9, %arg8] : memref<20x18xf32>
            %39 = arith.mulf %37, %38 : f32
            %40 = arith.addf %arg10, %39 : f32
            affine.yield %40 : f32
          }
          affine.store %36, %4[%arg7, %arg8] : memref<16x18xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_0", stream = 0 : i32}
    %8 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%5, %8) : (!llvm.ptr, i64) -> ()
    %9 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %10 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%5, %10) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%5) : (!llvm.ptr) -> ()
    ADORA.BlockStore %4, %arg0 [0, 0] : memref<16x18xf32> -> memref<?x18xf32>  {Id = "2", KernelName = "kernel_3mm_0", stream = 0 : i32}
    %11 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%9, %11) : (!llvm.ptr, i64) -> ()
    %12 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_1 = ADORA.BlockLoad %arg4 [0, 0] : memref<?x24xf32> -> memref<18x24xf32>  {Id = "0", KernelName = "kernel_3mm_1", stream = 2 : i32}
    %13 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%12, %13) : (!llvm.ptr, i64) -> ()
    %14 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %result_2 = ADORA.BlockLoad %arg5 [0, 0] : memref<?x22xf32> -> memref<24x22xf32>  {Id = "1", KernelName = "kernel_3mm_1", stream = 3 : i32}
    %15 = llvm.mlir.constant(3 : i64) : i64
    llvm.call @adoraEventRecord(%14, %15) : (!llvm.ptr, i64) -> ()
    %16 = ADORA.LocalMemAlloc memref<18x22xf32>  {Id = "2", KernelName = "kernel_3mm_1"}
    %17 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %18 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%14, %18) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%14) : (!llvm.ptr) -> ()
    %19 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%12, %19) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%12) : (!llvm.ptr) -> ()
    ADORA.kernel {
      affine.for %arg7 = 0 to 18 {
        affine.for %arg8 = 0 to 22 {
          %36 = affine.for %arg9 = 0 to 24 iter_args(%arg10 = %cst) -> (f32) {
            %37 = affine.load %result_1[%arg7, %arg9] : memref<18x24xf32>
            %38 = affine.load %result_2[%arg9, %arg8] : memref<24x22xf32>
            %39 = arith.mulf %37, %38 : f32
            %40 = arith.addf %arg10, %39 : f32
            affine.yield %40 : f32
          }
          affine.store %36, %16[%arg7, %arg8] : memref<18x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_1", stream = 2 : i32}
    %20 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%17, %20) : (!llvm.ptr, i64) -> ()
    %21 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %22 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%17, %22) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%17) : (!llvm.ptr) -> ()
    ADORA.BlockStore %16, %arg3 [0, 0] : memref<18x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_1", stream = 2 : i32}
    %23 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%21, %23) : (!llvm.ptr, i64) -> ()
    %24 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %25 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%9, %25) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%9) : (!llvm.ptr) -> ()
    %result_3 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x18xf32> -> memref<16x18xf32>  {Id = "0", KernelName = "kernel_3mm_2", stream = 0 : i32}
    %26 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%24, %26) : (!llvm.ptr, i64) -> ()
    %27 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %28 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventWait(%21, %28) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%21) : (!llvm.ptr) -> ()
    %result_4 = ADORA.BlockLoad %arg3 [0, 0] : memref<?x22xf32> -> memref<18x22xf32>  {Id = "1", KernelName = "kernel_3mm_2", stream = 2 : i32}
    %29 = llvm.mlir.constant(2 : i64) : i64
    llvm.call @adoraEventRecord(%27, %29) : (!llvm.ptr, i64) -> ()
    %30 = ADORA.LocalMemAlloc memref<16x22xf32>  {Id = "2", KernelName = "kernel_3mm_2"}
    %31 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %32 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%24, %32) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%24) : (!llvm.ptr) -> ()
    %33 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%27, %33) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%27) : (!llvm.ptr) -> ()
    ADORA.kernel {
      affine.for %arg7 = 0 to 16 {
        affine.for %arg8 = 0 to 22 {
          %36 = affine.for %arg9 = 0 to 18 iter_args(%arg10 = %cst) -> (f32) {
            %37 = affine.load %result_3[%arg7, %arg9] : memref<16x18xf32>
            %38 = affine.load %result_4[%arg9, %arg8] : memref<18x22xf32>
            %39 = arith.mulf %37, %38 : f32
            %40 = arith.addf %arg10, %39 : f32
            affine.yield %40 : f32
          }
          affine.store %36, %30[%arg7, %arg8] : memref<16x22xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "kernel_3mm_2", stream = 0 : i32}
    %34 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventRecord(%31, %34) : (!llvm.ptr, i64) -> ()
    %35 = llvm.mlir.constant(0 : i64) : i64
    llvm.call @adoraEventWait(%31, %35) : (!llvm.ptr, i64) -> ()
    llvm.call @adoraEventDestroy(%31) : (!llvm.ptr) -> ()
    ADORA.BlockStore %30, %arg6 [0, 0] : memref<16x22xf32> -> memref<?x22xf32>  {Id = "2", KernelName = "kernel_3mm_2", stream = 0 : i32}
    return
  }
}


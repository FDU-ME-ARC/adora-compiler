module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @gesummv_kernel_0(%arg0: memref<250xf32>, %arg1: memref<250xf32>, %arg2: memref<250x250xf32>, %arg3: memref<250xf32>, %arg4: memref<250x250xf32>) attributes {Kernel, adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 6 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 3 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = false, kind = "LC-RAR", step = 5 : i64}, {exact = true, kind = "LC-RAR", step = 5 : i64}, {exact = false, kind = "LC-RAR", step = 5 : i64}, {exact = false, kind = "LC-WAW", step = 5 : i64}, {exact = false, kind = "LC-WAW", step = 5 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}], gesummv_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %cst = arith.constant 0.000000e+00 : f32
    %cst_0 = arith.constant 1.500000e+00 : f32
    %cst_1 = arith.constant 1.200000e+00 : f32
    %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    %1 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
    affine.for %arg5 = 0 to 250 step 5 {
      %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result = ADORA.BlockLoad %arg2 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "0", KernelName = "gesummv_kernel_0", stream = 0 : i32}
      %3 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%2, %3) : (!llvm.ptr, i64) -> ()
      %4 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result_2 = ADORA.BlockLoad %arg3 [0] : memref<250xf32> -> memref<250xf32>  {Id = "1", KernelName = "gesummv_kernel_0", stream = 1 : i32}
      %5 = llvm.mlir.constant(1 : i64) : i64
      llvm.call @adoraEventRecord(%4, %5) : (!llvm.ptr, i64) -> ()
      %6 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %result_3 = ADORA.BlockLoad %arg4 [%arg5, 0] : memref<250x250xf32> -> memref<5x250xf32>  {Id = "2", KernelName = "gesummv_kernel_0", stream = 2 : i32}
      %7 = llvm.mlir.constant(2 : i64) : i64
      llvm.call @adoraEventRecord(%6, %7) : (!llvm.ptr, i64) -> ()
      %8 = ADORA.LocalMemAlloc memref<6xf32>  {Id = "3", KernelName = "gesummv_kernel_0"}
      %9 = ADORA.LocalMemAlloc memref<6xf32>  {Id = "4", KernelName = "gesummv_kernel_0"}
      %10 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %11 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%6, %11) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%6) : (!llvm.ptr) -> ()
      %12 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%4, %12) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%4) : (!llvm.ptr) -> ()
      %13 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%2, %13) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%2) : (!llvm.ptr) -> ()
      ADORA.kernel {
        affine.for %arg6 = 0 to 5 {
          %21:2 = affine.for %arg7 = 0 to 250 step 2 iter_args(%arg8 = %cst, %arg9 = %cst) -> (f32, f32) {
            %25 = affine.load %result[%arg6, %arg7] : memref<5x250xf32>
            %26 = affine.load %result_2[%arg7] : memref<250xf32>
            %27 = arith.mulf %25, %26 : f32
            %28 = arith.addf %27, %arg8 : f32
            %29 = affine.load %result_3[%arg6, %arg7] : memref<5x250xf32>
            %30 = arith.mulf %29, %26 : f32
            %31 = arith.addf %30, %arg9 : f32
            %32 = affine.load %result[%arg6, %arg7 + 1] : memref<5x250xf32>
            %33 = affine.load %result_2[%arg7 + 1] : memref<250xf32>
            %34 = arith.mulf %32, %33 : f32
            %35 = arith.addf %34, %28 : f32
            %36 = affine.load %result_3[%arg6, %arg7 + 1] : memref<5x250xf32>
            %37 = arith.mulf %36, %33 : f32
            %38 = arith.addf %37, %31 : f32
            affine.yield %35, %38 : f32, f32
          }
          affine.store %21#0, %8[%arg6] : memref<6xf32>
          %22 = arith.mulf %21#0, %cst_0 : f32
          %23 = arith.mulf %21#1, %cst_1 : f32
          %24 = arith.addf %22, %23 : f32
          affine.store %24, %9[%arg6] : memref<6xf32>
        }
        ADORA.terminator
      } {KernelName = "gesummv_kernel_0", stream = 0 : i32}
      %14 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%10, %14) : (!llvm.ptr, i64) -> ()
      %15 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %16 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%10, %16) : (!llvm.ptr, i64) -> ()
      ADORA.BlockStore %9, %arg1 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "4", KernelName = "gesummv_kernel_0", stream = 0 : i32}
      %17 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%15, %17) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%15) : (!llvm.ptr) -> ()
      %18 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
      %19 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventWait(%10, %19) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%10) : (!llvm.ptr) -> ()
      ADORA.BlockStore %8, %arg0 [%arg5] : memref<6xf32> -> memref<250xf32>  {Id = "3", KernelName = "gesummv_kernel_0", stream = 0 : i32}
      %20 = llvm.mlir.constant(0 : i64) : i64
      llvm.call @adoraEventRecord(%18, %20) : (!llvm.ptr, i64) -> ()
      llvm.call @adoraEventDestroy(%18) : (!llvm.ptr) -> ()
    }
    return
  }
}


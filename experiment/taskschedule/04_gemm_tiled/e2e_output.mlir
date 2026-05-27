module attributes {adora.scheduled} {
  llvm.func @adoraEventDestroy(!llvm.ptr)
  llvm.func @adoraEventWait(!llvm.ptr, i64)
  llvm.func @adoraEventRecord(!llvm.ptr, i64)
  llvm.func @adoraEventCreate() -> !llvm.ptr
  func.func @gemm_tiled(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 5 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = false, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}]} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %0 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
        %1 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
        %2 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
        affine.for %arg5 = 0 to 4 {
          %3 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
          %result = ADORA.BlockLoad %arg2 [%arg3 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "0", KernelName = "gemm_tiled_tk", stream = 0 : i32}
          %4 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventRecord(%3, %4) : (!llvm.ptr, i64) -> ()
          %5 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
          %result_0 = ADORA.BlockLoad %arg0 [%arg3 * 16, %arg5 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "1", KernelName = "gemm_tiled_tk", stream = 1 : i32}
          %6 = llvm.mlir.constant(1 : i64) : i64
          llvm.call @adoraEventRecord(%5, %6) : (!llvm.ptr, i64) -> ()
          %7 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
          %result_1 = ADORA.BlockLoad %arg1 [%arg5 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "2", KernelName = "gemm_tiled_tk", stream = 2 : i32}
          %8 = llvm.mlir.constant(2 : i64) : i64
          llvm.call @adoraEventRecord(%7, %8) : (!llvm.ptr, i64) -> ()
          %9 = ADORA.LocalMemAlloc memref<16x16xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          %10 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
          %11 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventWait(%7, %11) : (!llvm.ptr, i64) -> ()
          llvm.call @adoraEventDestroy(%7) : (!llvm.ptr) -> ()
          %12 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventWait(%5, %12) : (!llvm.ptr, i64) -> ()
          llvm.call @adoraEventDestroy(%5) : (!llvm.ptr) -> ()
          %13 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventWait(%3, %13) : (!llvm.ptr, i64) -> ()
          ADORA.kernel {
            affine.for %arg6 = 0 to 16 {
              affine.for %arg7 = 0 to 16 {
                %19 = affine.load %result[%arg6, %arg7] : memref<16x16xf32>
                %20 = affine.for %arg8 = 0 to 16 iter_args(%arg9 = %19) -> (f32) {
                  %21 = affine.load %result_0[%arg6, %arg8] : memref<16x16xf32>
                  %22 = affine.load %result_1[%arg8, %arg7] : memref<16x16xf32>
                  %23 = arith.mulf %21, %22 : f32
                  %24 = arith.addf %arg9, %23 : f32
                  affine.yield %24 : f32
                }
                affine.store %20, %9[%arg6, %arg7] : memref<16x16xf32>
              }
            }
            ADORA.terminator
          } {KernelName = "gemm_tiled_tk", stream = 0 : i32}
          %14 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventRecord(%10, %14) : (!llvm.ptr, i64) -> ()
          %15 = llvm.call @adoraEventCreate() : () -> !llvm.ptr
          %16 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventWait(%10, %16) : (!llvm.ptr, i64) -> ()
          llvm.call @adoraEventDestroy(%10) : (!llvm.ptr) -> ()
          %17 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventWait(%3, %17) : (!llvm.ptr, i64) -> ()
          llvm.call @adoraEventDestroy(%3) : (!llvm.ptr) -> ()
          ADORA.BlockStore %9, %arg2 [%arg3 * 16, %arg4 * 16] : memref<16x16xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "gemm_tiled_tk", stream = 0 : i32}
          %18 = llvm.mlir.constant(0 : i64) : i64
          llvm.call @adoraEventRecord(%15, %18) : (!llvm.ptr, i64) -> ()
          llvm.call @adoraEventDestroy(%15) : (!llvm.ptr) -> ()
        }
      }
    }
    return
  }
}


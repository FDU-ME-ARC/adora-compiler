module attributes {adora.scheduled} {
  func.func @fir(%arg0: memref<100xi32>, %arg1: memref<100xi32>) -> memref<i32> attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}]}]} {
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %result, %asyncToken = ADORA.BlockLoad %arg1 [0] : memref<100xi32> -> memref<100xi32>  {Id = "0", KernelName = "kernel_fir"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg0 [0] : memref<100xi32> -> memref<100xi32>  {Id = "1", KernelName = "kernel_fir"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "2", KernelName = "kernel_fir"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      %2 = affine.for %arg2 = 0 to 100 iter_args(%arg3 = %c0_i32) -> (i32) {
        %3 = affine.load %result[%arg2] : memref<100xi32>
        %4 = affine.load %result_0[-%arg2 + 99] : memref<100xi32>
        %5 = arith.muli %3, %4 : i32
        %6 = arith.addi %arg3, %5 : i32
        affine.yield %6 : i32
      }
      affine.store %2, %0[0] : memref<2xi32>
      ADORA.terminator
    } {KernelName = "kernel_fir"}
    ADORA.BlockStore async [%1] %0, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "2", KernelName = "kernel_fir"}
    return %alloca : memref<i32>
  }
}


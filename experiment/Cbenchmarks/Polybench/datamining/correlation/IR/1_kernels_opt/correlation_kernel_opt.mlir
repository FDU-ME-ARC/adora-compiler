#map = affine_map<(d0) -> (-d0 + 27)>
module attributes {adora.scheduled} {
  func.func @correlation(%arg0: f32, %arg1: memref<?x28xf32>, %arg2: memref<?x28xf32>, %arg3: memref<?xf32>, %arg4: memref<?xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 7 : i64, kind = "RAW", overlap = true, src = 5 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 7 : i64}, {dst = 8 : i64, kind = "RAW", overlap = true, src = 6 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 9 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 10 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 11 : i64}, {dst = 13 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 13 : i64}, {dst = 14 : i64, kind = "RAW", overlap = true, src = 12 : i64}, {dst = 18 : i64, kind = "RAW", overlap = true, src = 17 : i64}, {dst = 18 : i64, kind = "RAW", overlap = true, src = 15 : i64}, {dst = 18 : i64, kind = "RAW", overlap = true, src = 16 : i64}, {dst = 19 : i64, kind = "RAW", overlap = true, src = 18 : i64}, {dst = 19 : i64, kind = "RAW", overlap = true, src = 17 : i64}, {dst = 24 : i64, kind = "RAW", overlap = true, src = 20 : i64}, {dst = 24 : i64, kind = "RAW", overlap = true, src = 21 : i64}, {dst = 24 : i64, kind = "RAW", overlap = true, src = 22 : i64}, {dst = 24 : i64, kind = "RAW", overlap = true, src = 23 : i64}, {dst = 25 : i64, kind = "RAW", overlap = true, src = 24 : i64}, {dst = 25 : i64, kind = "RAW", overlap = true, src = 23 : i64}, {dst = 29 : i64, kind = "RAW", overlap = true, src = 26 : i64}, {dst = 29 : i64, kind = "RAW", overlap = true, src = 28 : i64}, {dst = 29 : i64, kind = "RAW", overlap = true, src = 27 : i64}, {dst = 30 : i64, kind = "RAW", overlap = true, src = 29 : i64}, {dst = 30 : i64, kind = "RAW", overlap = true, src = 28 : i64}, {dst = 19 : i64, kind = "WAR", overlap = true, src = 11 : i64}, {dst = 22 : i64, kind = "RAR", overlap = true, src = 11 : i64}, {dst = 14 : i64, kind = "WAR", overlap = true, src = 11 : i64}, {dst = 26 : i64, kind = "RAW", overlap = true, src = 25 : i64}, {dst = 27 : i64, kind = "RAW", overlap = false, src = 25 : i64}, {dst = 16 : i64, kind = "RAR", overlap = true, src = 10 : i64}, {dst = 20 : i64, kind = "RAR", overlap = true, src = 10 : i64}, {dst = 25 : i64, kind = "WAR", overlap = true, src = 9 : i64}, {dst = 26 : i64, kind = "RAR", overlap = true, src = 9 : i64}, {dst = 15 : i64, kind = "RAR", overlap = true, src = 9 : i64}, {dst = 21 : i64, kind = "RAR", overlap = true, src = 9 : i64}, {dst = 27 : i64, kind = "RAR", overlap = false, src = 9 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 16 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 20 : i64, kind = "RAW", overlap = true, src = 8 : i64}, {dst = 27 : i64, kind = "RAR", overlap = false, src = 26 : i64}, {dst = 10 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 8 : i64, kind = "WAW", overlap = true, src = 4 : i64}, {dst = 16 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 5 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 20 : i64, kind = "RAW", overlap = true, src = 4 : i64}, {dst = 22 : i64, kind = "RAW", overlap = true, src = 19 : i64}, {dst = 20 : i64, kind = "RAR", overlap = true, src = 16 : i64}, {dst = 25 : i64, kind = "WAR", overlap = true, src = 15 : i64}, {dst = 26 : i64, kind = "RAR", overlap = true, src = 15 : i64}, {dst = 21 : i64, kind = "RAR", overlap = true, src = 15 : i64}, {dst = 27 : i64, kind = "RAR", overlap = false, src = 15 : i64}, {dst = 10 : i64, kind = "RAR", overlap = true, src = 1 : i64}, {dst = 8 : i64, kind = "WAR", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "WAR", overlap = true, src = 1 : i64}, {dst = 16 : i64, kind = "RAR", overlap = true, src = 1 : i64}, {dst = 5 : i64, kind = "RAR", overlap = true, src = 1 : i64}, {dst = 20 : i64, kind = "RAR", overlap = true, src = 1 : i64}, {dst = 10 : i64, kind = "RAR", overlap = true, src = 5 : i64}, {dst = 8 : i64, kind = "WAR", overlap = true, src = 5 : i64}, {dst = 16 : i64, kind = "RAR", overlap = true, src = 5 : i64}, {dst = 20 : i64, kind = "RAR", overlap = true, src = 5 : i64}, {dst = 25 : i64, kind = "WAR", overlap = true, src = 21 : i64}, {dst = 26 : i64, kind = "RAR", overlap = true, src = 21 : i64}, {dst = 27 : i64, kind = "RAR", overlap = false, src = 21 : i64}, {dst = 25 : i64, kind = "WAR", overlap = true, src = 0 : i64}, {dst = 9 : i64, kind = "RAR", overlap = true, src = 0 : i64}, {dst = 26 : i64, kind = "RAR", overlap = true, src = 0 : i64}, {dst = 15 : i64, kind = "RAR", overlap = true, src = 0 : i64}, {dst = 21 : i64, kind = "RAR", overlap = true, src = 0 : i64}, {dst = 27 : i64, kind = "RAR", overlap = false, src = 0 : i64}, {dst = 19 : i64, kind = "WAW", overlap = true, src = 14 : i64}, {dst = 22 : i64, kind = "RAW", overlap = true, src = 14 : i64}]}], llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 5.65685415 : f32
    %cst_0 = arith.constant 1.000000e+00 : f32
    %cst_1 = arith.constant 0.000000e+00 : f32
    %cst_2 = arith.constant 1.000000e-01 : f32
    %cst_3 = arith.constant 3.200000e+01 : f32
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg3[%arg5] : memref<?xf32>
    }
    %result, %asyncToken = ADORA.BlockLoad %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_0"} -> !ADORA.token
    %result_4, %asyncToken_5 = ADORA.BlockLoad %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_0"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "2", KernelName = "correlation_0"}
    %1 = ADORA.kernel async [%asyncToken_5, %asyncToken] {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %17 = affine.load %result[%arg5, %arg6] : memref<32x28xf32>
          %18 = affine.load %result_4[%arg6] : memref<28xf32>
          %19 = arith.addf %18, %17 : f32
          affine.store %19, %0[%arg6] : memref<28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_0"}
    %2 = ADORA.BlockStore async [%1, %asyncToken_5] %0, %arg3 [0] : memref<28xf32> -> memref<?xf32>  {Id = "2", KernelName = "correlation_0"} -> !ADORA.token
    %result_6, %asyncToken_7 = ADORA.BlockLoad async [%2, %asyncToken_5] %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "0", KernelName = "correlation_1"} -> !ADORA.token
    %3 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "1", KernelName = "correlation_1"}
    %4 = ADORA.kernel async [%asyncToken_7] {
      affine.for %arg5 = 0 to 28 {
        %17 = affine.load %result_6[%arg5] : memref<28xf32>
        %18 = arith.divf %17, %cst_3 : f32
        affine.store %18, %3[%arg5] : memref<28xf32>
      }
      ADORA.terminator
    } {KernelName = "correlation_1"}
    %5 = ADORA.BlockStore async [%4, %2, %asyncToken_5, %asyncToken_7] %3, %arg3 [0] : memref<28xf32> -> memref<?xf32>  {Id = "1", KernelName = "correlation_1"} -> !ADORA.token
    affine.for %arg5 = 0 to 28 {
      affine.store %cst_1, %arg4[%arg5] : memref<?xf32>
    }
    %result_8, %asyncToken_9 = ADORA.BlockLoad async [%asyncToken] %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_2"} -> !ADORA.token
    %result_10, %asyncToken_11 = ADORA.BlockLoad async [%5, %2, %asyncToken_5, %asyncToken_7] %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_2"} -> !ADORA.token
    %result_12, %asyncToken_13 = ADORA.BlockLoad %arg4 [0] : memref<?xf32> -> memref<28xf32>  {Id = "2", KernelName = "correlation_2"} -> !ADORA.token
    %6 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "3", KernelName = "correlation_2"}
    %7 = ADORA.kernel async [%asyncToken_9, %asyncToken_11, %asyncToken_13] {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %17 = affine.load %result_8[%arg5, %arg6] : memref<32x28xf32>
          %18 = affine.load %result_10[%arg6] : memref<28xf32>
          %19 = arith.subf %17, %18 : f32
          %20 = arith.mulf %19, %19 : f32
          %21 = affine.load %result_12[%arg6] : memref<28xf32>
          %22 = arith.addf %21, %20 : f32
          affine.store %22, %6[%arg6] : memref<28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_2"}
    %8 = ADORA.BlockStore async [%7, %asyncToken_13] %6, %arg4 [0] : memref<28xf32> -> memref<?xf32>  {Id = "3", KernelName = "correlation_2"} -> !ADORA.token
    %result_14, %asyncToken_15 = ADORA.BlockLoad async [%asyncToken_9, %asyncToken] %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "0", KernelName = "correlation_3"} -> !ADORA.token
    %result_16, %asyncToken_17 = ADORA.BlockLoad async [%asyncToken_11, %5, %2, %asyncToken_5, %asyncToken_7] %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "1", KernelName = "correlation_3"} -> !ADORA.token
    %9 = ADORA.LocalMemAlloc memref<28xf32>  {Id = "2", KernelName = "correlation_3"}
    %10 = ADORA.kernel async [%asyncToken_15, %asyncToken_17] {
      affine.for %arg5 = 0 to 28 {
        %17 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %cst_1) -> (f32) {
          %22 = affine.load %result_14[%arg6, %arg5] : memref<32x28xf32>
          %23 = affine.load %result_16[%arg5] : memref<28xf32>
          %24 = arith.subf %22, %23 : f32
          %25 = arith.mulf %24, %24 : f32
          %26 = arith.addf %arg7, %25 : f32
          affine.yield %26 : f32
        }
        %18 = arith.divf %17, %cst_3 : f32
        %19 = math.sqrt %18 : f32
        %20 = arith.cmpf ole, %19, %cst_2 : f32
        %21 = arith.select %20, %cst_0, %19 : f32
        affine.store %21, %9[%arg5] : memref<28xf32>
      }
      ADORA.terminator
    } {KernelName = "correlation_3"}
    %11 = ADORA.BlockStore async [%10, %asyncToken_13, %8] %9, %arg4 [0] : memref<28xf32> -> memref<?xf32>  {Id = "2", KernelName = "correlation_3"} -> !ADORA.token
    %result_18, %asyncToken_19 = ADORA.BlockLoad async [%asyncToken_11, %5, %2, %asyncToken_17, %asyncToken_5, %asyncToken_7] %arg3 [0] : memref<?xf32> -> memref<28xf32>  {Id = "0", KernelName = "correlation_4"} -> !ADORA.token
    %result_20, %asyncToken_21 = ADORA.BlockLoad async [%asyncToken_9, %asyncToken_15, %asyncToken] %arg1 [0, 0] : memref<?x28xf32> -> memref<32x28xf32>  {Id = "1", KernelName = "correlation_4"} -> !ADORA.token
    %result_22, %asyncToken_23 = ADORA.BlockLoad async [%asyncToken_13, %11, %8] %arg4 [0] : memref<?xf32> -> memref<28xf32>  {Id = "2", KernelName = "correlation_4"} -> !ADORA.token
    %12 = ADORA.LocalMemAlloc memref<32x28xf32>  {Id = "3", KernelName = "correlation_4"}
    %13 = ADORA.kernel async [%asyncToken_19, %asyncToken_21, %asyncToken_23] {
      affine.for %arg5 = 0 to 32 {
        affine.for %arg6 = 0 to 28 {
          %17 = affine.load %result_18[%arg6] : memref<28xf32>
          %18 = affine.load %result_20[%arg5, %arg6] : memref<32x28xf32>
          %19 = arith.subf %18, %17 : f32
          %20 = affine.load %result_22[%arg6] : memref<28xf32>
          %21 = arith.mulf %20, %cst : f32
          %22 = arith.divf %19, %21 : f32
          affine.store %22, %12[%arg5, %arg6] : memref<32x28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_4"}
    %14 = ADORA.BlockStore async [%13, %asyncToken_9, %asyncToken_15, %asyncToken_21, %asyncToken] %12, %arg1 [0, 0] : memref<32x28xf32> -> memref<?x28xf32>  {Id = "3", KernelName = "correlation_4"} -> !ADORA.token
    %result_24, %asyncToken_25 = ADORA.BlockLoad async [%14, %asyncToken_9, %asyncToken_15, %asyncToken_21, %asyncToken] %arg1 [0, 0] : memref<?x28xf32> -> memref<32x27xf32>  {Id = "0", KernelName = "correlation_5"} -> !ADORA.token
    %result_26, %asyncToken_27 = ADORA.BlockLoad async [%14, %asyncToken_9, %asyncToken_25, %asyncToken_15, %asyncToken_21, %asyncToken] %arg1 [0, 1] : memref<?x28xf32> -> memref<32x27xf32>  {Id = "1", KernelName = "correlation_5"} -> !ADORA.token
    %15 = ADORA.LocalMemAlloc memref<28x28xf32>  {Id = "2", KernelName = "correlation_5"}
    %16 = ADORA.kernel async [%asyncToken_25, %asyncToken_27] {
      affine.for %arg5 = 0 to 27 {
        affine.store %cst_0, %15[%arg5, %arg5] : memref<28x28xf32>
        affine.for %arg6 = 0 to #map(%arg5) {
          %17 = affine.for %arg7 = 0 to 32 iter_args(%arg8 = %cst_1) -> (f32) {
            %18 = affine.load %result_24[%arg7, %arg5] : memref<32x27xf32>
            %19 = affine.load %result_26[%arg7, %arg5 + %arg6] : memref<32x27xf32>
            %20 = arith.mulf %18, %19 : f32
            %21 = arith.addf %arg8, %20 : f32
            affine.yield %21 : f32
          }
          affine.store %17, %15[%arg5, %arg5 + %arg6 + 1] : memref<28x28xf32>
          affine.store %17, %15[%arg5 + %arg6 + 1, %arg5] : memref<28x28xf32>
        }
      }
      ADORA.terminator
    } {KernelName = "correlation_5"}
    ADORA.BlockStore async [%16] %15, %arg2 [0, 0] : memref<28x28xf32> -> memref<?x28xf32>  {Id = "2", KernelName = "correlation_5"}
    affine.store %cst_0, %arg2[27, 27] : memref<?x28xf32>
    return
  }
}


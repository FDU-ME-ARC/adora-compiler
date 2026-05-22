module attributes {adora.scheduled} {
  func.func @ss_sort(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0 to %c32 step %c2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      func.call @init(%arg2) : (memref<?xi32>) -> ()
      %2 = arith.cmpi eq, %arg5, %c0_i32 : i32
      scf.if %2 {
        func.call @hist(%arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @hist(%arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      func.call @local_scan(%arg2) : (memref<?xi32>) -> ()
      func.call @sum_scan(%arg3, %arg2) : (memref<?xi32>, memref<?xi32>) -> ()
      func.call @last_step_scan(%arg2, %arg3) : (memref<?xi32>, memref<?xi32>) -> ()
      %3 = arith.extui %2 : i1 to i32
      scf.if %2 {
        func.call @update(%arg1, %arg2, %arg0, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      } else {
        func.call @update(%arg0, %arg2, %arg1, %1) : (memref<?xi32>, memref<?xi32>, memref<?xi32>, i32) -> ()
      }
      scf.yield %3 : i32
    }
    return
  }
  func.func @init(%arg0: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<available_externally>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg1 = 0 to 2048 {
      affine.store %c0_i32, %arg0[%arg1] : memref<?xi32>
    }
    return
  }
  func.func @hist(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 1 : i64, kind = "RAW", overlap = true, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<available_externally>} {
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    %result, %asyncToken = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "hist"} -> !ADORA.token
    ADORA.kernel {
      affine.for %arg3 = 0 to 512 {
        %0 = arith.index_cast %arg3 : index to i32
        affine.for %arg4 = 0 to 4 {
          %1 = affine.load %result[%arg4 + %arg3 * 4] : memref<2048xi32>
          %2 = arith.shrsi %1, %arg2 : i32
          %3 = arith.andi %2, %c3_i32 : i32
          %4 = arith.muli %3, %c512_i32 : i32
          %5 = arith.addi %4, %0 : i32
          %6 = arith.addi %5, %c1_i32 : i32
          %7 = arith.index_cast %6 : i32 to index
          %8 = memref.load %arg0[%7] : memref<?xi32>
          %9 = arith.addi %8, %c1_i32 : i32
          memref.store %9, %arg0[%7] : memref<?xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "hist"}
    return
  }
  func.func @local_scan(%arg0: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "WAR", overlap = true, src = 1 : i64}, {dst = 4 : i64, kind = "WAR", overlap = false, src = 0 : i64}, {dst = 1 : i64, kind = "RAR", overlap = false, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<available_externally>} {
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<2047xi32>  {Id = "0", KernelName = "local_scan"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad async [%asyncToken] %arg0 [1] : memref<?xi32> -> memref<2047xi32>  {Id = "1", KernelName = "local_scan"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<2047xi32>  {Id = "2", KernelName = "local_scan"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      affine.for %arg1 = 0 to 128 {
        affine.for %arg2 = 0 to 15 {
          %2 = affine.load %result[%arg1 * 16 + %arg2] : memref<2047xi32>
          %3 = affine.load %result_0[%arg1 * 16 + %arg2] : memref<2047xi32>
          %4 = arith.addi %3, %2 : i32
          affine.store %4, %0[%arg1 * 16 + %arg2] : memref<2047xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "local_scan"}
    ADORA.BlockStore async [%1, %asyncToken_1, %asyncToken] %0, %arg0 [1] : memref<2047xi32> -> memref<?xi32>  {Id = "2", KernelName = "local_scan"}
    return
  }
  func.func @sum_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "WAR", overlap = false, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<available_externally>} {
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg0[0] : memref<?xi32>
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<127xi32>  {Id = "0", KernelName = "sum_scan"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg1 [15] : memref<?xi32> -> memref<2017xi32>  {Id = "1", KernelName = "sum_scan"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<127xi32>  {Id = "2", KernelName = "sum_scan"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      affine.for %arg2 = 0 to 127 {
        %2 = affine.load %result[%arg2] : memref<127xi32>
        %3 = affine.load %result_0[%arg2 * 16] : memref<2017xi32>
        %4 = arith.addi %2, %3 : i32
        affine.store %4, %0[%arg2] : memref<127xi32>
      }
      ADORA.terminator
    } {KernelName = "sum_scan"}
    ADORA.BlockStore async [%1, %asyncToken] %0, %arg0 [1] : memref<127xi32> -> memref<?xi32>  {Id = "2", KernelName = "sum_scan"}
    return
  }
  func.func @last_step_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 3 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 4 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 4 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<available_externally>} {
    %result, %asyncToken = ADORA.BlockLoad %arg0 [0] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "last_step_scan"} -> !ADORA.token
    %result_0, %asyncToken_1 = ADORA.BlockLoad %arg1 [0] : memref<?xi32> -> memref<128xi32>  {Id = "1", KernelName = "last_step_scan"} -> !ADORA.token
    %0 = ADORA.LocalMemAlloc memref<2048xi32>  {Id = "2", KernelName = "last_step_scan"}
    %1 = ADORA.kernel async [%asyncToken_1, %asyncToken] {
      affine.for %arg2 = 0 to 128 {
        affine.for %arg3 = 0 to 16 {
          %2 = affine.load %result[%arg3 + %arg2 * 16] : memref<2048xi32>
          %3 = affine.load %result_0[%arg2] : memref<128xi32>
          %4 = arith.addi %2, %3 : i32
          affine.store %4, %0[%arg3 + %arg2 * 16] : memref<2048xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "last_step_scan"}
    ADORA.BlockStore async [%1, %asyncToken] %0, %arg0 [0] : memref<2048xi32> -> memref<?xi32>  {Id = "2", KernelName = "last_step_scan"}
    return
  }
  func.func @update(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 1 : i64, kind = "RAW", overlap = true, src = 0 : i64}]}], llvm.linkage = #llvm.linkage<available_externally>} {
    %c1_i32 = arith.constant 1 : i32
    %c512_i32 = arith.constant 512 : i32
    %c3_i32 = arith.constant 3 : i32
    %result, %asyncToken = ADORA.BlockLoad %arg2 [0] : memref<?xi32> -> memref<2048xi32>  {Id = "0", KernelName = "update"} -> !ADORA.token
    ADORA.kernel {
      affine.for %arg4 = 0 to 512 {
        %0 = arith.index_cast %arg4 : index to i32
        affine.for %arg5 = 0 to 4 {
          %1 = affine.load %result[%arg5 + %arg4 * 4] : memref<2048xi32>
          %2 = arith.shrsi %1, %arg3 : i32
          %3 = arith.andi %2, %c3_i32 : i32
          %4 = arith.muli %3, %c512_i32 : i32
          %5 = arith.addi %4, %0 : i32
          %6 = arith.index_cast %5 : i32 to index
          %7 = memref.load %arg1[%6] : memref<?xi32>
          %8 = arith.index_cast %7 : i32 to index
          memref.store %1, %arg0[%8] : memref<?xi32>
          %9 = memref.load %arg1[%6] : memref<?xi32>
          %10 = arith.addi %9, %c1_i32 : i32
          memref.store %10, %arg1[%6] : memref<?xi32>
        }
      }
      ADORA.terminator
    } {KernelName = "update"}
    return
  }
}


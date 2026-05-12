module attributes {adora.scheduled} {
  func.func @loop_carried_min(%arg0: memref<16xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 2 : i64, kind = "RAW", overlap = true, src = 1 : i64}, {dst = 2 : i64, kind = "RAW", overlap = true, src = 0 : i64}, {dst = 3 : i64, kind = "RAW", overlap = true, src = 2 : i64}, {dst = 3 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAR", step = 1 : i64}, {exact = true, kind = "LC-WAR", step = 1 : i64}, {exact = true, kind = "LC-RAW", step = 1 : i64}, {exact = true, kind = "LC-WAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}]} {
    %0 = ADORA.event.create -> !ADORA.token
    %1 = ADORA.event.create -> !ADORA.token
    %2 = ADORA.event.create -> !ADORA.token
    %3:3 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %2) -> (!ADORA.token, !ADORA.token, !ADORA.token) {
      %result, %asyncToken = ADORA.BlockLoad async [%arg3] %arg0 [0] : memref<16xf32> -> memref<16xf32>  {Id = "0", KernelName = "loop_carried_min_kernel"} -> !ADORA.token
      %4 = ADORA.LocalMemAlloc memref<16xf32>  {Id = "1", KernelName = "loop_carried_min_kernel"}
      %5 = ADORA.kernel async [%asyncToken] {
        affine.for %arg5 = 0 to 16 {
          %7 = affine.load %result[%arg5] : memref<16xf32>
          affine.store %7, %4[%arg5] : memref<16xf32>
        }
        ADORA.terminator
      } {KernelName = "loop_carried_min_kernel"}
      %6 = ADORA.BlockStore async [%5, %asyncToken, %arg2, %arg4] %4, %arg0 [0] : memref<16xf32> -> memref<16xf32>  {Id = "2", KernelName = "loop_carried_min_kernel"} -> !ADORA.token
      affine.yield %asyncToken, %6, %6 : !ADORA.token, !ADORA.token, !ADORA.token
    }
    return
  }
}


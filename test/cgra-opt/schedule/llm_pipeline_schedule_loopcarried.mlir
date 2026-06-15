// Test: LLMPipelineSchedule recognises iteration-to-iteration kernel reuse.
//
// A single kernel inside an affine.for that carries !ADORA.token iter_args is a
// loop-carried task: its "previous task" is the PREVIOUS LOOP ITERATION.  Before
// this was supported, the pass collected only one (Load,Kernel,Store) triple in
// the loop body, hit `tasks.size() < 2`, and early-returned — missing all
// cross-iteration pipelining.
//
// Here the C-tile BlockLoad takes its async dep from a token iter_arg
// (cross-iteration RAW: the next iteration must wait for the previous store to
// the same tile).  The legal set is therefore {ST_LAST} only, and dry-run emits
// hw_dep_type = "LD_DEP_ST_LAST_TASK" on the task's load and store.
//
// The input below is the post-`schedule-tasks` IR for a tiled GEMM inner tk
// loop (see schedule_gemm_tiled.mlir for the pre-schedule source).
//
// RUN: cgra-opt %s --llm-pipeline-schedule --llm-pipeline-schedule-dry-run 2>/dev/null | FileCheck %s

module attributes {adora.scheduled} {
  func.func @gemm_tiled(%arg0: memref<64x64xf32>, %arg1: memref<64x64xf32>, %arg2: memref<64x64xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = [{dst = 4 : i64, kind = "RAW", overlap = true, src = 3 : i64}, {dst = 5 : i64, kind = "WAR", overlap = true, src = 0 : i64}]}], adora.lc_dep_summary = [{edges = [{exact = true, kind = "LC-RAW", step = 1 : i64}], loop_idx = 0 : i64, loop_op = "affine.for"}]} {
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 4 {
      affine.for %arg4 = 0 to 4 {
        %0 = ADORA.event.create -> !ADORA.token
        %1 = ADORA.event.create -> !ADORA.token
        %2 = ADORA.event.create -> !ADORA.token
        // The tk loop carries !ADORA.token iter_args -> loop-carried task.
        // CHECK: affine.for {{.*}} iter_args({{.*}}) -> (!ADORA.token
        %3:3 = affine.for %arg5 = 0 to 4 iter_args(%arg6 = %0, %arg7 = %1, %arg8 = %2) -> (!ADORA.token, !ADORA.token, !ADORA.token) {
          // C-tile load: async dep [%arg7] is a token iter_arg = cross-iteration RAW.
          %result, %asyncToken = ADORA.BlockLoad async [%arg7] %arg2 [%arg3 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "0", KernelName = "gemm_tiled_tk"}
          %result_0, %asyncToken_1 = ADORA.BlockLoad async [] %arg0 [%arg3 * 16, %arg5 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "1", KernelName = "gemm_tiled_tk"}
          // The task's collected load (last one) gets the serial dep_type.
          // CHECK: ADORA.BlockLoad async {{.*}} %arg1 {{.*}}hw_dep_type = "LD_DEP_ST_LAST_TASK"
          %result_2, %asyncToken_3 = ADORA.BlockLoad async [] %arg1 [%arg5 * 16, %arg4 * 16] : memref<64x64xf32> -> memref<16x16xf32>  {Id = "2", KernelName = "gemm_tiled_tk"}
          %4 = ADORA.LocalMemAlloc memref<16x16xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          %5 = ADORA.kernel async [%asyncToken_3, %asyncToken_1, %asyncToken] {
            ADORA.terminator
          } {KernelName = "gemm_tiled_tk"}
          // Cross-iteration RAW -> store also serial.
          // CHECK: ADORA.BlockStore async {{.*}}hw_dep_type = "LD_DEP_ST_LAST_TASK"
          %6 = ADORA.BlockStore async [%5, %asyncToken, %arg6, %arg8] %4, %arg2 [%arg3 * 16, %arg4 * 16] : memref<16x16xf32> -> memref<64x64xf32>  {Id = "3", KernelName = "gemm_tiled_tk"}
          affine.yield %asyncToken, %6, %6 : !ADORA.token, !ADORA.token, !ADORA.token
        }
      }
    }
    return
  }
}

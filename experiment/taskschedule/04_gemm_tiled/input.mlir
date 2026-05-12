// Tiled GEMM schedule test — 64×64×64, tile size 16×16×16.
//
// Matrix: C[M×N] += A[M×K] × B[K×N]   (M=N=K=64, tile=16 → 4 tiles/dim)
//
// Loop structure (after tiling):
//   affine.for %ti = 0 to 4 {              // tile-i (M)
//     affine.for %tj = 0 to 4 {            // tile-j (N)
//       affine.for %tk = 0 to 4 {          // tile-k (K)  ← reduction tile loop
//         BlockLoad  C_tile  (%C, [ti*16, tj*16])  <- LOOP-CARRIED RAW: prev tk wrote here
//         BlockLoad  A_tile  (%A, [ti*16, tk*16])
//         BlockLoad  B_tile  (%B, [tk*16, tj*16])
//         LocalMemAlloc C_local
//         kernel gemm_tiled_tk
//         BlockStore C_local -> %C [ti*16, tj*16]  <- LOOP-CARRIED WRITE
//       }
//     }
//   }
//
// Intra-iteration dependency (already supported):
//   BlockStore(C_local→%C) is the last op in the tk body.
//   The NEXT iteration's BlockLoad(%C) is a loop-carried RAW dep on the same
//   tile.  Within a SINGLE iteration, the token chain is:
//     (no predecessor) → kernel → BlockStore → token
//
// Loop-carried token yield (NOT YET IMPLEMENTED — future PR6):
//   The token produced by BlockStore in iteration tk must be passed to the
//   BlockLoad in iteration tk+1 via scf.for iter_args or an equivalent
//   mechanism.  Until that is implemented, the test only verifies:
//   1. The intra-iteration BlockStore produces a token.
//   2. The BlockLoad of A and B (no RAW pred) has no async dep.
//
// RUN: %cgra-opt %s --adora-schedule-tasks="emit-token=true" 2>/dev/null \
// RUN:   | FileCheck %s

// --- What the pass currently does (intra-iteration WAR dep) ---
//
// analyzeDependencyInGraph detects a WAR edge:
//   BlockLoad(%arg2, C-tile)  →  BlockStore(%arg2, C-tile)
// (same memref, overlapping region → WAR).
// threadTokensOnDMAs wires it as:
//   %result, %asyncToken = ADORA.BlockLoad %arg2 ... -> !ADORA.token
//   ADORA.BlockStore async [%asyncToken] ... %arg2 ...
//
// This is the INTRA-iteration fence: within one (ti,tj,tk) body the store
// waits for the load to finish before overwriting the C tile.

// C-tile load acquires a token (WAR producer).
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg2 {{.*}} -> !ADORA.token

// A-tile and B-tile loads have no dep predecessor → plain (no async).
// CHECK: ADORA.BlockLoad %arg0
// CHECK: ADORA.BlockLoad %arg1

// BlockStore consumes the C-load token (WAR consumer).
// CHECK: ADORA.BlockStore async [%{{.*}}] %{{.*}}, %arg2

// --- PR6.1: loop-carried dep analysis result (adora.lc_dep_summary) ---
//
// The tk-loop is a reduction loop: every iteration reads and writes the same
// C-tile at [ti*16, tj*16]. analyzeLoopCarriedDeps detects:
//   - LC-RAW  (BlockStore in iter k → BlockLoad in iter k+1, same C-tile)
//   - LC-WAW  (BlockStore in iter k → BlockStore in iter k+1, same C-tile)
//
// CHECK: "adora.lc_dep_summary"
// CHECK-SAME: LC-RAW
// CHECK-SAME: LC-WAW

// TODO(PR6.3 loop-carried token yield): once iter_args token yield is
// implemented, also verify:
//   scf.for {{.*}} iter_args(%lc_tok = {{.*}}) -> !ADORA.token
//   ADORA.BlockLoad async [%lc_tok] %arg2 ...
//   scf.yield %{{.*}} : !ADORA.token

module {
  func.func @gemm_tiled(%arg0: memref<64x64xf32>,   // A
                         %arg1: memref<64x64xf32>,   // B
                         %arg2: memref<64x64xf32>) { // C  (read-accumulate-write)
    %cst = arith.constant 0.000000e+00 : f32

    // Tile loops: ti (M), tj (N), tk (K-reduction)
    affine.for %ti = 0 to 4 {
      affine.for %tj = 0 to 4 {
        affine.for %tk = 0 to 4 {

          // --- Load C tile (accumulator) ---
          // This is the loop-carried read: previous tk iteration stored to the
          // same %arg2[ti*16..ti*16+16, tj*16..tj*16+16] region.
          %c_tile = ADORA.BlockLoad %arg2 [%ti * 16, %tj * 16]
              : memref<64x64xf32> -> memref<16x16xf32>
              {Id = "0", KernelName = "gemm_tiled_tk"}

          // --- Load A tile ---
          %a_tile = ADORA.BlockLoad %arg0 [%ti * 16, %tk * 16]
              : memref<64x64xf32> -> memref<16x16xf32>
              {Id = "1", KernelName = "gemm_tiled_tk"}

          // --- Load B tile ---
          %b_tile = ADORA.BlockLoad %arg1 [%tk * 16, %tj * 16]
              : memref<64x64xf32> -> memref<16x16xf32>
              {Id = "2", KernelName = "gemm_tiled_tk"}

          // --- Output scratchpad ---
          %c_local = ADORA.LocalMemAlloc memref<16x16xf32>
              {Id = "3", KernelName = "gemm_tiled_tk"}

          // --- Compute: C_local = C_tile + A_tile × B_tile ---
          ADORA.kernel {
            affine.for %i = 0 to 16 {
              affine.for %j = 0 to 16 {
                // Load accumulator value from C tile
                %c_val = affine.load %c_tile[%i, %j] : memref<16x16xf32>
                // Inner product over k-dimension of this tile
                %result = affine.for %kk = 0 to 16 iter_args(%acc = %c_val) -> (f32) {
                  %a_val = affine.load %a_tile[%i, %kk] : memref<16x16xf32>
                  %b_val = affine.load %b_tile[%kk, %j] : memref<16x16xf32>
                  %prod  = arith.mulf %a_val, %b_val : f32
                  %sum   = arith.addf %acc, %prod : f32
                  affine.yield %sum : f32
                }
                affine.store %result, %c_local[%i, %j] : memref<16x16xf32>
              }
            }
            ADORA.terminator
          } {KernelName = "gemm_tiled_tk"}

          // --- Write updated C tile back to DRAM ---
          // This BlockStore is the loop-carried WRITE: the next tk iteration
          // must wait for this store to complete before loading the same tile.
          ADORA.BlockStore %c_local, %arg2 [%ti * 16, %tj * 16]
              : memref<16x16xf32> -> memref<64x64xf32>
              {Id = "3", KernelName = "gemm_tiled_tk"}

        }  // affine.for %tk
      }  // affine.for %tj
    }  // affine.for %ti

    return
  }
}

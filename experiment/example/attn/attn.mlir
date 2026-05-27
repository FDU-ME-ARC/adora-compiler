// Scaled Dot-Product Attention kernel (integer approximation)
//
// Computes a simplified attention:
//   S = Q * K^T          (8x8 score matrix)
//   S = S >> 2           (integer scale approximation)
//   O = ReLU(S) * V      (8x16 output)
//
// Q, K shape: 8x16  (8 tokens, head_dim=16)
// V     shape: 8x16
// Output      shape: 8x16
//
// Run with adoracc:
//   adoracc.py attn.mlir --work-dir /tmp/attn_out -o /tmp/attn_out/result.mlir

module attributes {} {
  func.func @attention(%arg0: memref<?x16xi32>, %arg1: memref<?x16xi32>,
                       %arg2: memref<?x16xi32>, %arg3: memref<?x16xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32   = arith.constant 2 : i32
    %c0_i32   = arith.constant 0 : i32
    %score    = memref.alloca() : memref<8x8xi32>
    %acc      = memref.alloca() : memref<i32>
    %undef    = llvm.mlir.undef : i32

    // --- Phase 1: S = Q * K^T ---
    affine.store %undef, %acc[] : memref<i32>
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        affine.store %c0_i32, %acc[] : memref<i32>
        affine.for %k = 0 to 16 {
          %q  = affine.load %arg0[%i, %k] : memref<?x16xi32>
          %kt = affine.load %arg1[%j, %k] : memref<?x16xi32>
          %p  = arith.muli %q, %kt        : i32
          %s  = affine.load %acc[]        : memref<i32>
          %ns = arith.addi %s, %p         : i32
          affine.store %ns, %acc[]        : memref<i32>
        }
        %v = affine.load %acc[]           : memref<i32>
        affine.store %v, %score[%i, %j]  : memref<8x8xi32>
      }
    }

    // --- Phase 2: S = S >> 2  (integer scale) ---
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 8 {
        %s  = affine.load %score[%i, %j]  : memref<8x8xi32>
        %ss = arith.shrsi %s, %c2_i32     : i32
        affine.store %ss, %score[%i, %j]  : memref<8x8xi32>
      }
    }

    // --- Phase 3: O = ReLU(S) * V ---
    %acc2 = memref.alloca() : memref<i32>
    affine.store %undef, %acc2[] : memref<i32>
    affine.for %i = 0 to 8 {
      affine.for %j = 0 to 16 {
        affine.store %c0_i32, %acc2[] : memref<i32>
        affine.for %k = 0 to 8 {
          %s    = affine.load %score[%i, %k]  : memref<8x8xi32>
          %pos  = arith.cmpi sgt, %s, %c0_i32 : i32
          %relu = arith.select %pos, %s, %c0_i32 : i32
          %v    = affine.load %arg2[%k, %j]   : memref<?x16xi32>
          %p    = arith.muli %relu, %v         : i32
          %acc_v = affine.load %acc2[]         : memref<i32>
          %nacc  = arith.addi %acc_v, %p       : i32
          affine.store %nacc, %acc2[]          : memref<i32>
        }
        %out = affine.load %acc2[]           : memref<i32>
        affine.store %out, %arg3[%i, %j]     : memref<?x16xi32>
      }
    }
    return
  }
}

// Matrix-Vector Transpose (MVT) kernel — PolyBench benchmark
//
// Computes:
//   x1 = x1 + A  * x2
//   x2 = x2 + A' * x1
//
// Run with adoracc:
//   adoracc.py mvt.mlir --work-dir /tmp/mvt_out -o /tmp/mvt_out/result.mlir

module attributes {} {
  func.func @kernel_mvt(%arg0: memref<?xf32>, %arg1: memref<?xf32>,
                        %arg2: memref<?xf32>, %arg3: memref<?xf32>,
                        %arg4: memref<?x40xf32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    affine.for %i = 0 to 40 {
      affine.for %j = 0 to 40 {
        %0 = affine.load %arg0[%i]       : memref<?xf32>
        %1 = affine.load %arg4[%i, %j]   : memref<?x40xf32>
        %2 = affine.load %arg2[%j]       : memref<?xf32>
        %3 = arith.mulf %1, %2           : f32
        %4 = arith.addf %0, %3           : f32
        affine.store %4, %arg0[%i]       : memref<?xf32>
      }
    }
    affine.for %i = 0 to 40 {
      affine.for %j = 0 to 40 {
        %0 = affine.load %arg1[%i]       : memref<?xf32>
        %1 = affine.load %arg4[%j, %i]   : memref<?x40xf32>
        %2 = affine.load %arg3[%j]       : memref<?xf32>
        %3 = arith.mulf %1, %2           : f32
        %4 = arith.addf %0, %3           : f32
        affine.store %4, %arg1[%i]       : memref<?xf32>
      }
    }
    return
  }
}

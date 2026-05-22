#map = affine_map<(d0) -> (d0)>
module attributes {adora.scheduled} {
  func.func @cholesky(%arg0: memref<120x120xf32>) attributes {adora.dep_summary = [{block_idx = 0 : i64, edges = []}, {block_idx = 1 : i64, edges = []}], llvm.linkage = #llvm.linkage<external>} {
    affine.for %arg1 = 0 to 120 {
      affine.for %arg2 = 0 to #map(%arg1) {
        ADORA.kernel {
          affine.for %arg3 = 0 to #map(%arg2) {
            %5 = affine.load %arg0[%arg1, %arg3] : memref<120x120xf32>
            %6 = affine.load %arg0[%arg2, %arg3] : memref<120x120xf32>
            %7 = arith.mulf %5, %6 : f32
            %8 = affine.load %arg0[%arg1, %arg2] : memref<120x120xf32>
            %9 = arith.subf %8, %7 : f32
            affine.store %9, %arg0[%arg1, %arg2] : memref<120x120xf32>
          }
          ADORA.terminator
        }
        %2 = affine.load %arg0[%arg2, %arg2] : memref<120x120xf32>
        %3 = affine.load %arg0[%arg1, %arg2] : memref<120x120xf32>
        %4 = arith.divf %3, %2 : f32
        affine.store %4, %arg0[%arg1, %arg2] : memref<120x120xf32>
      }
      ADORA.kernel {
        affine.for %arg2 = 0 to #map(%arg1) {
          %2 = affine.load %arg0[%arg1, %arg2] : memref<120x120xf32>
          %3 = arith.mulf %2, %2 : f32
          %4 = affine.load %arg0[%arg1, %arg1] : memref<120x120xf32>
          %5 = arith.subf %4, %3 : f32
          affine.store %5, %arg0[%arg1, %arg1] : memref<120x120xf32>
        }
        ADORA.terminator
      }
      %0 = affine.load %arg0[%arg1, %arg1] : memref<120x120xf32>
      %1 = math.sqrt %0 : f32
      affine.store %1, %arg0[%arg1, %arg1] : memref<120x120xf32>
    }
    return
  }
}


#map = affine_map<(d0) -> (d0 + 1)>
module {
  func.func @jacobi_2d(%arg0: memref<?x30xi32>, %arg1: memref<?x30xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    affine.for %arg2 = 0 to 10 {
      affine.for %arg3 = 0 to 28 {
        %0 = affine.apply #map(%arg3)
        affine.for %arg4 = 0 to 28 {
          %1 = affine.apply #map(%arg4)
          %2 = affine.load %arg0[%0, %1] : memref<?x30xi32>
          %3 = affine.load %arg0[%0, %1 - 1] : memref<?x30xi32>
          %4 = arith.addi %2, %3 : i32
          %5 = affine.load %arg0[%0, %1 + 1] : memref<?x30xi32>
          %6 = arith.addi %4, %5 : i32
          %7 = affine.load %arg0[%0 + 1, %1] : memref<?x30xi32>
          %8 = arith.addi %6, %7 : i32
          %9 = affine.load %arg0[%0 - 1, %1] : memref<?x30xi32>
          %10 = arith.addi %8, %9 : i32
          %11 = arith.divsi %10, %c5_i32 : i32
          affine.store %11, %arg1[%0, %1] : memref<?x30xi32>
        }
      }
      affine.for %arg3 = 0 to 28 {
        %0 = affine.apply #map(%arg3)
        affine.for %arg4 = 0 to 28 {
          %1 = affine.apply #map(%arg4)
          %2 = affine.load %arg1[%0, %1] : memref<?x30xi32>
          %3 = affine.load %arg1[%0, %1 - 1] : memref<?x30xi32>
          %4 = arith.addi %2, %3 : i32
          %5 = affine.load %arg1[%0, %1 + 1] : memref<?x30xi32>
          %6 = arith.addi %4, %5 : i32
          %7 = affine.load %arg1[%0 + 1, %1] : memref<?x30xi32>
          %8 = arith.addi %6, %7 : i32
          %9 = affine.load %arg1[%0 - 1, %1] : memref<?x30xi32>
          %10 = arith.addi %8, %9 : i32
          %11 = arith.divsi %10, %c5_i32 : i32
          affine.store %11, %arg0[%0, %1] : memref<?x30xi32>
        }
      }
    }
    return
  }
}


module {
  func.func @jacobi_2d(%arg0: memref<?x30xi32>, %arg1: memref<?x30xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c5_i32 = arith.constant 5 : i32
    affine.for %arg2 = 0 to 10 {
      ADORA.kernel {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %0 = affine.load %arg0[%arg3 + 1, %arg4 + 1] : memref<?x30xi32>
            %1 = affine.load %arg0[%arg3 + 1, %arg4] : memref<?x30xi32>
            %2 = arith.addi %0, %1 : i32
            %3 = affine.load %arg0[%arg3 + 1, %arg4 + 2] : memref<?x30xi32>
            %4 = arith.addi %2, %3 : i32
            %5 = affine.load %arg0[%arg3 + 2, %arg4 + 1] : memref<?x30xi32>
            %6 = arith.addi %4, %5 : i32
            %7 = affine.load %arg0[%arg3, %arg4 + 1] : memref<?x30xi32>
            %8 = arith.addi %6, %7 : i32
            %9 = arith.divsi %8, %c5_i32 : i32
            affine.store %9, %arg1[%arg3 + 1, %arg4 + 1] : memref<?x30xi32>
          }
        }
        ADORA.terminator
      } {KernelName = "jacobi_2d_0"}
      ADORA.kernel {
        affine.for %arg3 = 0 to 28 {
          affine.for %arg4 = 0 to 28 {
            %0 = affine.load %arg1[%arg3 + 1, %arg4 + 1] : memref<?x30xi32>
            %1 = affine.load %arg1[%arg3 + 1, %arg4] : memref<?x30xi32>
            %2 = arith.addi %0, %1 : i32
            %3 = affine.load %arg1[%arg3 + 1, %arg4 + 2] : memref<?x30xi32>
            %4 = arith.addi %2, %3 : i32
            %5 = affine.load %arg1[%arg3 + 2, %arg4 + 1] : memref<?x30xi32>
            %6 = arith.addi %4, %5 : i32
            %7 = affine.load %arg1[%arg3, %arg4 + 1] : memref<?x30xi32>
            %8 = arith.addi %6, %7 : i32
            %9 = arith.divsi %8, %c5_i32 : i32
            affine.store %9, %arg0[%arg3 + 1, %arg4 + 1] : memref<?x30xi32>
          }
        }
        ADORA.terminator
      } {KernelName = "jacobi_2d_1"}
    }
    return
  }
}


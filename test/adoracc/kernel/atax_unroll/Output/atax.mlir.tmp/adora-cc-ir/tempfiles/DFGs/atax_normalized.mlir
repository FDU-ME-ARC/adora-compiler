module {
  func.func @atax(%arg0: memref<?x24xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 24 {
      affine.store %c0_i32, %arg2[%arg4] : memref<?xi32>
    }
    affine.for %arg4 = 0 to 24 {
      affine.store %c0_i32, %arg3[%arg4] : memref<?xi32>
      affine.for %arg5 = 0 to 24 {
        %0 = affine.load %arg3[%arg4] : memref<?xi32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<?x24xi32>
        %2 = affine.load %arg1[%arg5] : memref<?xi32>
        %3 = arith.muli %1, %2 : i32
        %4 = arith.addi %0, %3 : i32
        affine.store %4, %arg3[%arg4] : memref<?xi32>
      }
      affine.for %arg5 = 0 to 24 {
        %0 = affine.load %arg2[%arg5] : memref<?xi32>
        %1 = affine.load %arg0[%arg4, %arg5] : memref<?x24xi32>
        %2 = affine.load %arg3[%arg4] : memref<?xi32>
        %3 = arith.muli %1, %2 : i32
        %4 = arith.addi %0, %3 : i32
        affine.store %4, %arg2[%arg5] : memref<?xi32>
      }
    }
    return
  }
}


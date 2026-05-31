module {
  func.func @doitgenTriple(%arg0: memref<16xi32>, %arg1: memref<16xi32>, %arg2: memref<256xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg3 = 0 to 16 {
      %0 = affine.for %arg4 = 0 to 16 iter_args(%arg5 = %c0_i32) -> (i32) {
        %1 = affine.load %arg0[%arg4] : memref<16xi32>
        %2 = affine.load %arg2[%arg4 + %arg3 * 16] : memref<256xi32>
        %3 = arith.cmpi sgt, %1, %c0_i32 : i32
        %4 = scf.if %3 -> (i32) {
          %5 = arith.muli %1, %2 : i32
          %6 = arith.addi %5, %2 : i32
          %7 = arith.muli %6, %1 : i32
          %8 = arith.addi %arg5, %7 : i32
          scf.yield %8 : i32
        } else {
          scf.yield %arg5 : i32
        }
        affine.yield %4 : i32
      }
      affine.store %0, %arg1[%arg3] : memref<16xi32>
    }
    return
  }
}
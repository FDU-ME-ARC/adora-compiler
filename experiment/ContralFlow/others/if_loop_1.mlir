func.func @if_loop_1(%arg0: memref<200xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %cst = arith.constant 2 : i32
    %cst_0 = arith.constant 10 : i32
    %cst_1 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 200 iter_args(%arg2 = %cst_1) -> (i32) {
      %1 = affine.load %arg0[%arg1] : memref<200xi32>
      %2 = arith.muli %1, %cst : i32
      %3 = arith.cmpi ult, %cst_0, %2: i32
      %4 = scf.if %3 -> (i32) {
        %5 = arith.addi %2, %arg2 : i32
        scf.yield %5 : i32
      } else {
        scf.yield %arg2 : i32
      }
      affine.yield %4 : i32
      // %1 = affine.store %arg0[%arg1] : memref<1xi32>
    }
    return %0 : i32
  }

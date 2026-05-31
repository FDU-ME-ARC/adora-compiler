#map = affine_map<(d0) -> (d0 + 1)>
#map1 = affine_map<(d0) -> (d0 + 2)>
#map2 = affine_map<(d0) -> (d0 + 3)>
module {
  func.func @if_loop_1(%arg0: memref<200xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c10_i32 = arith.constant 10 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = affine.for %arg1 = 0 to 200 step 4 iter_args(%arg2 = %c0_i32) -> (i32) {
      %1 = affine.load %arg0[%arg1] : memref<200xi32>
      %2 = arith.muli %1, %c2_i32 : i32
      %3 = arith.cmpi ult, %c10_i32, %2 : i32
      %4 = scf.if %3 -> (i32) {
        %20 = arith.addi %2, %arg2 : i32
        scf.yield %20 : i32
      } else {
        scf.yield %arg2 : i32
      }
      %5 = affine.apply #map(%arg1)
      %6 = affine.load %arg0[%5] : memref<200xi32>
      %7 = arith.muli %6, %c2_i32 : i32
      %8 = arith.cmpi ult, %c10_i32, %7 : i32
      %9 = scf.if %8 -> (i32) {
        %20 = arith.addi %7, %4 : i32
        scf.yield %20 : i32
      } else {
        scf.yield %4 : i32
      }
      %10 = affine.apply #map1(%arg1)
      %11 = affine.load %arg0[%10] : memref<200xi32>
      %12 = arith.muli %11, %c2_i32 : i32
      %13 = arith.cmpi ult, %c10_i32, %12 : i32
      %14 = scf.if %13 -> (i32) {
        %20 = arith.addi %12, %9 : i32
        scf.yield %20 : i32
      } else {
        scf.yield %9 : i32
      }
      %15 = affine.apply #map2(%arg1)
      %16 = affine.load %arg0[%15] : memref<200xi32>
      %17 = arith.muli %16, %c2_i32 : i32
      %18 = arith.cmpi ult, %c10_i32, %17 : i32
      %19 = scf.if %18 -> (i32) {
        %20 = arith.addi %17, %14 : i32
        scf.yield %20 : i32
      } else {
        scf.yield %14 : i32
      }
      affine.yield %19 : i32
    }
    return %0 : i32
  }
}
module {
  func.func @getTanh(%arg0: memref<100xi32>) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c3_i32 = arith.constant 3 : i32
    %c19_i32 = arith.constant 19 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<i32>
    %0 = ADORA.BlockLoad %arg0 [0] : memref<100xi32> -> memref<100xi32>  {Id = "0", KernelName = ""}
    %1 = ADORA.LocalMemAlloc memref<2xi32>  {Id = "1", KernelName = ""}
    ADORA.kernel {
      %3 = affine.for %arg1 = 0 to 100 iter_args(%arg2 = %c0_i32) -> (i32) {
        %4 = affine.load %0[%arg1] : memref<100xi32>
        %5 = arith.cmpi ult, %4, %c1_i32 : i32
        %6 = scf.if %5 -> (i32) {
          %8 = arith.muli %4, %4 : i32
          %9 = arith.addi %8, %c19_i32 : i32
          %10 = arith.muli %9, %4 : i32
          %11 = arith.muli %10, %4 : i32
          %12 = arith.addi %11, %c3_i32 : i32
          %13 = arith.muli %12, %4 : i32
          scf.yield %13 : i32
        } else {
          scf.yield %c1_i32 : i32
        }
        %7 = arith.addi %arg2, %6 : i32
        affine.store %7, %1[0] : memref<2xi32>
        affine.yield %7 : i32
      }
      ADORA.terminator
    }
    ADORA.BlockStore %1, %alloca [] : memref<2xi32> -> memref<i32>  {Id = "1", KernelName = ""}
    %2 = affine.load %alloca[] : memref<i32>
    return %2 : i32
  }
}


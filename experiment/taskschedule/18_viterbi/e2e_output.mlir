module attributes {adora.scheduled} {
  memref.global "private" @trans : memref<4x4xi32> = dense<[[-1, -3, -5, -7], [-3, -1, -3, -5], [-5, -3, -1, -3], [-7, -5, -3, -1]]>
  memref.global "private" @emit : memref<4x4xi32> = dense<[[0, -2, -4, -6], [-2, 0, -2, -4], [-4, -2, 0, -2], [-6, -4, -2, 0]]>
  func.func @viterbi(%arg0: memref<?xi32>, %arg1: memref<?xi32>) -> i32 attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %c6 = arith.constant 6 : index
    %c-9999_i32 = arith.constant -9999 : i32
    %c1_i32 = arith.constant 1 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<8x4xi32>
    %alloca_0 = memref.alloca() : memref<8x4xi32>
    %0 = memref.get_global @emit : memref<4x4xi32>
    %1 = affine.load %arg0[0] : memref<?xi32>
    %2 = arith.index_cast %1 : i32 to index
    scf.for %arg2 = %c0 to %c4 step %c1 {
      %7 = memref.load %0[%arg2, %2] : memref<4x4xi32>
      memref.store %7, %alloca_0[%c0, %arg2] : memref<8x4xi32>
      memref.store %c-1_i32, %alloca[%c0, %arg2] : memref<8x4xi32>
    }
    %3 = memref.get_global @trans : memref<4x4xi32>
    scf.for %arg2 = %c1 to %c8 step %c1 {
      %7 = arith.index_cast %arg2 : index to i32
      %8 = arith.addi %7, %c-1_i32 : i32
      %9 = arith.index_cast %8 : i32 to index
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %10 = memref.load %arg0[%arg2] : memref<?xi32>
        %11 = arith.index_cast %10 : i32 to index
        %12 = memref.load %0[%arg3, %11] : memref<4x4xi32>
        %13:2 = scf.for %arg4 = %c0 to %c4 step %c1 iter_args(%arg5 = %c0_i32, %arg6 = %c-9999_i32) -> (i32, i32) {
          %14 = arith.index_cast %arg4 : index to i32
          %15 = memref.load %alloca_0[%9, %arg4] : memref<8x4xi32>
          %16 = memref.load %3[%arg4, %arg3] : memref<4x4xi32>
          %17 = arith.addi %15, %16 : i32
          %18 = arith.addi %17, %12 : i32
          %19 = arith.cmpi sgt, %18, %arg6 : i32
          %20 = arith.select %19, %14, %arg5 : i32
          %21 = arith.select %19, %18, %arg6 : i32
          scf.yield %20, %21 : i32, i32
        }
        memref.store %13#1, %alloca_0[%arg2, %arg3] : memref<8x4xi32>
        memref.store %13#0, %alloca[%arg2, %arg3] : memref<8x4xi32>
      }
    }
    %4 = scf.for %arg2 = %c1 to %c4 step %c1 iter_args(%arg3 = %c0_i32) -> (i32) {
      %7 = arith.index_cast %arg2 : index to i32
      %8 = memref.load %alloca_0[%c7, %arg2] : memref<8x4xi32>
      %9 = arith.index_cast %arg3 : i32 to index
      %10 = memref.load %alloca_0[%c7, %9] : memref<8x4xi32>
      %11 = arith.cmpi sgt, %8, %10 : i32
      %12 = arith.select %11, %7, %arg3 : i32
      scf.yield %12 : i32
    }
    affine.store %4, %arg1[7] : memref<?xi32>
    scf.for %arg2 = %c0 to %c7 step %c1 {
      %7 = arith.subi %c6, %arg2 : index
      %8 = arith.index_cast %7 : index to i32
      %9 = arith.addi %8, %c1_i32 : i32
      %10 = arith.index_cast %9 : i32 to index
      %11 = memref.load %arg1[%10] : memref<?xi32>
      %12 = arith.index_cast %11 : i32 to index
      %13 = memref.load %alloca[%10, %12] : memref<8x4xi32>
      memref.store %13, %arg1[%7] : memref<?xi32>
    }
    %5 = arith.index_cast %4 : i32 to index
    %6 = affine.load %alloca_0[7, symbol(%5)] : memref<8x4xi32>
    return %6 : i32
  }
}


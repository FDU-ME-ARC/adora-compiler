module attributes {adora.scheduled} {
  func.func @sobel(%arg0: memref<?x8xi32>, %arg1: memref<?x8xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    %c2_i32 = arith.constant 2 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %alloca = memref.alloca() : memref<8x8xi32>
    %alloca_0 = memref.alloca() : memref<8x8xi32>
    scf.for %arg2 = %c0 to %c8 step %c1 {
      scf.for %arg3 = %c0 to %c8 step %c1 {
        memref.store %c0_i32, %alloca_0[%arg2, %arg3] : memref<8x8xi32>
        memref.store %c0_i32, %alloca[%arg2, %arg3] : memref<8x8xi32>
      }
    }
    scf.for %arg2 = %c1 to %c7 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.addi %0, %c-1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.addi %0, %c1_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      scf.for %arg3 = %c1 to %c7 step %c1 {
        %5 = arith.index_cast %arg3 : index to i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%2, %7] : memref<?x8xi32>
        %9 = arith.addi %5, %c1_i32 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = memref.load %arg0[%2, %10] : memref<?x8xi32>
        %12 = arith.subi %11, %8 : i32
        %13 = memref.load %arg0[%arg2, %7] : memref<?x8xi32>
        %14 = arith.muli %13, %c2_i32 : i32
        %15 = arith.subi %12, %14 : i32
        %16 = memref.load %arg0[%arg2, %10] : memref<?x8xi32>
        %17 = arith.muli %16, %c2_i32 : i32
        %18 = arith.addi %15, %17 : i32
        %19 = memref.load %arg0[%4, %7] : memref<?x8xi32>
        %20 = arith.subi %18, %19 : i32
        %21 = memref.load %arg0[%4, %10] : memref<?x8xi32>
        %22 = arith.addi %20, %21 : i32
        memref.store %22, %alloca_0[%arg2, %arg3] : memref<8x8xi32>
      }
    }
    scf.for %arg2 = %c1 to %c7 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.addi %0, %c-1_i32 : i32
      %2 = arith.index_cast %1 : i32 to index
      %3 = arith.addi %0, %c1_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      scf.for %arg3 = %c1 to %c7 step %c1 {
        %5 = arith.index_cast %arg3 : index to i32
        %6 = arith.addi %5, %c-1_i32 : i32
        %7 = arith.index_cast %6 : i32 to index
        %8 = memref.load %arg0[%2, %7] : memref<?x8xi32>
        %9 = arith.muli %8, %c-1_i32 : i32
        %10 = memref.load %arg0[%2, %arg3] : memref<?x8xi32>
        %11 = arith.muli %10, %c2_i32 : i32
        %12 = arith.subi %9, %11 : i32
        %13 = arith.addi %5, %c1_i32 : i32
        %14 = arith.index_cast %13 : i32 to index
        %15 = memref.load %arg0[%2, %14] : memref<?x8xi32>
        %16 = arith.subi %12, %15 : i32
        %17 = memref.load %arg0[%4, %7] : memref<?x8xi32>
        %18 = arith.addi %16, %17 : i32
        %19 = memref.load %arg0[%4, %arg3] : memref<?x8xi32>
        %20 = arith.muli %19, %c2_i32 : i32
        %21 = arith.addi %18, %20 : i32
        %22 = memref.load %arg0[%4, %14] : memref<?x8xi32>
        %23 = arith.addi %21, %22 : i32
        memref.store %23, %alloca[%arg2, %arg3] : memref<8x8xi32>
      }
    }
    scf.for %arg2 = %c1 to %c7 step %c1 {
      scf.for %arg3 = %c1 to %c7 step %c1 {
        %0 = memref.load %alloca_0[%arg2, %arg3] : memref<8x8xi32>
        %1 = arith.cmpi slt, %0, %c0_i32 : i32
        %2 = scf.if %1 -> (i32) {
          %7 = arith.subi %c0_i32, %0 : i32
          scf.yield %7 : i32
        } else {
          scf.yield %0 : i32
        }
        %3 = memref.load %alloca[%arg2, %arg3] : memref<8x8xi32>
        %4 = arith.cmpi slt, %3, %c0_i32 : i32
        %5 = scf.if %4 -> (i32) {
          %7 = arith.subi %c0_i32, %3 : i32
          scf.yield %7 : i32
        } else {
          scf.yield %3 : i32
        }
        %6 = arith.addi %2, %5 : i32
        memref.store %6, %arg1[%arg2, %arg3] : memref<?x8xi32>
      }
    }
    return
  }
}


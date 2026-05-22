module attributes {adora.scheduled} {
  memref.global "private" @W_im : memref<8xi32> = dense<[0, -98, -181, -237, -256, -237, -181, -98]>
  memref.global "private" @W_re : memref<8xi32> = dense<[256, 237, 181, 98, 0, -98, -181, -237]>
  func.func @fft_stage(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8_i32 = arith.constant 8 : i32
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c16_i32 = arith.constant 16 : i32
    %0 = arith.addi %arg2, %c1_i32 : i32
    %1 = arith.shrsi %c16_i32, %0 : i32
    %2 = arith.shrsi %c16_i32, %arg2 : i32
    %3 = arith.index_cast %2 : i32 to index
    %4 = arith.index_cast %1 : i32 to index
    %5 = memref.get_global @W_re : memref<8xi32>
    %6 = memref.get_global @W_im : memref<8xi32>
    %7 = arith.divsi %c16_i32, %2 : i32
    %8 = arith.divsi %7, %c2_i32 : i32
    scf.for %arg3 = %c0 to %c16 step %3 {
      %9 = arith.divui %arg3, %3 : index
      %10 = arith.muli %9, %3 : index
      %11 = arith.index_cast %10 : index to i32
      scf.for %arg4 = %c0 to %4 step %c1 {
        %12 = arith.index_cast %arg4 : index to i32
        %13 = arith.addi %11, %12 : i32
        %14 = arith.addi %13, %1 : i32
        %15 = arith.muli %12, %8 : i32
        %16 = arith.index_cast %15 : i32 to index
        %17 = memref.load %5[%16] : memref<8xi32>
        %18 = memref.load %6[%16] : memref<8xi32>
        %19 = arith.index_cast %14 : i32 to index
        %20 = memref.load %arg0[%19] : memref<?xi32>
        %21 = memref.load %arg1[%19] : memref<?xi32>
        %22 = arith.muli %17, %20 : i32
        %23 = arith.muli %18, %21 : i32
        %24 = arith.subi %22, %23 : i32
        %25 = arith.shrsi %24, %c8_i32 : i32
        %26 = arith.muli %17, %21 : i32
        %27 = arith.muli %18, %20 : i32
        %28 = arith.addi %26, %27 : i32
        %29 = arith.shrsi %28, %c8_i32 : i32
        %30 = arith.index_cast %13 : i32 to index
        %31 = memref.load %arg0[%30] : memref<?xi32>
        %32 = arith.subi %31, %25 : i32
        memref.store %32, %arg0[%19] : memref<?xi32>
        %33 = memref.load %arg1[%30] : memref<?xi32>
        %34 = arith.subi %33, %29 : i32
        memref.store %34, %arg1[%19] : memref<?xi32>
        %35 = memref.load %arg0[%30] : memref<?xi32>
        %36 = arith.addi %35, %25 : i32
        memref.store %36, %arg0[%30] : memref<?xi32>
        %37 = memref.load %arg1[%30] : memref<?xi32>
        %38 = arith.addi %37, %29 : i32
        memref.store %38, %arg1[%30] : memref<?xi32>
      }
    }
    return
  }
  func.func @fft(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c2 = arith.constant 2 : index
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c4_i32 = arith.constant 4 : i32
    %c4 = arith.constant 4 : index
    %c8 = arith.constant 8 : index
    %c8_i32 = arith.constant 8 : i32
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %0 = memref.get_global @W_re : memref<8xi32>
    %1 = memref.get_global @W_im : memref<8xi32>
    scf.for %arg2 = %c0 to %c8 step %c1 {
      %2 = arith.index_cast %arg2 : index to i32
      %3 = arith.addi %2, %c8_i32 : i32
      %4 = affine.load %0[0] : memref<8xi32>
      %5 = affine.load %1[0] : memref<8xi32>
      %6 = arith.index_cast %3 : i32 to index
      %7 = memref.load %arg0[%6] : memref<?xi32>
      %8 = memref.load %arg1[%6] : memref<?xi32>
      %9 = arith.muli %4, %7 : i32
      %10 = arith.muli %5, %8 : i32
      %11 = arith.subi %9, %10 : i32
      %12 = arith.shrsi %11, %c8_i32 : i32
      %13 = arith.muli %4, %8 : i32
      %14 = arith.muli %5, %7 : i32
      %15 = arith.addi %13, %14 : i32
      %16 = arith.shrsi %15, %c8_i32 : i32
      %17 = memref.load %arg0[%arg2] : memref<?xi32>
      %18 = arith.subi %17, %12 : i32
      memref.store %18, %arg0[%6] : memref<?xi32>
      %19 = memref.load %arg1[%arg2] : memref<?xi32>
      %20 = arith.subi %19, %16 : i32
      memref.store %20, %arg1[%6] : memref<?xi32>
      %21 = memref.load %arg0[%arg2] : memref<?xi32>
      %22 = arith.addi %21, %12 : i32
      memref.store %22, %arg0[%arg2] : memref<?xi32>
      %23 = memref.load %arg1[%arg2] : memref<?xi32>
      %24 = arith.addi %23, %16 : i32
      memref.store %24, %arg1[%arg2] : memref<?xi32>
    }
    scf.for %arg2 = %c0 to %c16 step %c8 {
      %2 = arith.divui %arg2, %c8 : index
      %3 = arith.muli %2, %c8 : index
      %4 = arith.index_cast %3 : index to i32
      scf.for %arg3 = %c0 to %c4 step %c1 {
        %5 = arith.index_cast %arg3 : index to i32
        %6 = arith.addi %4, %5 : i32
        %7 = arith.addi %6, %c4_i32 : i32
        %8 = memref.load %0[%arg3] : memref<8xi32>
        %9 = memref.load %1[%arg3] : memref<8xi32>
        %10 = arith.index_cast %7 : i32 to index
        %11 = memref.load %arg0[%10] : memref<?xi32>
        %12 = memref.load %arg1[%10] : memref<?xi32>
        %13 = arith.muli %8, %11 : i32
        %14 = arith.muli %9, %12 : i32
        %15 = arith.subi %13, %14 : i32
        %16 = arith.shrsi %15, %c8_i32 : i32
        %17 = arith.muli %8, %12 : i32
        %18 = arith.muli %9, %11 : i32
        %19 = arith.addi %17, %18 : i32
        %20 = arith.shrsi %19, %c8_i32 : i32
        %21 = arith.index_cast %6 : i32 to index
        %22 = memref.load %arg0[%21] : memref<?xi32>
        %23 = arith.subi %22, %16 : i32
        memref.store %23, %arg0[%10] : memref<?xi32>
        %24 = memref.load %arg1[%21] : memref<?xi32>
        %25 = arith.subi %24, %20 : i32
        memref.store %25, %arg1[%10] : memref<?xi32>
        %26 = memref.load %arg0[%21] : memref<?xi32>
        %27 = arith.addi %26, %16 : i32
        memref.store %27, %arg0[%21] : memref<?xi32>
        %28 = memref.load %arg1[%21] : memref<?xi32>
        %29 = arith.addi %28, %20 : i32
        memref.store %29, %arg1[%21] : memref<?xi32>
      }
    }
    scf.for %arg2 = %c0 to %c16 step %c4 {
      %2 = arith.divui %arg2, %c4 : index
      %3 = arith.muli %2, %c4 : index
      %4 = arith.index_cast %3 : index to i32
      scf.for %arg3 = %c0 to %c2 step %c1 {
        %5 = arith.index_cast %arg3 : index to i32
        %6 = arith.addi %4, %5 : i32
        %7 = arith.addi %6, %c2_i32 : i32
        %8 = arith.muli %5, %c2_i32 : i32
        %9 = arith.index_cast %8 : i32 to index
        %10 = memref.load %0[%9] : memref<8xi32>
        %11 = memref.load %1[%9] : memref<8xi32>
        %12 = arith.index_cast %7 : i32 to index
        %13 = memref.load %arg0[%12] : memref<?xi32>
        %14 = memref.load %arg1[%12] : memref<?xi32>
        %15 = arith.muli %10, %13 : i32
        %16 = arith.muli %11, %14 : i32
        %17 = arith.subi %15, %16 : i32
        %18 = arith.shrsi %17, %c8_i32 : i32
        %19 = arith.muli %10, %14 : i32
        %20 = arith.muli %11, %13 : i32
        %21 = arith.addi %19, %20 : i32
        %22 = arith.shrsi %21, %c8_i32 : i32
        %23 = arith.index_cast %6 : i32 to index
        %24 = memref.load %arg0[%23] : memref<?xi32>
        %25 = arith.subi %24, %18 : i32
        memref.store %25, %arg0[%12] : memref<?xi32>
        %26 = memref.load %arg1[%23] : memref<?xi32>
        %27 = arith.subi %26, %22 : i32
        memref.store %27, %arg1[%12] : memref<?xi32>
        %28 = memref.load %arg0[%23] : memref<?xi32>
        %29 = arith.addi %28, %18 : i32
        memref.store %29, %arg0[%23] : memref<?xi32>
        %30 = memref.load %arg1[%23] : memref<?xi32>
        %31 = arith.addi %30, %22 : i32
        memref.store %31, %arg1[%23] : memref<?xi32>
      }
    }
    scf.for %arg2 = %c0 to %c16 step %c2 {
      %2 = arith.divui %arg2, %c2 : index
      %3 = arith.muli %2, %c2 : index
      %4 = arith.index_cast %3 : index to i32
      %5 = arith.addi %4, %c1_i32 : i32
      %6 = affine.load %0[0] : memref<8xi32>
      %7 = affine.load %1[0] : memref<8xi32>
      %8 = arith.index_cast %5 : i32 to index
      %9 = memref.load %arg0[%8] : memref<?xi32>
      %10 = memref.load %arg1[%8] : memref<?xi32>
      %11 = arith.muli %6, %9 : i32
      %12 = arith.muli %7, %10 : i32
      %13 = arith.subi %11, %12 : i32
      %14 = arith.shrsi %13, %c8_i32 : i32
      %15 = arith.muli %6, %10 : i32
      %16 = arith.muli %7, %9 : i32
      %17 = arith.addi %15, %16 : i32
      %18 = arith.shrsi %17, %c8_i32 : i32
      %19 = memref.load %arg0[%3] : memref<?xi32>
      %20 = arith.subi %19, %14 : i32
      memref.store %20, %arg0[%8] : memref<?xi32>
      %21 = memref.load %arg1[%3] : memref<?xi32>
      %22 = arith.subi %21, %18 : i32
      memref.store %22, %arg1[%8] : memref<?xi32>
      %23 = memref.load %arg0[%3] : memref<?xi32>
      %24 = arith.addi %23, %14 : i32
      memref.store %24, %arg0[%3] : memref<?xi32>
      %25 = memref.load %arg1[%3] : memref<?xi32>
      %26 = arith.addi %25, %18 : i32
      memref.store %26, %arg1[%3] : memref<?xi32>
    }
    return
  }
}


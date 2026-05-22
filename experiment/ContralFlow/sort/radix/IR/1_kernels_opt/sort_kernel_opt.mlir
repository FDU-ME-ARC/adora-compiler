module attributes {adora.scheduled} {
  func.func @local_scan(%arg0: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c-1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    scf.for %arg1 = %c0 to %c128 step %c1 {
      %0 = arith.index_cast %arg1 : index to i32
      %1 = arith.muli %0, %c16_i32 : i32
      scf.for %arg2 = %c1 to %c16 step %c1 {
        %2 = arith.index_cast %arg2 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = arith.addi %3, %c-1_i32 : i32
        %6 = arith.index_cast %5 : i32 to index
        %7 = memref.load %arg0[%6] : memref<?xi32>
        %8 = memref.load %arg0[%4] : memref<?xi32>
        %9 = arith.addi %8, %7 : i32
        memref.store %9, %arg0[%4] : memref<?xi32>
      }
    }
    return
  }
  func.func @sum_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c1 = arith.constant 1 : index
    %c-1_i32 = arith.constant -1 : i32
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    affine.store %c0_i32, %arg0[0] : memref<?xi32>
    scf.for %arg2 = %c1 to %c128 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c16_i32 : i32
      %2 = arith.addi %1, %c-1_i32 : i32
      %3 = arith.addi %0, %c-1_i32 : i32
      %4 = arith.index_cast %3 : i32 to index
      %5 = memref.load %arg0[%4] : memref<?xi32>
      %6 = arith.index_cast %2 : i32 to index
      %7 = memref.load %arg1[%6] : memref<?xi32>
      %8 = arith.addi %5, %7 : i32
      memref.store %8, %arg0[%arg2] : memref<?xi32>
    }
    return
  }
  func.func @last_step_scan(%arg0: memref<?xi32>, %arg1: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c128 = arith.constant 128 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c16_i32 = arith.constant 16 : i32
    scf.for %arg2 = %c0 to %c128 step %c1 {
      %0 = arith.index_cast %arg2 : index to i32
      %1 = arith.muli %0, %c16_i32 : i32
      scf.for %arg3 = %c0 to %c16 step %c1 {
        %2 = arith.index_cast %arg3 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg0[%4] : memref<?xi32>
        %6 = memref.load %arg1[%arg2] : memref<?xi32>
        %7 = arith.addi %5, %6 : i32
        memref.store %7, %arg0[%4] : memref<?xi32>
      }
    }
    return
  }
  func.func @init(%arg0: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c2048 = arith.constant 2048 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg1 = %c0 to %c2048 step %c1 {
      memref.store %c0_i32, %arg0[%arg1] : memref<?xi32>
    }
    return
  }
  func.func @hist(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c512_i32 = arith.constant 512 : i32
    scf.for %arg3 = %c0 to %c512 step %c1 {
      %0 = arith.index_cast %arg3 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      scf.for %arg4 = %c0 to %c4 step %c1 {
        %2 = arith.index_cast %arg4 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg1[%4] : memref<?xi32>
        %6 = arith.shrsi %5, %arg2 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.muli %7, %c512_i32 : i32
        %9 = arith.addi %8, %0 : i32
        %10 = arith.addi %9, %c1_i32 : i32
        %11 = arith.index_cast %10 : i32 to index
        %12 = memref.load %arg0[%11] : memref<?xi32>
        %13 = arith.addi %12, %c1_i32 : i32
        memref.store %13, %arg0[%11] : memref<?xi32>
      }
    }
    return
  }
  func.func @update(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: i32) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c1_i32 = arith.constant 1 : i32
    %c3_i32 = arith.constant 3 : i32
    %c4_i32 = arith.constant 4 : i32
    %c512_i32 = arith.constant 512 : i32
    scf.for %arg4 = %c0 to %c512 step %c1 {
      %0 = arith.index_cast %arg4 : index to i32
      %1 = arith.muli %0, %c4_i32 : i32
      scf.for %arg5 = %c0 to %c4 step %c1 {
        %2 = arith.index_cast %arg5 : index to i32
        %3 = arith.addi %1, %2 : i32
        %4 = arith.index_cast %3 : i32 to index
        %5 = memref.load %arg2[%4] : memref<?xi32>
        %6 = arith.shrsi %5, %arg3 : i32
        %7 = arith.andi %6, %c3_i32 : i32
        %8 = arith.muli %7, %c512_i32 : i32
        %9 = arith.addi %8, %0 : i32
        %10 = arith.index_cast %9 : i32 to index
        %11 = memref.load %arg1[%10] : memref<?xi32>
        %12 = arith.index_cast %11 : i32 to index
        memref.store %5, %arg0[%12] : memref<?xi32>
        %13 = memref.load %arg1[%10] : memref<?xi32>
        %14 = arith.addi %13, %c1_i32 : i32
        memref.store %14, %arg1[%10] : memref<?xi32>
      }
    }
    return
  }
  func.func @ss_sort(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: memref<?xi32>, %arg3: memref<?xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c16_i32 = arith.constant 16 : i32
    %c-1_i32 = arith.constant -1 : i32
    %c16 = arith.constant 16 : index
    %c128 = arith.constant 128 : index
    %c512_i32 = arith.constant 512 : i32
    %c4_i32 = arith.constant 4 : i32
    %c3_i32 = arith.constant 3 : i32
    %c1_i32 = arith.constant 1 : i32
    %c4 = arith.constant 4 : index
    %c512 = arith.constant 512 : index
    %c1 = arith.constant 1 : index
    %c2048 = arith.constant 2048 : index
    %c32 = arith.constant 32 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.for %arg4 = %c0 to %c32 step %c2 iter_args(%arg5 = %c0_i32) -> (i32) {
      %1 = arith.index_cast %arg4 : index to i32
      scf.for %arg6 = %c0 to %c2048 step %c1 {
        memref.store %c0_i32, %arg2[%arg6] : memref<?xi32>
      }
      %2 = arith.cmpi eq, %arg5, %c0_i32 : i32
      scf.if %2 {
        scf.for %arg6 = %c0 to %c512 step %c1 {
          %4 = arith.index_cast %arg6 : index to i32
          %5 = arith.muli %4, %c4_i32 : i32
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %6 = arith.index_cast %arg7 : index to i32
            %7 = arith.addi %5, %6 : i32
            %8 = arith.index_cast %7 : i32 to index
            %9 = memref.load %arg0[%8] : memref<?xi32>
            %10 = arith.shrsi %9, %1 : i32
            %11 = arith.andi %10, %c3_i32 : i32
            %12 = arith.muli %11, %c512_i32 : i32
            %13 = arith.addi %12, %4 : i32
            %14 = arith.addi %13, %c1_i32 : i32
            %15 = arith.index_cast %14 : i32 to index
            %16 = memref.load %arg2[%15] : memref<?xi32>
            %17 = arith.addi %16, %c1_i32 : i32
            memref.store %17, %arg2[%15] : memref<?xi32>
          }
        }
      } else {
        scf.for %arg6 = %c0 to %c512 step %c1 {
          %4 = arith.index_cast %arg6 : index to i32
          %5 = arith.muli %4, %c4_i32 : i32
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %6 = arith.index_cast %arg7 : index to i32
            %7 = arith.addi %5, %6 : i32
            %8 = arith.index_cast %7 : i32 to index
            %9 = memref.load %arg1[%8] : memref<?xi32>
            %10 = arith.shrsi %9, %1 : i32
            %11 = arith.andi %10, %c3_i32 : i32
            %12 = arith.muli %11, %c512_i32 : i32
            %13 = arith.addi %12, %4 : i32
            %14 = arith.addi %13, %c1_i32 : i32
            %15 = arith.index_cast %14 : i32 to index
            %16 = memref.load %arg2[%15] : memref<?xi32>
            %17 = arith.addi %16, %c1_i32 : i32
            memref.store %17, %arg2[%15] : memref<?xi32>
          }
        }
      }
      scf.for %arg6 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg6 : index to i32
        %5 = arith.muli %4, %c16_i32 : i32
        scf.for %arg7 = %c1 to %c16 step %c1 {
          %6 = arith.index_cast %arg7 : index to i32
          %7 = arith.addi %5, %6 : i32
          %8 = arith.index_cast %7 : i32 to index
          %9 = arith.addi %7, %c-1_i32 : i32
          %10 = arith.index_cast %9 : i32 to index
          %11 = memref.load %arg2[%10] : memref<?xi32>
          %12 = memref.load %arg2[%8] : memref<?xi32>
          %13 = arith.addi %12, %11 : i32
          memref.store %13, %arg2[%8] : memref<?xi32>
        }
      }
      func.call @sum_scan(%arg3, %arg2) : (memref<?xi32>, memref<?xi32>) -> ()
      scf.for %arg6 = %c0 to %c128 step %c1 {
        %4 = arith.index_cast %arg6 : index to i32
        %5 = arith.muli %4, %c16_i32 : i32
        scf.for %arg7 = %c0 to %c16 step %c1 {
          %6 = arith.index_cast %arg7 : index to i32
          %7 = arith.addi %5, %6 : i32
          %8 = arith.index_cast %7 : i32 to index
          %9 = memref.load %arg2[%8] : memref<?xi32>
          %10 = memref.load %arg3[%arg6] : memref<?xi32>
          %11 = arith.addi %9, %10 : i32
          memref.store %11, %arg2[%8] : memref<?xi32>
        }
      }
      %3 = arith.extui %2 : i1 to i32
      scf.if %2 {
        scf.for %arg6 = %c0 to %c512 step %c1 {
          %4 = arith.index_cast %arg6 : index to i32
          %5 = arith.muli %4, %c4_i32 : i32
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %6 = arith.index_cast %arg7 : index to i32
            %7 = arith.addi %5, %6 : i32
            %8 = arith.index_cast %7 : i32 to index
            %9 = memref.load %arg0[%8] : memref<?xi32>
            %10 = arith.shrsi %9, %1 : i32
            %11 = arith.andi %10, %c3_i32 : i32
            %12 = arith.muli %11, %c512_i32 : i32
            %13 = arith.addi %12, %4 : i32
            %14 = arith.index_cast %13 : i32 to index
            %15 = memref.load %arg2[%14] : memref<?xi32>
            %16 = arith.index_cast %15 : i32 to index
            memref.store %9, %arg1[%16] : memref<?xi32>
            %17 = memref.load %arg2[%14] : memref<?xi32>
            %18 = arith.addi %17, %c1_i32 : i32
            memref.store %18, %arg2[%14] : memref<?xi32>
          }
        }
      } else {
        scf.for %arg6 = %c0 to %c512 step %c1 {
          %4 = arith.index_cast %arg6 : index to i32
          %5 = arith.muli %4, %c4_i32 : i32
          scf.for %arg7 = %c0 to %c4 step %c1 {
            %6 = arith.index_cast %arg7 : index to i32
            %7 = arith.addi %5, %6 : i32
            %8 = arith.index_cast %7 : i32 to index
            %9 = memref.load %arg1[%8] : memref<?xi32>
            %10 = arith.shrsi %9, %1 : i32
            %11 = arith.andi %10, %c3_i32 : i32
            %12 = arith.muli %11, %c512_i32 : i32
            %13 = arith.addi %12, %4 : i32
            %14 = arith.index_cast %13 : i32 to index
            %15 = memref.load %arg2[%14] : memref<?xi32>
            %16 = arith.index_cast %15 : i32 to index
            memref.store %9, %arg0[%16] : memref<?xi32>
            %17 = memref.load %arg2[%14] : memref<?xi32>
            %18 = arith.addi %17, %c1_i32 : i32
            memref.store %18, %arg2[%14] : memref<?xi32>
          }
        }
      }
      scf.yield %3 : i32
    }
    return
  }
}


module attributes {adora.scheduled} {
  func.func @attention(%arg0: memref<?x16xi32>, %arg1: memref<?x16xi32>, %arg2: memref<?x16xi32>, %arg3: memref<?x16xi32>) attributes {adora.dep_summary = [], llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %alloca = memref.alloca() : memref<8x8xi32>
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c8 step %c1 {
        %0 = scf.for %arg6 = %c0 to %c16 step %c1 iter_args(%arg7 = %c0_i32) -> (i32) {
          %1 = memref.load %arg0[%arg4, %arg6] : memref<?x16xi32>
          %2 = memref.load %arg1[%arg5, %arg6] : memref<?x16xi32>
          %3 = arith.muli %1, %2 : i32
          %4 = arith.addi %arg7, %3 : i32
          scf.yield %4 : i32
        }
        memref.store %0, %alloca[%arg4, %arg5] : memref<8x8xi32>
      }
    }
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c8 step %c1 {
        %0 = memref.load %alloca[%arg4, %arg5] : memref<8x8xi32>
        %1 = arith.shrsi %0, %c2_i32 : i32
        memref.store %1, %alloca[%arg4, %arg5] : memref<8x8xi32>
      }
    }
    scf.for %arg4 = %c0 to %c8 step %c1 {
      scf.for %arg5 = %c0 to %c16 step %c1 {
        %0 = scf.for %arg6 = %c0 to %c8 step %c1 iter_args(%arg7 = %c0_i32) -> (i32) {
          %1 = memref.load %alloca[%arg4, %arg6] : memref<8x8xi32>
          %2 = arith.cmpi sgt, %1, %c0_i32 : i32
          %3 = arith.select %2, %1, %c0_i32 : i32
          %4 = memref.load %arg2[%arg6, %arg5] : memref<?x16xi32>
          %5 = arith.muli %3, %4 : i32
          %6 = arith.addi %arg7, %5 : i32
          scf.yield %6 : i32
        }
        memref.store %0, %arg3[%arg4, %arg5] : memref<?x16xi32>
      }
    }
    return
  }
}


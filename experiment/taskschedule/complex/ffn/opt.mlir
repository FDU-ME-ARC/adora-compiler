module {
  func.func @ffn_fc1(%arg0: memref<?x32xi32>, %arg1: memref<?x64xi32>, %arg2: memref<?x64xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg3 = %c0 to %c16 step %c1 {
      scf.for %arg4 = %c0 to %c64 step %c1 {
        %0 = scf.for %arg5 = %c0 to %c32 step %c1 iter_args(%arg6 = %c0_i32) -> (i32) {
          %3 = memref.load %arg0[%arg3, %arg5] : memref<?x32xi32>
          %4 = memref.load %arg1[%arg5, %arg4] : memref<?x64xi32>
          %5 = arith.muli %3, %4 : i32
          %6 = arith.addi %arg6, %5 : i32
          scf.yield %6 : i32
        }
        %1 = arith.cmpi sgt, %0, %c0_i32 : i32
        %2 = arith.select %1, %0, %c0_i32 : i32
        memref.store %2, %arg2[%arg3, %arg4] : memref<?x64xi32>
      }
    }
    return
  }
  func.func @ffn_fc2(%arg0: memref<?x64xi32>, %arg1: memref<?x32xi32>, %arg2: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %c64 = arith.constant 64 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg3 = %c0 to %c16 step %c1 {
      scf.for %arg4 = %c0 to %c32 step %c1 {
        %0 = scf.for %arg5 = %c0 to %c64 step %c1 iter_args(%arg6 = %c0_i32) -> (i32) {
          %1 = memref.load %arg0[%arg3, %arg5] : memref<?x64xi32>
          %2 = memref.load %arg1[%arg5, %arg4] : memref<?x32xi32>
          %3 = arith.muli %1, %2 : i32
          %4 = arith.addi %arg6, %3 : i32
          scf.yield %4 : i32
        }
        memref.store %0, %arg2[%arg3, %arg4] : memref<?x32xi32>
      }
    }
    return
  }
  func.func @ffn(%arg0: memref<?x32xi32>, %arg1: memref<?x64xi32>, %arg2: memref<?x32xi32>, %arg3: memref<?x64xi32>, %arg4: memref<?x32xi32>) attributes {llvm.linkage = #llvm.linkage<external>} {
    %c16 = arith.constant 16 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c64 = arith.constant 64 : index
    %c32 = arith.constant 32 : index
    %c0_i32 = arith.constant 0 : i32
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c64 step %c1 {
        %0 = scf.for %arg7 = %c0 to %c32 step %c1 iter_args(%arg8 = %c0_i32) -> (i32) {
          %3 = memref.load %arg0[%arg5, %arg7] : memref<?x32xi32>
          %4 = memref.load %arg1[%arg7, %arg6] : memref<?x64xi32>
          %5 = arith.muli %3, %4 : i32
          %6 = arith.addi %arg8, %5 : i32
          scf.yield %6 : i32
        }
        %1 = arith.cmpi sgt, %0, %c0_i32 : i32
        %2 = arith.select %1, %0, %c0_i32 : i32
        memref.store %2, %arg3[%arg5, %arg6] : memref<?x64xi32>
      }
    }
    scf.for %arg5 = %c0 to %c16 step %c1 {
      scf.for %arg6 = %c0 to %c32 step %c1 {
        %0 = scf.for %arg7 = %c0 to %c64 step %c1 iter_args(%arg8 = %c0_i32) -> (i32) {
          %1 = memref.load %arg3[%arg5, %arg7] : memref<?x64xi32>
          %2 = memref.load %arg2[%arg7, %arg6] : memref<?x32xi32>
          %3 = arith.muli %1, %2 : i32
          %4 = arith.addi %arg8, %3 : i32
          scf.yield %4 : i32
        }
        memref.store %0, %arg4[%arg5, %arg6] : memref<?x32xi32>
      }
    }
    return
  }
}


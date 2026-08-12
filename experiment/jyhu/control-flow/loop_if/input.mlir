// The loop-carried value updates only when input[i] is positive.
module attributes {} {
  func.func @loop_if(%input: memref<16xi32>, %output: memref<16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    %sum = affine.for %i = 0 to 16 iter_args(%acc = %c0_i32) -> (i32) {
      %loaded = affine.load %input[%i] : memref<16xi32>
      %positive = arith.cmpi sgt, %loaded, %c0_i32 : i32
      %next = scf.if %positive -> (i32) {
        %updated = arith.addi %acc, %loaded : i32
        scf.yield %updated : i32
      } else {
        scf.yield %acc : i32
      }
      affine.store %next, %output[%i] : memref<16xi32>
      affine.yield %next : i32
    }
    return
  }
}

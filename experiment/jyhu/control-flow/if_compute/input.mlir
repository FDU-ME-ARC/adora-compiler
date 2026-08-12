// Both branches are pure calculations; only the yielded value is selected.
module attributes {} {
  func.func @if_compute(%a: i1, %input: memref<16xi32>, %output: memref<16xi32>) {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c3_i32 = arith.constant 3 : i32
    affine.for %i = 0 to 16 {
      %in = affine.load %input[%i] : memref<16xi32>
      %value = scf.if %a -> (i32) {
        %twice = arith.muli %in, %c2_i32 : i32
        %then = arith.addi %twice, %c1_i32 : i32
        scf.yield %then : i32
      } else {
        %else = arith.addi %in, %c3_i32 : i32
        scf.yield %else : i32
      }
      affine.store %value, %output[%i] : memref<16xi32>
    }
    return
  }
}

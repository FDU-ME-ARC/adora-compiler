// Path condition: store executes under a.
module attributes {} {
  func.func @if_simple(%a: i1, %output: memref<16xi32>) {
    %c1_i32 = arith.constant 1 : i32
    affine.for %i = 0 to 16 {
      %old = affine.load %output[%i] : memref<16xi32>
      %next = arith.addi %old, %c1_i32 : i32
      scf.if %a {
        affine.store %next, %output[%i] : memref<16xi32>
      }
    }
    return
  }
}

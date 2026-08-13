// Path condition: store executes under a&&b.
module attributes {} {
  func.func @nested_if(%a: i1, %b: i1, %output: memref<16xi32>) {
    %c1_i32 = arith.constant 1 : i32
    affine.for %i = 0 to 16 {
      scf.if %a {
        scf.if %b {
          affine.store %c1_i32, %output[%i] : memref<16xi32>
        }
      }
    }
    return
  }
}

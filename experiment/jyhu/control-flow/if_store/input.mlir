// The store is control-dependent; false must preserve the old value.
module attributes {} {
  func.func @if_store(%a: i1, %output: memref<16xi32>) {
    %c7_i32 = arith.constant 7 : i32
    affine.for %i = 0 to 16 {
      scf.if %a {
        affine.store %c7_i32, %output[%i] : memref<16xi32>
      }
    }
    return
  }
}

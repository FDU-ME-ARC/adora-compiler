// Path conditions: then=a, else=!a.
module attributes {} {
  func.func @if_else(%a: i1, %output: memref<16xi32>) {
    %c11_i32 = arith.constant 11 : i32
    %c22_i32 = arith.constant 22 : i32
    affine.for %i = 0 to 16 {
      %value = scf.if %a -> (i32) {
        scf.yield %c11_i32 : i32
      } else {
        scf.yield %c22_i32 : i32
      }
      affine.store %value, %output[%i] : memref<16xi32>
    }
    return
  }
}

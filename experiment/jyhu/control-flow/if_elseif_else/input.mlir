// Path conditions: S1=a, S2=!a&&b, S3=!a&&!b.
module attributes {} {
  func.func @if_elseif_else(%a: i1, %b: i1, %output: memref<16xi32>) {
    %c11_i32 = arith.constant 11 : i32
    %c22_i32 = arith.constant 22 : i32
    %c33_i32 = arith.constant 33 : i32
    affine.for %i = 0 to 16 {
      %value = scf.if %a -> (i32) {
        scf.yield %c11_i32 : i32
      } else {
        %fallback = scf.if %b -> (i32) {
          scf.yield %c22_i32 : i32
        } else {
          scf.yield %c33_i32 : i32
        }
        scf.yield %fallback : i32
      }
      affine.store %value, %output[%i] : memref<16xi32>
    }
    return
  }
}

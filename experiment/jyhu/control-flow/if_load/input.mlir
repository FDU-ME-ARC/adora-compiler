// The input load is control-dependent in the source program.
module attributes {} {
  func.func @if_load(%a: i1, %input: memref<16xi32>, %output: memref<16xi32>) {
    %c0_i32 = arith.constant 0 : i32
    affine.for %i = 0 to 16 {
      %value = scf.if %a -> (i32) {
        %loaded = affine.load %input[%i] : memref<16xi32>
        scf.yield %loaded : i32
      } else {
        scf.yield %c0_i32 : i32
      }
      affine.store %value, %output[%i] : memref<16xi32>
    }
    return
  }
}

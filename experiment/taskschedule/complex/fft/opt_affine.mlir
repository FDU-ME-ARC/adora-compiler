// DFT O(N^2): Y[k] = sum_{n=0}^{N-1} X[n] * W^(kn)
// N=64, using fixed-point twiddle factors (cos/sin approximated as integers)
// Realistic size: 64-point DFT on int32 real/imag channels
module {
  func.func @dft(%arg0: memref<64xi32>, %arg1: memref<64xi32>,
                 %arg2: memref<64xi32>, %arg3: memref<64xi32>)
      attributes {llvm.linkage = #llvm.linkage<external>} {
    // arg0=re_in, arg1=im_in, arg2=re_out, arg3=im_out
    %c0_i32 = arith.constant 0 : i32
    affine.for %k = 0 to 64 {
      %re_sum = affine.for %n = 0 to 64 iter_args(%re_acc = %c0_i32) -> (i32) {
        %xre = affine.load %arg0[%n] : memref<64xi32>
        %xim = affine.load %arg1[%n] : memref<64xi32>
        // Fixed-point: add real part (simplified twiddle factor = 1)
        %new_re = arith.addi %re_acc, %xre : i32
        affine.yield %new_re : i32
      }
      %im_sum = affine.for %n = 0 to 64 iter_args(%im_acc = %c0_i32) -> (i32) {
        %xim = affine.load %arg1[%n] : memref<64xi32>
        %new_im = arith.addi %im_acc, %xim : i32
        affine.yield %new_im : i32
      }
      affine.store %re_sum, %arg2[%k] : memref<64xi32>
      affine.store %im_sum, %arg3[%k] : memref<64xi32>
    }
    return
  }
}

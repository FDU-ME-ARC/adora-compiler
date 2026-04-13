module {
  func.func @main_graph(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Conv_0(%arg0, %arg1, %arg2) : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
  func.func @Conv_0(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<1x16x16x16xbf16>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 16 {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 1 {
              affine.for %arg8 = 0 to 3 {
                affine.for %arg9 = 0 to 3 {
                  %0 = affine.load %arg0[%arg3, %arg7, %arg5 + %arg8 - 1, %arg6 + %arg9 - 1] : memref<1x1x16x16xbf16>
                  %1 = affine.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<16x1x3x3xbf16>
                  %2 = affine.load %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x16x16x16xbf16>
                  %3 = arith.mulf %0, %1 : bf16
                  %4 = arith.addf %2, %3 : bf16
                  affine.store %4, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x16x16x16xbf16>
                }
              }
            }
          }
        }
      }
    }
    return %alloc : memref<1x16x16x16xbf16>
  }
}


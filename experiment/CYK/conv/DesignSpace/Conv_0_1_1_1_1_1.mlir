module {
  func.func @main_graph(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Conv_0(%arg0, %arg1, %arg2) : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
  func.func @Conv_0(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<1x16x16x16xbf16>
    %0 = ADORA.BlockLoad %alloc [0, 0, 0, 0] : memref<1x16x16x16xbf16> -> memref<1x16x16x16xbf16>  {Id = "0", KernelName = "Conv_0"}
    %1 = ADORA.BlockLoad %arg0 [0, 0, 0, 0] : memref<1x1x16x16xbf16> -> memref<1x1x16x16xbf16>  {Id = "1", KernelName = "Conv_0"}
    %2 = ADORA.BlockLoad %arg1 [0, 0, 0, 0] : memref<16x1x3x3xbf16> -> memref<16x1x3x3xbf16>  {Id = "2", KernelName = "Conv_0"}
    %3 = ADORA.LocalMemAlloc memref<1x16x16x16xbf16>  {Id = "3", KernelName = "Conv_0"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %4 = affine.load %0[0, %arg3, %arg4, %arg5] : memref<1x16x16x16xbf16>
            %5 = affine.for %arg6 = 0 to 3 iter_args(%arg7 = %4) -> (bf16) {
              %6 = affine.for %arg8 = 0 to 3 iter_args(%arg9 = %arg7) -> (bf16) {
                %7 = affine.load %1[0, 0, -1, %arg5 + %arg8 - 1] : memref<1x1x16x16xbf16>
                %8 = affine.load %2[%arg3, 0, %arg6, %arg8] : memref<16x1x3x3xbf16>
                %9 = arith.mulf %7, %8 : bf16
                %10 = arith.addf %arg9, %9 : bf16
                affine.yield %10 : bf16
              }
              affine.yield %6 : bf16
            }
            affine.store %5, %3[0, %arg3, %arg4, %arg5] : memref<1x16x16x16xbf16>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "Conv_0"}
    ADORA.BlockStore %3, %alloc [0, 0, 0, 0] : memref<1x16x16x16xbf16> -> memref<1x16x16x16xbf16>  {Id = "3", KernelName = "Conv_0"}
    return %alloc : memref<1x16x16x16xbf16>
  }
}

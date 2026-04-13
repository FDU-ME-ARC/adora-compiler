module {
  func.func @main_graph(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Conv_0(%arg0, %arg1, %arg2) : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
  func.func @Conv_0(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
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
              %6 = affine.load %1[0, 0, -1, %arg5 - 1] : memref<1x1x16x16xbf16>
              %7 = affine.load %2[%arg3, 0, %arg6, 0] : memref<16x1x3x3xbf16>
              %8 = arith.mulf %6, %7 : bf16
              %9 = arith.addf %arg7, %8 : bf16
              %c1 = arith.constant 1 : index
              %10 = affine.load %1[0, 0, -1, %arg5] : memref<1x1x16x16xbf16>
              %11 = affine.load %2[%arg3, 0, %arg6, 1] : memref<16x1x3x3xbf16>
              %12 = arith.mulf %10, %11 : bf16
              %13 = arith.addf %9, %12 : bf16
              %c2 = arith.constant 2 : index
              %14 = affine.load %1[0, 0, -1, %arg5 + 1] : memref<1x1x16x16xbf16>
              %15 = affine.load %2[%arg3, 0, %arg6, 2] : memref<16x1x3x3xbf16>
              %16 = arith.mulf %14, %15 : bf16
              %17 = arith.addf %13, %16 : bf16
              affine.yield %17 : bf16
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

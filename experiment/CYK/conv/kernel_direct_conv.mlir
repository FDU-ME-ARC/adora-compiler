module {
  memref.global constant @constant_1 : memref<16xf32> = dense<1.000000e-01>
  memref.global constant @constant_0 : memref<16x3x3x3xf32> = dense<5.000000e-01>
  func.func @main_graph(%arg0: memref<1x3x32x32xf32>) -> memref<1x16x32x32xf32> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = memref.get_global @constant_0 : memref<16x3x3x3xf32>
    %1 = memref.get_global @constant_1 : memref<16xf32>
    %2 = call @Conv_0(%arg0, %0, %1) : (memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>) -> memref<1x16x32x32xf32>
    return %2 : memref<1x16x32x32xf32>
  }
  func.func @Conv_0(%arg0: memref<1x3x32x32xf32>, %arg1: memref<16x3x3x3xf32>, %arg2: memref<16xf32>) -> memref<1x16x32x32xf32> attributes {adora_kernel, llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<1x16x32x32xf32>
    affine.for %arg3 = 0 to 1 step 64 {
      affine.for %arg4 = 0 to 16 step 16 {
        affine.for %arg5 = 0 to 32 step 4 {
          affine.for %arg6 = 0 to 32 step 9 {
            %c0 = arith.constant {ADORAConv} 0 : index
            %0 = ADORA.BlockLoad %arg0 [%arg3, 0, %arg5 - 1, %arg6 - 1] : memref<1x3x32x32xf32> -> memref<64x3x6x11xf32>  {ADORAConv, Id = "0", KernelName = "ConvDirect", Pingpong}
            %1 = ADORA.BlockLoad %arg1 [%arg4, 0, 0, 0] : memref<16x3x3x3xf32> -> memref<16x3x3x3xf32>  {ADORAConv, Id = "1", KernelName = "ConvDirect", Pingpong}
            %2 = ADORA.LocalMemAlloc memref<64x16x4x9xf32>  {ADORAConv, Id = "2", KernelName = "ConvDirect"}
            affine.for %arg7 = 0 to 64 {
              affine.for %arg8 = 0 to 16 {
                affine.for %arg9 = 0 to 4 {
                  affine.for %arg10 = 0 to 9 {
                    %3 = affine.load %arg2[%arg8 + symbol(%arg4)] {ADORAConv, Pingpong} : memref<16xf32>
                    affine.store %3, %2[%arg7, %arg8, %arg9, %arg10] {ADORAConv, Pingpong} : memref<64x16x4x9xf32>
                  } {ADORAConv}
                } {ADORAConv}
              } {ADORAConv}
            } {ADORAConv}
            ADORA.kernel {
              affine.for %arg7 = 0 to 3 {
                affine.for %arg8 = 0 to 3 {
                  affine.for %arg9 = 0 to 3 {
                    affine.for %arg10 = 0 to 64 {
                      affine.for %arg11 = 0 to 16 {
                        affine.for %arg12 = 0 to 4 {
                          affine.for %arg13 = 0 to 9 {
                            %3 = affine.load %0[%arg10, %arg7, %arg12 + %arg8, %arg13 + %arg9] {ADORAConv, Pingpong} : memref<64x3x6x11xf32>
                            %4 = affine.load %1[%arg11, %arg7, %arg8, %arg9] {ADORAConv, Pingpong} : memref<16x3x3x3xf32>
                            %5 = affine.load %2[%arg10, %arg11, %arg12, %arg13] {ADORAConv, Pingpong} : memref<64x16x4x9xf32>
                            %6 = arith.mulf %3, %4 {ADORAConv} : f32
                            %7 = arith.addf %5, %6 {ADORAConv} : f32
                            affine.store %7, %2[%arg10, %arg11, %arg12, %arg13] {ADORAConv, Pingpong} : memref<64x16x4x9xf32>
                          } {ADORAConv}
                        } {ADORAConv}
                      } {ADORAConv}
                    } {ADORAConv}
                  } {ADORAConv}
                } {ADORAConv}
              } {ADORAConv}
              ADORA.terminator {ADORAConv}
            } {ADORAConv, KernelName = "ConvDirect"}
            ADORA.BlockStore %2, %alloc [%arg3, %arg4, %arg5, %arg6] : memref<64x16x4x9xf32> -> memref<1x16x32x32xf32>  {ADORAConv, Id = "2", KernelName = "ConvDirect", Pingpong}
          } {ADORAConv}
        } {ADORAConv}
      } {ADORAConv}
    } {ADORAConv}
    return %alloc : memref<1x16x32x32xf32>
  }
}


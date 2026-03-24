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
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    affine.for %arg3 = 0 to 32 step 4 {
      affine.for %arg4 = 0 to 32 step 9 {
        %c0_1 = arith.constant {ADORAConv} 0 : index
        %0 = ADORA.BlockLoad %arg0 [%c0, 0, %arg3 - 1, %arg4 - 1] : memref<1x3x32x32xf32> -> memref<64x3x6x11xf32>  {ADORAConv, Id = "0", KernelName = "ConvDirect", Pingpong}
        %1 = ADORA.BlockLoad %arg1 [%c0_0, 0, 0, 0] : memref<16x3x3x3xf32> -> memref<16x3x3x3xf32>  {ADORAConv, Id = "1", KernelName = "ConvDirect", Pingpong}
        %2 = ADORA.LocalMemAlloc memref<64x16x4x9xf32>  {ADORAConv, Id = "2", KernelName = "ConvDirect"}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            affine.for %arg7 = 0 to 4 {
              affine.for %arg8 = 0 to 9 {
                %3 = affine.load %arg2[%arg6 + symbol(%c0_0)] {ADORAConv, Pingpong} : memref<16xf32>
                affine.store %3, %2[%arg5, %arg6, %arg7, %arg8] {ADORAConv, Pingpong} : memref<64x16x4x9xf32>
              } {ADORAConv}
            } {ADORAConv}
          } {ADORAConv}
        } {ADORAConv}
        affine.for %arg5 = 0 to 3 {
          affine.for %arg6 = 0 to 3 {
            affine.for %arg7 = 0 to 3 {
              affine.for %arg8 = 0 to 64 step 2 {
                %3 = ADORA.BlockLoad %0 [%arg8, %arg5, %arg6, %arg7] : memref<64x3x6x11xf32> -> memref<2x1x4x11xf32>  {Id = "0", KernelName = "ConvDirect"}
                %4 = ADORA.BlockLoad %1 [0, %arg5, %arg6, %arg7] : memref<16x3x3x3xf32> -> memref<16x3x3x3xf32>  {Id = "1", KernelName = "ConvDirect"}
                %5 = ADORA.BlockLoad %2 [%arg8, 0, 0, 0] : memref<64x16x4x9xf32> -> memref<2x16x4x9xf32>  {Id = "2", KernelName = "ConvDirect"}
                %6 = ADORA.LocalMemAlloc memref<2x16x4x9xf32>  {Id = "3", KernelName = "ConvDirect"}
                ADORA.kernel {
                  affine.for %arg9 = 0 to 2 {
                    affine.for %arg10 = 0 to 16 {
                      affine.for %arg11 = 0 to 4 {
                        affine.for %arg12 = 0 to 9 {
                          %7 = affine.load %3[%arg9, 0, %arg11, %arg12] {ADORAConv, Pingpong} : memref<2x1x4x11xf32>
                          %8 = affine.load %4[%arg10, 0, 0, 0] {ADORAConv, Pingpong} : memref<16x3x3x3xf32>
                          %9 = affine.load %5[%arg9, %arg10, %arg11, %arg12] {ADORAConv, Pingpong} : memref<2x16x4x9xf32>
                          %10 = arith.mulf %7, %8 {ADORAConv} : f32
                          %11 = arith.addf %9, %10 {ADORAConv} : f32
                          affine.store %11, %6[%arg9, %arg10, %arg11, %arg12] {ADORAConv, Pingpong} : memref<2x16x4x9xf32>
                        } {ADORAConv}
                      } {ADORAConv}
                    } {ADORAConv}
                  } {ADORAConv}
                  ADORA.terminator
                } {KernelName = "ConvDirect"}
                ADORA.BlockStore %6, %2 [%arg8, 0, 0, 0] : memref<2x16x4x9xf32> -> memref<64x16x4x9xf32>  {Id = "3", KernelName = "ConvDirect"}
              }
            } {ADORAConv}
          } {ADORAConv}
        } {ADORAConv}
        ADORA.BlockStore %2, %alloc [%c0, %c0_0, %arg3, %arg4] : memref<64x16x4x9xf32> -> memref<1x16x32x32xf32>  {ADORAConv, Id = "2", KernelName = "ConvDirect", Pingpong}
      } {ADORAConv}
    } {ADORAConv}
    return %alloc : memref<1x16x32x32xf32>
  }
}


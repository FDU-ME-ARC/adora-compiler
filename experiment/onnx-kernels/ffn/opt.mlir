module {
  func.func @main_graph(%arg0: memref<64x128xbf16> {onnx.name = "input"}, %arg1: memref<128x256xbf16> {onnx.name = "weight1"}, %arg2: memref<64x256xbf16> {onnx.name = "bias1"}, %arg3: memref<256x128xbf16> {onnx.name = "weight2"}, %arg4: memref<64x128xbf16> {onnx.name = "bias2"}) -> (memref<64x128xbf16> {onnx.name = "output"}) attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x256xbf16>, memref<64x256xbf16>) -> memref<64x256xbf16>
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<64x256xbf16>
    affine.for %arg5 = 0 to 64 step 16 {
      %2 = ADORA.BlockLoad %0 [%arg5, 0] : memref<64x256xbf16> -> memref<16x256xbf16>  {Id = "0", KernelName = "main_graph"}
      %3 = ADORA.LocalMemAlloc memref<16x256xbf16>  {Id = "1", KernelName = "main_graph"}
      ADORA.kernel {
        affine.for %arg6 = 0 to 16 {
          affine.for %arg7 = 0 to 256 {
            %4 = affine.load %2[%arg6, %arg7] : memref<16x256xbf16>
            %5 = arith.cmpf ugt, %4, %cst : bf16
            %6 = arith.select %5, %4, %cst : bf16
            %7 = arith.cmpf uno, %4, %4 : bf16
            %8 = arith.select %7, %cst, %6 : bf16
            affine.store %8, %3[%arg6, %arg7] : memref<16x256xbf16>
          }
        }
        ADORA.terminator
      } {KernelName = "main_graph"}
      ADORA.BlockStore %3, %alloc [%arg5, 0] : memref<16x256xbf16> -> memref<64x256xbf16>  {Id = "1", KernelName = "main_graph"}
    }
    %1 = call @Gemm_1(%alloc, %arg3, %arg4) : (memref<64x256xbf16>, memref<256x128xbf16>, memref<64x128xbf16>) -> memref<64x128xbf16>
    return %1 : memref<64x128xbf16>
  }
  func.func @Gemm_0(%arg0: memref<64x128xbf16>, %arg1: memref<128x256xbf16>, %arg2: memref<64x256xbf16>) -> memref<64x256xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", stationary_kind = "InputStationary", tile_size = array<i64: 16, 256, 2, 8>} : (memref<64x128xbf16>, memref<128x256xbf16>, memref<64x256xbf16>) -> memref<64x256xbf16>
    return %0 : memref<64x256xbf16>
  }
  func.func @Gemm_1(%arg0: memref<64x256xbf16>, %arg1: memref<256x128xbf16>, %arg2: memref<64x128xbf16>) -> memref<64x128xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", stationary_kind = "InputStationary", tile_size = array<i64: 32, 128, 2, 8>} : (memref<64x256xbf16>, memref<256x128xbf16>, memref<64x128xbf16>) -> memref<64x128xbf16>
    return %0 : memref<64x128xbf16>
  }
}


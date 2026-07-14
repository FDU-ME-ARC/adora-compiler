module {
  func.func @main_graph(%arg0: memref<64x128xbf16> {onnx.name = "input"}, %arg1: memref<128x64xbf16> {onnx.name = "weight"}, %arg2: memref<64x64xbf16> {onnx.name = "bias"}) -> (memref<64x64xbf16> {onnx.name = "output"}) attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<64x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<64x64xbf16>) -> memref<64x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", stationary_kind = "InputStationary", tile_size = array<i64: 32, 64, 2, 16>} : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
}


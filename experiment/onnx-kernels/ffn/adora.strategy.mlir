module attributes {} {
  func.func @main_graph(%arg0: memref<64x128xbf16> {onnx.name = "input"}, %arg1: memref<128x256xbf16> {onnx.name = "weight1"}, %arg2: memref<64x256xbf16> {onnx.name = "bias1"}, %arg3: memref<256x128xbf16> {onnx.name = "weight2"}, %arg4: memref<64x128xbf16> {onnx.name = "bias2"}) -> (memref<64x128xbf16> {onnx.name = "output"}) attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x256xbf16>, memref<64x256xbf16>) -> memref<64x256xbf16>
    %c64 = arith.constant 64 : index
    %c256 = arith.constant 256 : index
    %c64_0 = arith.constant 64 : index
    %c256_1 = arith.constant 256 : index
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<64x256xbf16>
    %c0 = arith.constant 0 : index
    %c64_2 = arith.constant 64 : index
    %c256_3 = arith.constant 256 : index
    %cst = arith.constant 0.000000e+00 : bf16
    affine.for %arg5 = 0 to 64 {
      affine.for %arg6 = 0 to 256 {
        %2 = affine.load %0[%arg5, %arg6] : memref<64x256xbf16>
        %3 = arith.maxnumf %cst, %2 : bf16
        affine.store %3, %alloc[%arg5, %arg6] : memref<64x256xbf16>
      }
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


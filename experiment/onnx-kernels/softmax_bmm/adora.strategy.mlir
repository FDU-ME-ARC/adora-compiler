module attributes {} {
  func.func @main_graph(%arg0: memref<64x128xbf16> {onnx.name = "input"}, %arg1: memref<128x64xbf16> {onnx.name = "weight"}, %arg2: memref<64x64xbf16> {onnx.name = "bias"}) -> (memref<64x64xbf16> {onnx.name = "output"}) attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %alloc = memref.alloc() {alignment = 16 : i64} : memref<64x128xbf16>
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_0 = arith.constant 0xFF80 : bf16
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %c128 = arith.constant 128 : index
    affine.for %arg3 = 0 to 64 {
      %1 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst_0) -> (bf16) {
        %3 = affine.load %arg0[%arg3, %arg4] : memref<64x128xbf16>
        %4 = arith.maxnumf %arg5, %3 : bf16
        affine.yield %4 : bf16
      }
      %2 = affine.for %arg4 = 0 to 128 iter_args(%arg5 = %cst) -> (bf16) {
        %3 = affine.load %arg0[%arg3, %arg4] : memref<64x128xbf16>
        %4 = arith.subf %3, %1 : bf16
        %5 = math.exp %4 : bf16
        %6 = arith.addf %arg5, %5 : bf16
        affine.store %5, %alloc[%arg3, %arg4] : memref<64x128xbf16>
        affine.yield %6 : bf16
      }
      affine.for %arg4 = 0 to 128 {
        %3 = affine.load %alloc[%arg3, %arg4] : memref<64x128xbf16>
        %4 = arith.divf %3, %2 : bf16
        affine.store %4, %alloc[%arg3, %arg4] : memref<64x128xbf16>
      }
    }
    %0 = call @Gemm_0(%alloc, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<64x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<64x64xbf16>) -> memref<64x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", stationary_kind = "InputStationary", tile_size = array<i64: 32, 64, 2, 8>} : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
}


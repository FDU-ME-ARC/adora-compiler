module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "linear_bf16"} {
  func.func @main_graph(%arg0: memref<64x128xbf16> {onnx.name = "input"}, %arg1: memref<128x64xbf16> {onnx.name = "weight"}, %arg2: memref<64x64xbf16> {onnx.name = "bias"}) -> (memref<64x64xbf16> {onnx.name = "output"}) attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<64x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<64x64xbf16>) -> memref<64x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) : (memref<64x128xbf16>, memref<128x64xbf16>, memref<64x64xbf16>) -> memref<64x64xbf16>
    return %0 : memref<64x64xbf16>
  }
}


module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "softmax_bmm_bf16"} {
  func.func @main_graph(%arg0: tensor<64x128xbf16> {onnx.name = "input"}, %arg1: tensor<128x64xbf16> {onnx.name = "weight"}, %arg2: tensor<64x64xbf16> {onnx.name = "bias"}) -> (tensor<64x64xbf16> {onnx.name = "output"}) {
    %0 = "onnx.Softmax"(%arg0) {axis = -1 : si64, onnx_node_name = "/Softmax"} : (tensor<64x128xbf16>) -> tensor<64x128xbf16>
    %1 = "onnx.Gemm"(%0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/Add-/MatMul_0", transA = 0 : si64, transB = 0 : si64} : (tensor<64x128xbf16>, tensor<128x64xbf16>, tensor<64x64xbf16>) -> tensor<64x64xbf16>
    return %1 : tensor<64x64xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

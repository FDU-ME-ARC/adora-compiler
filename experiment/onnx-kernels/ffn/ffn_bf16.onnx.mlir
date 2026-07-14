module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "ffn_bf16"} {
  func.func @main_graph(%arg0: tensor<64x128xbf16> {onnx.name = "input"}, %arg1: tensor<128x256xbf16> {onnx.name = "weight1"}, %arg2: tensor<64x256xbf16> {onnx.name = "bias1"}, %arg3: tensor<256x128xbf16> {onnx.name = "weight2"}, %arg4: tensor<64x128xbf16> {onnx.name = "bias2"}) -> (tensor<64x128xbf16> {onnx.name = "output"}) {
    %0 = "onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/Add_1-/MatMul_1_0", transA = 0 : si64, transB = 0 : si64} : (tensor<64x128xbf16>, tensor<128x256xbf16>, tensor<64x256xbf16>) -> tensor<64x256xbf16>
    %1 = "onnx.Relu"(%0) {onnx_node_name = "/Relu"} : (tensor<64x256xbf16>) -> tensor<64x256xbf16>
    %2 = "onnx.Gemm"(%1, %arg3, %arg4) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/Add_2-/MatMul_2_1", transA = 0 : si64, transB = 0 : si64} : (tensor<64x256xbf16>, tensor<256x128xbf16>, tensor<64x128xbf16>) -> tensor<64x128xbf16>
    return %2 : tensor<64x128xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

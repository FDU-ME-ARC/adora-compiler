module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "linear_bf16"} {
  func.func @main_graph(%arg0: tensor<?x128xbf16> {onnx.dim_params = "0:batch_size", onnx.name = "input"}) -> (tensor<?x64xbf16> {onnx.dim_params = "0:batch_size", onnx.name = "output"}) {
    %0 = onnx.Constant dense_resource<__elided__> : tensor<64x128xbf16>
    %1 = onnx.Constant dense_resource<__elided__> : tensor<64xbf16>
    %2 = "onnx.Gemm"(%arg0, %0, %1) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, onnx_node_name = "/Gemm", transA = 0 : si64, transB = 1 : si64} : (tensor<?x128xbf16>, tensor<64x128xbf16>, tensor<64xbf16>) -> tensor<?x64xbf16>
    return %2 : tensor<?x64xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}


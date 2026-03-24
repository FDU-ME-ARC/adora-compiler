module {
  func.func @main_graph(%arg0: tensor<1x3x32x32xf32>) -> tensor<1x16x32x32xf32> {
    // 1. 权重: [OutChannel=16, InChannel=3, K=3, K=3]
    %W = onnx.Constant dense<0.5> : tensor<16x3x3x3xf32>
    
    // 2. Bias: [OutChannel=16]
    %B = onnx.Constant dense<0.1> : tensor<16xf32>
    
    // 3. Conv2D 操作
    // kernel_shape=[3,3], pads=[1,1,1,1] (上下左右各填补1), strides=[1,1]
    %0 = "onnx.Conv"(%arg0, %W, %B) {
      auto_pad = "NOTSET", 
      dilations = [1, 1], 
      group = 1 : si64, 
      kernel_shape = [3, 3], 
      pads = [1, 1, 1, 1], 
      strides = [1, 1]
    } : (tensor<1x3x32x32xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x32x32xf32>
    
    return %0 : tensor<1x16x32x32xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}
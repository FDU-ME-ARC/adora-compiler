// module {
//   func.func @main_graph(%arg0: tensor<1x3x32x32xbf16>, %arg1: tensor<16x3x3x3xbf16>, %arg2: tensor<16xbf16>) -> tensor<1x16x32x32xbf16> {
    
//     %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
//       auto_pad = "NOTSET", 
//       dilations = [1, 1], 
//       group = 1 : si64, 
//       kernel_shape = [3, 3], 
//       pads = [1, 1, 1, 1], 
//       strides = [1, 1]
//     } : (tensor<1x3x32x32xbf16>, tensor<16x3x3x3xbf16>, tensor<16xbf16>) -> tensor<1x16x32x32xbf16>
    
//     return %0 : tensor<1x16x32x32xbf16>
//   }
//   "onnx.EntryPoint"() {func = @main_graph} : () -> ()
// }

module {
  func.func @main_graph(%arg0: tensor<1x1x16x16xbf16>, %arg1: tensor<16x1x3x3xbf16>, %arg2: tensor<16xbf16>) -> tensor<1x16x16x16xbf16> {
    
    %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
      auto_pad = "NOTSET", 
      dilations = [1, 1], 
      group = 1 : si64, 
      kernel_shape = [3, 3], 
      pads = [1, 1, 1, 1], 
      strides = [1, 1]
    } : (tensor<1x1x16x16xbf16>, tensor<16x1x3x3xbf16>, tensor<16xbf16>) -> tensor<1x16x16x16xbf16>
    
    return %0 : tensor<1x16x16x16xbf16>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

// module {
//   func.func @main_graph(%arg0: tensor<1x1x16x16xf32>, %arg1: tensor<16x1x3x3xf32>, %arg2: tensor<16xf32>) -> tensor<1x16x16x16xf32> {
    
//     %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {
//       auto_pad = "NOTSET", 
//       dilations = [1, 1], 
//       group = 1 : si64, 
//       kernel_shape = [3, 3], 
//       pads = [1, 1, 1, 1], 
//       strides = [1, 1]
//     } : (tensor<1x1x16x16xf32>, tensor<16x1x3x3xf32>, tensor<16xf32>) -> tensor<1x16x16x16xf32>
    
//     return %0 : tensor<1x16x16x16xf32>
//   }
//   "onnx.EntryPoint"() {func = @main_graph} : () -> ()
// }
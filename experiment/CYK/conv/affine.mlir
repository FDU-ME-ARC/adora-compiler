module {
  func.func @main_graph(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Conv_0(%arg0, %arg1, %arg2) : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
  func.func @Conv_0(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Conv"(%arg0, %arg1, %arg2) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
}


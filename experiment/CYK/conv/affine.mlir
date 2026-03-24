module {
  memref.global constant @constant_1 : memref<16xf32> = dense<1.000000e-01>
  memref.global constant @constant_0 : memref<16x3x3x3xf32> = dense<5.000000e-01>
  // func.func @main_graph(%arg0: memref<1x3x32x32xf32>) -> memref<1x16x32x32xf32> attributes {llvm.emit_c_interface, onnxEntryPoint} {
  //   %0 = memref.get_global @constant_0 : memref<16x3x3x3xf32>
  //   %1 = memref.get_global @constant_1 : memref<16xf32>
  //   %2 = call @Conv_0(%arg0, %0, %1) : (memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>) -> memref<1x16x32x32xf32>
  //   return %2 : memref<1x16x32x32xf32>
  // }
  func.func @Conv_0(%arg0: memref<1x3x32x32xf32>, %arg1: memref<16x3x3x3xf32>, %arg2: memref<16xf32>) -> memref<1x16x32x32xf32> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Conv"(%arg0, %arg1, %arg2) <{auto_pad = "NOTSET", dilations = [1, 1], group = 1 : i64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> : (memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>) -> memref<1x16x32x32xf32>
    return %0 : memref<1x16x32x32xf32>
  }
}


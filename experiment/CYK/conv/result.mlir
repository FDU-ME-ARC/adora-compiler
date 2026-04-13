module {
  func.func @main_graph(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = call @Conv_0(%arg0, %arg1, %arg2) : (memref<1x1x16x16xbf16>, memref<16x1x3x3xbf16>, memref<16xbf16>) -> memref<1x16x16x16xbf16>
    return %0 : memref<1x16x16x16xbf16>
  }
  func.func @Conv_0(%arg0: memref<1x1x16x16xbf16>, %arg1: memref<16x1x3x3xbf16>, %arg2: memref<16xbf16>) -> memref<1x16x16x16xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c0_0 = arith.constant 0 : index
    %alloc = memref.alloc() : memref<1x16x16x16xbf16>
    %0 = ADORA.BlockLoad %alloc [0, 0, 0, 0] : memref<1x16x16x16xbf16> -> memref<1x16x16x16xbf16>  {Id = "0", KernelName = "Conv_0"}
    %1 = ADORA.BlockLoad %arg0 [0, 0, 0, 0] : memref<1x1x16x16xbf16> -> memref<1x1x16x16xbf16>  {Id = "1", KernelName = "Conv_0"}
    %2 = ADORA.BlockLoad %arg1 [0, 0, 0, 0] : memref<16x1x3x3xbf16> -> memref<16x1x3x3xbf16>  {Id = "2", KernelName = "Conv_0"}
    %3 = ADORA.LocalMemAlloc memref<1x16x16x16xbf16>  {Id = "3", KernelName = "Conv_0"}
    ADORA.kernel {
      affine.for %arg3 = 0 to 16 {
        affine.for %arg4 = 0 to 16 {
          affine.for %arg5 = 0 to 16 {
            %4 = affine.load %0[0, %arg3, %arg4, %arg5] : memref<1x16x16x16xbf16>
            %5 = affine.load %1[0, 0, -1, %arg5 - 1] : memref<1x1x16x16xbf16>
            %6 = affine.load %1[0, 0, -1, %arg5] : memref<1x1x16x16xbf16>
            %7 = affine.load %1[0, 0, -1, %arg5 + 1] : memref<1x1x16x16xbf16>
            %8 = affine.load %2[%arg3, 0, 0, 0] : memref<16x1x3x3xbf16>
            %9 = arith.mulf %5, %8 : bf16
            %10 = arith.addf %4, %9 : bf16
            %c1 = arith.constant 1 : index
            %11 = affine.load %2[%arg3, 0, 0, 1] : memref<16x1x3x3xbf16>
            %12 = arith.mulf %6, %11 : bf16
            %13 = arith.addf %10, %12 : bf16
            %c2 = arith.constant 2 : index
            %14 = affine.load %2[%arg3, 0, 0, 2] : memref<16x1x3x3xbf16>
            %15 = arith.mulf %7, %14 : bf16
            %16 = arith.addf %13, %15 : bf16
            %c1_1 = arith.constant 1 : index
            %17 = affine.load %2[%arg3, 0, 1, 0] : memref<16x1x3x3xbf16>
            %18 = arith.mulf %5, %17 : bf16
            %19 = arith.addf %16, %18 : bf16
            %c1_2 = arith.constant 1 : index
            %20 = affine.load %2[%arg3, 0, 1, 1] : memref<16x1x3x3xbf16>
            %21 = arith.mulf %6, %20 : bf16
            %22 = arith.addf %19, %21 : bf16
            %c2_3 = arith.constant 2 : index
            %23 = affine.load %2[%arg3, 0, 1, 2] : memref<16x1x3x3xbf16>
            %24 = arith.mulf %7, %23 : bf16
            %25 = arith.addf %22, %24 : bf16
            %c2_4 = arith.constant 2 : index
            %26 = affine.load %2[%arg3, 0, 2, 0] : memref<16x1x3x3xbf16>
            %27 = arith.mulf %5, %26 : bf16
            %28 = arith.addf %25, %27 : bf16
            %c1_5 = arith.constant 1 : index
            %29 = affine.load %2[%arg3, 0, 2, 1] : memref<16x1x3x3xbf16>
            %30 = arith.mulf %6, %29 : bf16
            %31 = arith.addf %28, %30 : bf16
            %c2_6 = arith.constant 2 : index
            %32 = affine.load %2[%arg3, 0, 2, 2] : memref<16x1x3x3xbf16>
            %33 = arith.mulf %7, %32 : bf16
            %34 = arith.addf %31, %33 : bf16
            affine.store %34, %3[0, %arg3, %arg4, %arg5] : memref<1x16x16x16xbf16>
          }
        }
      }
      ADORA.terminator
    } {KernelName = "Conv_0"}
    ADORA.BlockStore %3, %alloc [0, 0, 0, 0] : memref<1x16x16x16xbf16> -> memref<1x16x16x16xbf16>  {Id = "3", KernelName = "Conv_0"}
    return %alloc : memref<1x16x16x16xbf16>
  }
}


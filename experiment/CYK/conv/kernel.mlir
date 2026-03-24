module {
  memref.global constant @constant_1 : memref<16xf32> = dense<1.000000e-01>
  memref.global constant @constant_0 : memref<16x3x3x3xf32> = dense<5.000000e-01>
  func.func @main_graph(%arg0: memref<1x3x32x32xf32>) -> memref<1x16x32x32xf32> attributes {llvm.emit_c_interface, onnxEntryPoint} {
    %0 = memref.get_global @constant_0 : memref<16x3x3x3xf32>
    %1 = memref.get_global @constant_1 : memref<16xf32>
    %2 = call @Conv_0(%arg0, %0, %1) : (memref<1x3x32x32xf32>, memref<16x3x3x3xf32>, memref<16xf32>) -> memref<1x16x32x32xf32>
    return %2 : memref<1x16x32x32xf32>
  }
  func.func @Conv_0(%arg0: memref<1x3x32x32xf32>, %arg1: memref<16x3x3x3xf32>, %arg2: memref<16xf32>) -> memref<1x16x32x32xf32> attributes {adora_kernel, llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<1x3x34x34xf32>
    %cst = arith.constant 0.000000e+00 : f32
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 34 {
          affine.for %arg6 = 0 to 34 {
            affine.store %cst, %alloc[%arg3, %arg4, %arg5, %arg6] : memref<1x3x34x34xf32>
          }
        }
      }
    }
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 3 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 32 {
            %0 = affine.load %arg0[%arg3, %arg4, %arg5, %arg6] : memref<1x3x32x32xf32>
            affine.store %0, %alloc[%arg3, %arg4, %arg5 + 1, %arg6 + 1] : memref<1x3x34x34xf32>
          }
        }
      }
    }
    %alloc_0 = memref.alloc() : memref<1x16x32x32xf32>
    affine.for %arg3 = 0 to 1 {
      affine.for %arg4 = 0 to 16 {
        affine.for %arg5 = 0 to 32 {
          affine.for %arg6 = 0 to 32 {
            %0 = affine.load %arg2[%arg4] : memref<16xf32>
            affine.store %0, %alloc_0[%arg3, %arg4, %arg5, %arg6] : memref<1x16x32x32xf32>
          }
        }
      }
    }
    affine.for %arg3 = 0 to 27 step 9 {
      affine.for %arg4 = 0 to 1024 step 256 {
        affine.for %arg5 = 0 to 16 step 16 {
          %0 = ADORA.BlockLoad %alloc [%arg4 floordiv 1024, %arg3 floordiv 9, (%arg4 mod 1024) floordiv 32 + (%arg3 mod 9) floordiv 3, %arg4 mod 32 + %arg3 mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
          %1 = ADORA.BlockLoad %alloc [%arg4 floordiv 1024, (%arg3 + 4) floordiv 9, (%arg4 mod 1024) floordiv 32 + ((%arg3 + 4) mod 9) floordiv 3, %arg4 mod 32 + (%arg3 + 4) mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
          %2 = ADORA.BlockLoad %alloc [%arg4 floordiv 1024, (%arg3 + 8) floordiv 9, (%arg4 mod 1024) floordiv 32 + ((%arg3 + 8) mod 9) floordiv 3, %arg4 mod 32 + (%arg3 + 8) mod 3] : memref<1x3x34x34xf32> -> memref<64x1xf32>  {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
          %3 = ADORA.BlockLoad %alloc [(%arg4 + 1) floordiv 1024, %arg3 floordiv 9, ((%arg4 + 1) mod 1024) floordiv 32 + (%arg3 mod 9) floordiv 3, (%arg4 + 1) mod 32 + %arg3 mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
          %4 = ADORA.BlockLoad %alloc [(%arg4 + 1) floordiv 1024, (%arg3 + 4) floordiv 9, ((%arg4 + 1) mod 1024) floordiv 32 + ((%arg3 + 4) mod 9) floordiv 3, (%arg4 + 1) mod 32 + (%arg3 + 4) mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
          %5 = ADORA.BlockLoad %alloc [(%arg4 + 1) floordiv 1024, (%arg3 + 8) floordiv 9, ((%arg4 + 1) mod 1024) floordiv 32 + ((%arg3 + 8) mod 9) floordiv 3, (%arg4 + 1) mod 32 + (%arg3 + 8) mod 3] : memref<1x3x34x34xf32> -> memref<64x1xf32>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
          %6 = ADORA.BlockLoad %alloc [(%arg4 + 2) floordiv 1024, %arg3 floordiv 9, ((%arg4 + 2) mod 1024) floordiv 32 + (%arg3 mod 9) floordiv 3, (%arg4 + 2) mod 32 + %arg3 mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
          %7 = ADORA.BlockLoad %alloc [(%arg4 + 2) floordiv 1024, (%arg3 + 4) floordiv 9, ((%arg4 + 2) mod 1024) floordiv 32 + ((%arg3 + 4) mod 9) floordiv 3, (%arg4 + 2) mod 32 + (%arg3 + 4) mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
          %8 = ADORA.BlockLoad %alloc [(%arg4 + 2) floordiv 1024, (%arg3 + 8) floordiv 9, ((%arg4 + 2) mod 1024) floordiv 32 + ((%arg3 + 8) mod 9) floordiv 3, (%arg4 + 2) mod 32 + (%arg3 + 8) mod 3] : memref<1x3x34x34xf32> -> memref<64x1xf32>  {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
          %9 = ADORA.BlockLoad %alloc [(%arg4 + 3) floordiv 1024, %arg3 floordiv 9, ((%arg4 + 3) mod 1024) floordiv 32 + (%arg3 mod 9) floordiv 3, (%arg4 + 3) mod 32 + %arg3 mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          %10 = ADORA.BlockLoad %alloc [(%arg4 + 3) floordiv 1024, (%arg3 + 4) floordiv 9, ((%arg4 + 3) mod 1024) floordiv 32 + ((%arg3 + 4) mod 9) floordiv 3, (%arg4 + 3) mod 32 + (%arg3 + 4) mod 3] : memref<1x3x34x34xf32> -> memref<64x4xf32>  {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
          %11 = ADORA.BlockLoad %alloc [(%arg4 + 3) floordiv 1024, (%arg3 + 8) floordiv 9, ((%arg4 + 3) mod 1024) floordiv 32 + ((%arg3 + 8) mod 9) floordiv 3, (%arg4 + 3) mod 32 + (%arg3 + 8) mod 3] : memref<1x3x34x34xf32> -> memref<64x1xf32>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          %12 = ADORA.BlockLoad %arg1 [%arg5, %arg3 floordiv 9, (%arg3 mod 9) floordiv 3, %arg3 mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
          %13 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 1) floordiv 9, ((%arg3 + 1) mod 9) floordiv 3, (%arg3 + 1) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          %14 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 2) floordiv 9, ((%arg3 + 2) mod 9) floordiv 3, (%arg3 + 2) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
          %15 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 3) floordiv 9, ((%arg3 + 3) mod 9) floordiv 3, %arg3 mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
          %16 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 4) floordiv 9, ((%arg3 + 4) mod 9) floordiv 3, (%arg3 + 4) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "16", KernelName = "GEMMIS", Pingpong}
          %17 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 5) floordiv 9, ((%arg3 + 5) mod 9) floordiv 3, (%arg3 + 5) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "17", KernelName = "GEMMIS", Pingpong}
          %18 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 6) floordiv 9, ((%arg3 + 6) mod 9) floordiv 3, %arg3 mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "18", KernelName = "GEMMIS", Pingpong}
          %19 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 7) floordiv 9, ((%arg3 + 7) mod 9) floordiv 3, (%arg3 + 7) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "19", KernelName = "GEMMIS", Pingpong}
          %20 = ADORA.BlockLoad %arg1 [%arg5, (%arg3 + 8) floordiv 9, ((%arg3 + 8) mod 9) floordiv 3, (%arg3 + 8) mod 3] : memref<16x3x3x3xf32> -> memref<1x16xf32>  {ADORAGemm, Id = "20", KernelName = "GEMMIS", Pingpong}
          %21 = ADORA.BlockLoad %alloc_0 [%arg4 floordiv 1024, %arg5, (%arg4 mod 1024) floordiv 32, %arg4 mod 32] : memref<1x16x32x32xf32> -> memref<64x16xf32>  {ADORAGemm, Id = "21", KernelName = "GEMMIS", Pingpong}
          %22 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "22", KernelName = "GEMMIS", Pingpong}
          %23 = ADORA.BlockLoad %alloc_0 [(%arg4 + 1) floordiv 1024, %arg5, ((%arg4 + 1) mod 1024) floordiv 32, (%arg4 + 1) mod 32] : memref<1x16x32x32xf32> -> memref<64x16xf32>  {ADORAGemm, Id = "23", KernelName = "GEMMIS", Pingpong}
          %24 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "24", KernelName = "GEMMIS", Pingpong}
          %25 = ADORA.BlockLoad %alloc_0 [(%arg4 + 2) floordiv 1024, %arg5, ((%arg4 + 2) mod 1024) floordiv 32, (%arg4 + 2) mod 32] : memref<1x16x32x32xf32> -> memref<64x16xf32>  {ADORAGemm, Id = "25", KernelName = "GEMMIS", Pingpong}
          %26 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "26", KernelName = "GEMMIS", Pingpong}
          %27 = ADORA.BlockLoad %alloc_0 [(%arg4 + 3) floordiv 1024, %arg5, ((%arg4 + 3) mod 1024) floordiv 32, (%arg4 + 3) mod 32] : memref<1x16x32x32xf32> -> memref<64x16xf32>  {ADORAGemm, Id = "27", KernelName = "GEMMIS", Pingpong}
          %28 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "28", KernelName = "GEMMIS", Pingpong}
          ADORA.kernel {
            affine.for %arg6 = 0 to 64 {
              affine.for %arg7 = 0 to 16 {
                %29 = affine.vector_load %0[%arg6, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %30:4 = ADORA.deinterleaver %29 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %31 = affine.vector_load %1[%arg6, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %32:4 = ADORA.deinterleaver %31 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %33 = affine.load %2[%arg6, 8] {ADORAGemm, Pingpong} : memref<64x1xf32>
                %34 = affine.vector_load %3[%arg6 + 1, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %35:4 = ADORA.deinterleaver %34 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %36 = affine.vector_load %4[%arg6 + 1, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %37:4 = ADORA.deinterleaver %36 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %38 = affine.load %5[%arg6 + 1, 8] {ADORAGemm, Pingpong} : memref<64x1xf32>
                %39 = affine.vector_load %6[%arg6 + 2, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %40:4 = ADORA.deinterleaver %39 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %41 = affine.vector_load %7[%arg6 + 2, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %42:4 = ADORA.deinterleaver %41 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %43 = affine.load %8[%arg6 + 2, 8] {ADORAGemm, Pingpong} : memref<64x1xf32>
                %44 = affine.vector_load %9[%arg6 + 3, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %45:4 = ADORA.deinterleaver %44 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %46 = affine.vector_load %10[%arg6 + 3, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
                %47:4 = ADORA.deinterleaver %46 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
                %48 = affine.load %11[%arg6 + 3, 8] {ADORAGemm, Pingpong} : memref<64x1xf32>
                affine.for %arg8 = 0 to 16 {
                  %49 = affine.load %12[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %50 = arith.mulf %30#0, %49 {ADORAGemm} : f32
                  %51 = affine.load %13[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %52 = arith.mulf %30#1, %51 {ADORAGemm} : f32
                  %53 = arith.addf %52, %50 {ADORAGemm} : f32
                  %54 = affine.load %14[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %55 = arith.mulf %30#2, %54 {ADORAGemm} : f32
                  %56 = arith.addf %55, %53 {ADORAGemm} : f32
                  %57 = affine.load %15[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %58 = arith.mulf %30#3, %57 {ADORAGemm} : f32
                  %59 = arith.addf %58, %56 {ADORAGemm} : f32
                  %60 = affine.load %16[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %61 = arith.mulf %32#0, %60 {ADORAGemm} : f32
                  %62 = arith.addf %61, %59 {ADORAGemm} : f32
                  %63 = affine.load %17[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %64 = arith.mulf %32#1, %63 {ADORAGemm} : f32
                  %65 = arith.addf %64, %62 {ADORAGemm} : f32
                  %66 = affine.load %18[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %67 = arith.mulf %32#2, %66 {ADORAGemm} : f32
                  %68 = arith.addf %67, %65 {ADORAGemm} : f32
                  %69 = affine.load %19[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %70 = arith.mulf %32#3, %69 {ADORAGemm} : f32
                  %71 = arith.addf %70, %68 {ADORAGemm} : f32
                  %72 = affine.load %20[0, %arg8] {ADORAGemm, Pingpong} : memref<1x16xf32>
                  %73 = arith.mulf %33, %72 {ADORAGemm} : f32
                  %74 = arith.addf %73, %71 {ADORAGemm} : f32
                  %75 = arith.mulf %35#0, %49 {ADORAGemm} : f32
                  %76 = arith.mulf %35#1, %51 {ADORAGemm} : f32
                  %77 = arith.addf %76, %75 {ADORAGemm} : f32
                  %78 = arith.mulf %35#2, %54 {ADORAGemm} : f32
                  %79 = arith.addf %78, %77 {ADORAGemm} : f32
                  %80 = arith.mulf %35#3, %57 {ADORAGemm} : f32
                  %81 = arith.addf %80, %79 {ADORAGemm} : f32
                  %82 = arith.mulf %37#0, %60 {ADORAGemm} : f32
                  %83 = arith.addf %82, %81 {ADORAGemm} : f32
                  %84 = arith.mulf %37#1, %63 {ADORAGemm} : f32
                  %85 = arith.addf %84, %83 {ADORAGemm} : f32
                  %86 = arith.mulf %37#2, %66 {ADORAGemm} : f32
                  %87 = arith.addf %86, %85 {ADORAGemm} : f32
                  %88 = arith.mulf %37#3, %69 {ADORAGemm} : f32
                  %89 = arith.addf %88, %87 {ADORAGemm} : f32
                  %90 = arith.mulf %38, %72 {ADORAGemm} : f32
                  %91 = arith.addf %90, %89 {ADORAGemm} : f32
                  %92 = arith.mulf %40#0, %49 {ADORAGemm} : f32
                  %93 = arith.mulf %40#1, %51 {ADORAGemm} : f32
                  %94 = arith.addf %93, %92 {ADORAGemm} : f32
                  %95 = arith.mulf %40#2, %54 {ADORAGemm} : f32
                  %96 = arith.addf %95, %94 {ADORAGemm} : f32
                  %97 = arith.mulf %40#3, %57 {ADORAGemm} : f32
                  %98 = arith.addf %97, %96 {ADORAGemm} : f32
                  %99 = arith.mulf %42#0, %60 {ADORAGemm} : f32
                  %100 = arith.addf %99, %98 {ADORAGemm} : f32
                  %101 = arith.mulf %42#1, %63 {ADORAGemm} : f32
                  %102 = arith.addf %101, %100 {ADORAGemm} : f32
                  %103 = arith.mulf %42#2, %66 {ADORAGemm} : f32
                  %104 = arith.addf %103, %102 {ADORAGemm} : f32
                  %105 = arith.mulf %42#3, %69 {ADORAGemm} : f32
                  %106 = arith.addf %105, %104 {ADORAGemm} : f32
                  %107 = arith.mulf %43, %72 {ADORAGemm} : f32
                  %108 = arith.addf %107, %106 {ADORAGemm} : f32
                  %109 = arith.mulf %45#0, %49 {ADORAGemm} : f32
                  %110 = arith.mulf %45#1, %51 {ADORAGemm} : f32
                  %111 = arith.addf %110, %109 {ADORAGemm} : f32
                  %112 = arith.mulf %45#2, %54 {ADORAGemm} : f32
                  %113 = arith.addf %112, %111 {ADORAGemm} : f32
                  %114 = arith.mulf %45#3, %57 {ADORAGemm} : f32
                  %115 = arith.addf %114, %113 {ADORAGemm} : f32
                  %116 = arith.mulf %47#0, %60 {ADORAGemm} : f32
                  %117 = arith.addf %116, %115 {ADORAGemm} : f32
                  %118 = arith.mulf %47#1, %63 {ADORAGemm} : f32
                  %119 = arith.addf %118, %117 {ADORAGemm} : f32
                  %120 = arith.mulf %47#2, %66 {ADORAGemm} : f32
                  %121 = arith.addf %120, %119 {ADORAGemm} : f32
                  %122 = arith.mulf %47#3, %69 {ADORAGemm} : f32
                  %123 = arith.addf %122, %121 {ADORAGemm} : f32
                  %124 = arith.mulf %48, %72 {ADORAGemm} : f32
                  %125 = arith.addf %124, %123 {ADORAGemm} : f32
                  %126 = affine.load %21[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %127 = arith.addf %74, %126 {ADORAGemm} : f32
                  affine.store %127, %22[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %128 = affine.load %23[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %129 = arith.addf %91, %128 {ADORAGemm} : f32
                  affine.store %129, %24[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %130 = affine.load %25[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %131 = arith.addf %108, %130 {ADORAGemm} : f32
                  affine.store %131, %26[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %132 = affine.load %27[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                  %133 = arith.addf %125, %132 {ADORAGemm} : f32
                  affine.store %133, %28[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<64x16xf32>
                } {ADORAGemm}
              } {ADORAGemm}
            } {ADORAGemm}
            ADORA.terminator {ADORAGemm}
          } {ADORAGemm, KernelName = "GEMMIS"}
          ADORA.BlockStore %22, %alloc_0 [%arg4 floordiv 1024, %arg5, (%arg4 mod 1024) floordiv 32, %arg4 mod 32] : memref<64x16xf32> -> memref<1x16x32x32xf32>  {ADORAGemm, Id = "22", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %24, %alloc_0 [(%arg4 + 1) floordiv 1024, %arg5, ((%arg4 + 1) mod 1024) floordiv 32, (%arg4 + 1) mod 32] : memref<64x16xf32> -> memref<1x16x32x32xf32>  {ADORAGemm, Id = "24", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %26, %alloc_0 [(%arg4 + 2) floordiv 1024, %arg5, ((%arg4 + 2) mod 1024) floordiv 32, (%arg4 + 2) mod 32] : memref<64x16xf32> -> memref<1x16x32x32xf32>  {ADORAGemm, Id = "26", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %28, %alloc_0 [(%arg4 + 3) floordiv 1024, %arg5, ((%arg4 + 3) mod 1024) floordiv 32, (%arg4 + 3) mod 32] : memref<64x16xf32> -> memref<1x16x32x32xf32>  {ADORAGemm, Id = "28", KernelName = "GEMMIS", Pingpong}
        } {ADORAGemm}
      } {ADORAGemm}
    } {ADORAGemm}
    return %alloc_0 : memref<1x16x32x32xf32>
  }
}


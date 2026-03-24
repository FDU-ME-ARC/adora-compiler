module {
  func.func @matmul_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {llvm.emit_c_interface} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    return %0 : memref<32x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %alloc = memref.alloc() : memref<32x64xbf16>
    memref.copy %arg2, %alloc : memref<32x64xbf16> to memref<32x64xbf16>
    affine.for %arg3 = 0 to 128 step 8 {
      affine.for %arg4 = 0 to 32 step 32 {
        affine.for %arg5 = 0 to 64 step 64 {
          %0 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
          %1 = ADORA.BlockLoad %arg0 [%arg4, %arg3 + 4] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
          %2 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
          %3 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3 + 4] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
          %4 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
          %5 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3 + 4] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
          %6 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
          %7 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3 + 4] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
          %8 = ADORA.BlockLoad %arg1 [%arg3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
          %9 = ADORA.BlockLoad %arg1 [%arg3 + 1, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          %10 = ADORA.BlockLoad %arg1 [%arg3 + 2, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
          %11 = ADORA.BlockLoad %arg1 [%arg3 + 3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          %12 = ADORA.BlockLoad %arg1 [%arg3 + 4, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
          %13 = ADORA.BlockLoad %arg1 [%arg3 + 5, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          %14 = ADORA.BlockLoad %arg1 [%arg3 + 6, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
          %15 = ADORA.BlockLoad %arg1 [%arg3 + 7, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
          %16 = ADORA.BlockLoad %alloc [%arg4, %arg5] : memref<32x64xbf16> -> memref<8x64xbf16> , stride [4, 1] {ADORAGemm, Id = "16", KernelName = "GEMMIS", Pingpong}
          %17 = ADORA.LocalMemAlloc memref<8x64xbf16>  {ADORAGemm, Id = "17", KernelName = "GEMMIS", Pingpong}
          %18 = ADORA.BlockLoad %alloc [%arg4 + 1, %arg5] : memref<32x64xbf16> -> memref<8x64xbf16> , stride [4, 1] {ADORAGemm, Id = "18", KernelName = "GEMMIS", Pingpong}
          %19 = ADORA.LocalMemAlloc memref<8x64xbf16>  {ADORAGemm, Id = "19", KernelName = "GEMMIS", Pingpong}
          %20 = ADORA.BlockLoad %alloc [%arg4 + 2, %arg5] : memref<32x64xbf16> -> memref<8x64xbf16> , stride [4, 1] {ADORAGemm, Id = "20", KernelName = "GEMMIS", Pingpong}
          %21 = ADORA.LocalMemAlloc memref<8x64xbf16>  {ADORAGemm, Id = "21", KernelName = "GEMMIS", Pingpong}
          %22 = ADORA.BlockLoad %alloc [%arg4 + 3, %arg5] : memref<32x64xbf16> -> memref<8x64xbf16> , stride [4, 1] {ADORAGemm, Id = "22", KernelName = "GEMMIS", Pingpong}
          %23 = ADORA.LocalMemAlloc memref<8x64xbf16>  {ADORAGemm, Id = "23", KernelName = "GEMMIS", Pingpong}
          ADORA.kernel {
            affine.for %arg6 = 0 to 8 {
              affine.for %arg7 = 0 to 64 {
                %24 = affine.vector_load %0[%arg6, 0] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %25:4 = ADORA.deinterleaver %24 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %26 = affine.vector_load %1[%arg6, 4] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %27:4 = ADORA.deinterleaver %26 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %28 = affine.vector_load %2[%arg6 + 1, 0] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %29:4 = ADORA.deinterleaver %28 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %30 = affine.vector_load %3[%arg6 + 1, 4] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %31:4 = ADORA.deinterleaver %30 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %32 = affine.vector_load %4[%arg6 + 2, 0] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %33:4 = ADORA.deinterleaver %32 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %34 = affine.vector_load %5[%arg6 + 2, 4] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %35:4 = ADORA.deinterleaver %34 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %36 = affine.vector_load %6[%arg6 + 3, 0] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %37:4 = ADORA.deinterleaver %36 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                %38 = affine.vector_load %7[%arg6 + 3, 4] {ADORAGemm, Pingpong} : memref<8x4xbf16>, vector<4xbf16>
                %39:4 = ADORA.deinterleaver %38 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
                affine.for %arg8 = 0 to 64 {
                  %40 = affine.load %8[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %41 = arith.mulf %25#0, %40 {ADORAGemm} : bf16
                  %42 = affine.load %9[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %43 = arith.mulf %25#1, %42 {ADORAGemm} : bf16
                  %44 = arith.addf %43, %41 {ADORAGemm} : bf16
                  %45 = affine.load %10[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %46 = arith.mulf %25#2, %45 {ADORAGemm} : bf16
                  %47 = arith.addf %46, %44 {ADORAGemm} : bf16
                  %48 = affine.load %11[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %49 = arith.mulf %25#3, %48 {ADORAGemm} : bf16
                  %50 = arith.addf %49, %47 {ADORAGemm} : bf16
                  %51 = affine.load %12[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %52 = arith.mulf %27#0, %51 {ADORAGemm} : bf16
                  %53 = arith.addf %52, %50 {ADORAGemm} : bf16
                  %54 = affine.load %13[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %55 = arith.mulf %27#1, %54 {ADORAGemm} : bf16
                  %56 = arith.addf %55, %53 {ADORAGemm} : bf16
                  %57 = affine.load %14[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %58 = arith.mulf %27#2, %57 {ADORAGemm} : bf16
                  %59 = arith.addf %58, %56 {ADORAGemm} : bf16
                  %60 = affine.load %15[0, %arg8] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                  %61 = arith.mulf %27#3, %60 {ADORAGemm} : bf16
                  %62 = arith.addf %61, %59 {ADORAGemm} : bf16
                  %63 = arith.mulf %29#0, %40 {ADORAGemm} : bf16
                  %64 = arith.mulf %29#1, %42 {ADORAGemm} : bf16
                  %65 = arith.addf %64, %63 {ADORAGemm} : bf16
                  %66 = arith.mulf %29#2, %45 {ADORAGemm} : bf16
                  %67 = arith.addf %66, %65 {ADORAGemm} : bf16
                  %68 = arith.mulf %29#3, %48 {ADORAGemm} : bf16
                  %69 = arith.addf %68, %67 {ADORAGemm} : bf16
                  %70 = arith.mulf %31#0, %51 {ADORAGemm} : bf16
                  %71 = arith.addf %70, %69 {ADORAGemm} : bf16
                  %72 = arith.mulf %31#1, %54 {ADORAGemm} : bf16
                  %73 = arith.addf %72, %71 {ADORAGemm} : bf16
                  %74 = arith.mulf %31#2, %57 {ADORAGemm} : bf16
                  %75 = arith.addf %74, %73 {ADORAGemm} : bf16
                  %76 = arith.mulf %31#3, %60 {ADORAGemm} : bf16
                  %77 = arith.addf %76, %75 {ADORAGemm} : bf16
                  %78 = arith.mulf %33#0, %40 {ADORAGemm} : bf16
                  %79 = arith.mulf %33#1, %42 {ADORAGemm} : bf16
                  %80 = arith.addf %79, %78 {ADORAGemm} : bf16
                  %81 = arith.mulf %33#2, %45 {ADORAGemm} : bf16
                  %82 = arith.addf %81, %80 {ADORAGemm} : bf16
                  %83 = arith.mulf %33#3, %48 {ADORAGemm} : bf16
                  %84 = arith.addf %83, %82 {ADORAGemm} : bf16
                  %85 = arith.mulf %35#0, %51 {ADORAGemm} : bf16
                  %86 = arith.addf %85, %84 {ADORAGemm} : bf16
                  %87 = arith.mulf %35#1, %54 {ADORAGemm} : bf16
                  %88 = arith.addf %87, %86 {ADORAGemm} : bf16
                  %89 = arith.mulf %35#2, %57 {ADORAGemm} : bf16
                  %90 = arith.addf %89, %88 {ADORAGemm} : bf16
                  %91 = arith.mulf %35#3, %60 {ADORAGemm} : bf16
                  %92 = arith.addf %91, %90 {ADORAGemm} : bf16
                  %93 = arith.mulf %37#0, %40 {ADORAGemm} : bf16
                  %94 = arith.mulf %37#1, %42 {ADORAGemm} : bf16
                  %95 = arith.addf %94, %93 {ADORAGemm} : bf16
                  %96 = arith.mulf %37#2, %45 {ADORAGemm} : bf16
                  %97 = arith.addf %96, %95 {ADORAGemm} : bf16
                  %98 = arith.mulf %37#3, %48 {ADORAGemm} : bf16
                  %99 = arith.addf %98, %97 {ADORAGemm} : bf16
                  %100 = arith.mulf %39#0, %51 {ADORAGemm} : bf16
                  %101 = arith.addf %100, %99 {ADORAGemm} : bf16
                  %102 = arith.mulf %39#1, %54 {ADORAGemm} : bf16
                  %103 = arith.addf %102, %101 {ADORAGemm} : bf16
                  %104 = arith.mulf %39#2, %57 {ADORAGemm} : bf16
                  %105 = arith.addf %104, %103 {ADORAGemm} : bf16
                  %106 = arith.mulf %39#3, %60 {ADORAGemm} : bf16
                  %107 = arith.addf %106, %105 {ADORAGemm} : bf16
                  %108 = affine.load %16[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %109 = arith.addf %62, %108 {ADORAGemm} : bf16
                  affine.store %109, %17[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %110 = affine.load %18[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %111 = arith.addf %77, %110 {ADORAGemm} : bf16
                  affine.store %111, %19[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %112 = affine.load %20[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %113 = arith.addf %92, %112 {ADORAGemm} : bf16
                  affine.store %113, %21[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %114 = affine.load %22[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                  %115 = arith.addf %107, %114 {ADORAGemm} : bf16
                  affine.store %115, %23[%arg6, %arg8] {ADORAGemm, Pingpong} : memref<8x64xbf16>
                } {ADORAGemm}
              } {ADORAGemm}
            } {ADORAGemm}
            ADORA.terminator {ADORAGemm}
          } {ADORAGemm, KernelName = "GEMMIS"}
          ADORA.BlockStore %17, %alloc [%arg4, %arg5] : memref<8x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "17", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %19, %alloc [%arg4 + 1, %arg5] : memref<8x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "19", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %21, %alloc [%arg4 + 2, %arg5] : memref<8x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "21", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %23, %alloc [%arg4 + 3, %arg5] : memref<8x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "23", KernelName = "GEMMIS", Pingpong}
        } {ADORAGemm}
      } {ADORAGemm}
    } {ADORAGemm}
    return %alloc : memref<32x64xbf16>
  }
}


#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {llvm.emit_c_interface} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    return %0 : memref<32x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", linalg.memoized_indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 8, 64, 4, 8>} : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    memref.copy %arg2, %alloc : memref<32x64xbf16> to memref<32x64xbf16>
    affine.for %arg3 = 0 to 128 step 4 {
      affine.for %arg4 = 0 to 32 step 16 {
        affine.for %arg5 = 0 to 64 step 64 {
          %1 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
          %2 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
          %3 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
          %4 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
          %5 = ADORA.BlockLoad %arg1 [%arg3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
          %6 = ADORA.BlockLoad %arg1 [%arg3 + 1, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
          %7 = ADORA.BlockLoad %arg1 [%arg3 + 2, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
          %8 = ADORA.BlockLoad %arg1 [%arg3 + 3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
          %9 = ADORA.BlockLoad %arg2 [%arg4, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
          %10 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          %11 = ADORA.BlockLoad %arg2 [%arg4 + 1, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
          %12 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          %13 = ADORA.BlockLoad %arg2 [%arg4 + 2, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
          %14 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          %15 = ADORA.BlockLoad %arg2 [%arg4 + 3, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
          %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
          ADORA.kernel {
            affine.for %arg6 = 0 to 4 {
              %17 = affine.vector_load %1[%arg6, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %18:4 = ADORA.deinterleaver %17 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %19 = affine.vector_load %2[%arg6 + 1, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %20:4 = ADORA.deinterleaver %19 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %21 = affine.vector_load %3[%arg6 + 2, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %22:4 = ADORA.deinterleaver %21 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %23 = affine.vector_load %4[%arg6 + 3, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %24:4 = ADORA.deinterleaver %23 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              affine.for %arg7 = 0 to 64 {
                %25 = affine.load %5[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %26 = arith.mulf %18#0, %25 {ADORAGemm} : bf16
                %27 = affine.load %6[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %28 = arith.mulf %18#1, %27 {ADORAGemm} : bf16
                %29 = arith.addf %28, %26 {ADORAGemm} : bf16
                %30 = affine.load %7[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %31 = arith.mulf %18#2, %30 {ADORAGemm} : bf16
                %32 = arith.addf %31, %29 {ADORAGemm} : bf16
                %33 = affine.load %8[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %34 = arith.mulf %18#3, %33 {ADORAGemm} : bf16
                %35 = arith.addf %34, %32 {ADORAGemm} : bf16
                %36 = arith.mulf %20#0, %25 {ADORAGemm} : bf16
                %37 = arith.mulf %20#1, %27 {ADORAGemm} : bf16
                %38 = arith.addf %37, %36 {ADORAGemm} : bf16
                %39 = arith.mulf %20#2, %30 {ADORAGemm} : bf16
                %40 = arith.addf %39, %38 {ADORAGemm} : bf16
                %41 = arith.mulf %20#3, %33 {ADORAGemm} : bf16
                %42 = arith.addf %41, %40 {ADORAGemm} : bf16
                %43 = arith.mulf %22#0, %25 {ADORAGemm} : bf16
                %44 = arith.mulf %22#1, %27 {ADORAGemm} : bf16
                %45 = arith.addf %44, %43 {ADORAGemm} : bf16
                %46 = arith.mulf %22#2, %30 {ADORAGemm} : bf16
                %47 = arith.addf %46, %45 {ADORAGemm} : bf16
                %48 = arith.mulf %22#3, %33 {ADORAGemm} : bf16
                %49 = arith.addf %48, %47 {ADORAGemm} : bf16
                %50 = arith.mulf %24#0, %25 {ADORAGemm} : bf16
                %51 = arith.mulf %24#1, %27 {ADORAGemm} : bf16
                %52 = arith.addf %51, %50 {ADORAGemm} : bf16
                %53 = arith.mulf %24#2, %30 {ADORAGemm} : bf16
                %54 = arith.addf %53, %52 {ADORAGemm} : bf16
                %55 = arith.mulf %24#3, %33 {ADORAGemm} : bf16
                %56 = arith.addf %55, %54 {ADORAGemm} : bf16
                %57 = affine.load %9[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %58 = arith.addf %35, %57 {ADORAGemm} : bf16
                affine.store %58, %10[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %59 = affine.load %11[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %60 = arith.addf %42, %59 {ADORAGemm} : bf16
                affine.store %60, %12[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %61 = affine.load %13[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %62 = arith.addf %49, %61 {ADORAGemm} : bf16
                affine.store %62, %14[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %63 = affine.load %15[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %64 = arith.addf %56, %63 {ADORAGemm} : bf16
                affine.store %64, %16[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              } {ADORAGemm}
            } {ADORAGemm}
            ADORA.terminator {ADORAGemm}
          } {ADORAGemm, KernelName = "GEMMIS"}
          ADORA.BlockStore %10, %arg2 [%arg4, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %12, %arg2 [%arg4 + 1, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %14, %arg2 [%arg4 + 2, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %16, %arg2 [%arg4 + 3, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
        } {ADORAGemm}
      } {ADORAGemm}
    } {ADORAGemm}
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    memref.copy %arg2, %alloc_0 : memref<32x64xbf16> to memref<32x64xbf16>
    affine.for %arg3 = 0 to 128 step 4 {
      affine.for %arg4 = 0 to 32 step 16 {
        affine.for %arg5 = 0 to 64 step 64 {
          %1 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
          %2 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
          %3 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
          %4 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
          %5 = ADORA.BlockLoad %arg1 [%arg3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
          %6 = ADORA.BlockLoad %arg1 [%arg3 + 1, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
          %7 = ADORA.BlockLoad %arg1 [%arg3 + 2, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
          %8 = ADORA.BlockLoad %arg1 [%arg3 + 3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
          %9 = ADORA.BlockLoad %arg2 [%arg4, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
          %10 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          %11 = ADORA.BlockLoad %arg2 [%arg4 + 1, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
          %12 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          %13 = ADORA.BlockLoad %arg2 [%arg4 + 2, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
          %14 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          %15 = ADORA.BlockLoad %arg2 [%arg4 + 3, %arg5] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
          %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
          ADORA.kernel {
            affine.for %arg6 = 0 to 4 {
              %17 = affine.vector_load %1[%arg6, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %18:4 = ADORA.deinterleaver %17 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %19 = affine.vector_load %2[%arg6 + 1, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %20:4 = ADORA.deinterleaver %19 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %21 = affine.vector_load %3[%arg6 + 2, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %22:4 = ADORA.deinterleaver %21 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              %23 = affine.vector_load %4[%arg6 + 3, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
              %24:4 = ADORA.deinterleaver %23 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
              affine.for %arg7 = 0 to 64 {
                %25 = affine.load %5[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %26 = arith.mulf %18#0, %25 {ADORAGemm} : bf16
                %27 = affine.load %6[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %28 = arith.mulf %18#1, %27 {ADORAGemm} : bf16
                %29 = arith.addf %28, %26 {ADORAGemm} : bf16
                %30 = affine.load %7[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %31 = arith.mulf %18#2, %30 {ADORAGemm} : bf16
                %32 = arith.addf %31, %29 {ADORAGemm} : bf16
                %33 = affine.load %8[0, %arg7] {ADORAGemm, Pingpong} : memref<1x64xbf16>
                %34 = arith.mulf %18#3, %33 {ADORAGemm} : bf16
                %35 = arith.addf %34, %32 {ADORAGemm} : bf16
                %36 = arith.mulf %20#0, %25 {ADORAGemm} : bf16
                %37 = arith.mulf %20#1, %27 {ADORAGemm} : bf16
                %38 = arith.addf %37, %36 {ADORAGemm} : bf16
                %39 = arith.mulf %20#2, %30 {ADORAGemm} : bf16
                %40 = arith.addf %39, %38 {ADORAGemm} : bf16
                %41 = arith.mulf %20#3, %33 {ADORAGemm} : bf16
                %42 = arith.addf %41, %40 {ADORAGemm} : bf16
                %43 = arith.mulf %22#0, %25 {ADORAGemm} : bf16
                %44 = arith.mulf %22#1, %27 {ADORAGemm} : bf16
                %45 = arith.addf %44, %43 {ADORAGemm} : bf16
                %46 = arith.mulf %22#2, %30 {ADORAGemm} : bf16
                %47 = arith.addf %46, %45 {ADORAGemm} : bf16
                %48 = arith.mulf %22#3, %33 {ADORAGemm} : bf16
                %49 = arith.addf %48, %47 {ADORAGemm} : bf16
                %50 = arith.mulf %24#0, %25 {ADORAGemm} : bf16
                %51 = arith.mulf %24#1, %27 {ADORAGemm} : bf16
                %52 = arith.addf %51, %50 {ADORAGemm} : bf16
                %53 = arith.mulf %24#2, %30 {ADORAGemm} : bf16
                %54 = arith.addf %53, %52 {ADORAGemm} : bf16
                %55 = arith.mulf %24#3, %33 {ADORAGemm} : bf16
                %56 = arith.addf %55, %54 {ADORAGemm} : bf16
                %57 = affine.load %9[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %58 = arith.addf %35, %57 {ADORAGemm} : bf16
                affine.store %58, %10[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %59 = affine.load %11[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %60 = arith.addf %42, %59 {ADORAGemm} : bf16
                affine.store %60, %12[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %61 = affine.load %13[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %62 = arith.addf %49, %61 {ADORAGemm} : bf16
                affine.store %62, %14[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %63 = affine.load %15[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
                %64 = arith.addf %56, %63 {ADORAGemm} : bf16
                affine.store %64, %16[%arg6, %arg7] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              } {ADORAGemm}
            } {ADORAGemm}
            ADORA.terminator {ADORAGemm}
          } {ADORAGemm, KernelName = "GEMMIS"}
          ADORA.BlockStore %10, %arg2 [%arg4, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %12, %arg2 [%arg4 + 1, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %14, %arg2 [%arg4 + 2, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
          ADORA.BlockStore %16, %arg2 [%arg4 + 3, %arg5] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
        } {ADORAGemm}
      } {ADORAGemm}
    } {ADORAGemm}
    return %alloc_0 : memref<32x64xbf16>
  }
}


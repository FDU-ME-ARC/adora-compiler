#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  // func.func @matmul_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {llvm.emit_c_interface} {
  //   %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
  //   return %0 : memref<32x64xbf16>
  // }
  func.func @Gemm_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", linalg.memoized_indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 8, 64, 4, 8>} : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    %alloc = memref.alloc() : memref<32x64xbf16>
    memref.copy %arg2, %alloc : memref<32x64xbf16> to memref<32x64xbf16>
    affine.for %arg3 = 0 to 128 step 4 {
      affine.for %arg4 = 0 to 32 step 16 {
        %c0 = arith.constant 0 : index
        %1 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
        %2 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
        %3 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
        %4 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
        %5 = ADORA.BlockLoad %arg1 [%arg3, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
        %6 = ADORA.BlockLoad %arg1 [%arg3 + 1, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
        %7 = ADORA.BlockLoad %arg1 [%arg3 + 2, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
        %8 = ADORA.BlockLoad %arg1 [%arg3 + 3, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
        %9 = ADORA.BlockLoad %arg2 [%arg4, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
        %10 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
        %11 = ADORA.BlockLoad %arg2 [%arg4 + 1, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
        %12 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
        %13 = ADORA.BlockLoad %arg2 [%arg4 + 2, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
        %14 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
        %15 = ADORA.BlockLoad %arg2 [%arg4 + 3, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
        %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
        %17 = ADORA.BlockLoad %5 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "0", KernelName = "GEMMIS"}
        %18 = ADORA.BlockLoad %6 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "1", KernelName = "GEMMIS"}
        %19 = ADORA.BlockLoad %7 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "2", KernelName = "GEMMIS"}
        %20 = ADORA.BlockLoad %8 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "3", KernelName = "GEMMIS"}
        %21 = ADORA.BlockLoad %9 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "4", KernelName = "GEMMIS"}
        %22 = ADORA.BlockLoad %11 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "5", KernelName = "GEMMIS"}
        %23 = ADORA.BlockLoad %13 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "6", KernelName = "GEMMIS"}
        %24 = ADORA.BlockLoad %15 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "7", KernelName = "GEMMIS"}
        %25 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "8", KernelName = "GEMMIS"}
        %26 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "9", KernelName = "GEMMIS"}
        %27 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "10", KernelName = "GEMMIS"}
        %28 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "11", KernelName = "GEMMIS"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 4 {
            %29 = affine.vector_load %1[%arg5, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %30:4 = ADORA.deinterleaver %29 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %31 = affine.vector_load %2[%arg5 + 1, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %32:4 = ADORA.deinterleaver %31 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %33 = affine.vector_load %3[%arg5 + 2, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %34:4 = ADORA.deinterleaver %33 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %35 = affine.vector_load %4[%arg5 + 3, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %36:4 = ADORA.deinterleaver %35 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            affine.for %arg6 = 0 to 64 {
              %37 = affine.load %17[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %38 = arith.mulf %30#0, %37 {ADORAGemm} : bf16
              %39 = affine.load %18[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %40 = arith.mulf %30#1, %39 {ADORAGemm} : bf16
              %41 = arith.addf %40, %38 {ADORAGemm} : bf16
              %42 = affine.load %19[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %43 = arith.mulf %30#2, %42 {ADORAGemm} : bf16
              %44 = arith.addf %43, %41 {ADORAGemm} : bf16
              %45 = affine.load %20[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %46 = arith.mulf %30#3, %45 {ADORAGemm} : bf16
              %47 = arith.addf %46, %44 {ADORAGemm} : bf16
              %48 = arith.mulf %32#0, %37 {ADORAGemm} : bf16
              %49 = arith.mulf %32#1, %39 {ADORAGemm} : bf16
              %50 = arith.addf %49, %48 {ADORAGemm} : bf16
              %51 = arith.mulf %32#2, %42 {ADORAGemm} : bf16
              %52 = arith.addf %51, %50 {ADORAGemm} : bf16
              %53 = arith.mulf %32#3, %45 {ADORAGemm} : bf16
              %54 = arith.addf %53, %52 {ADORAGemm} : bf16
              %55 = arith.mulf %34#0, %37 {ADORAGemm} : bf16
              %56 = arith.mulf %34#1, %39 {ADORAGemm} : bf16
              %57 = arith.addf %56, %55 {ADORAGemm} : bf16
              %58 = arith.mulf %34#2, %42 {ADORAGemm} : bf16
              %59 = arith.addf %58, %57 {ADORAGemm} : bf16
              %60 = arith.mulf %34#3, %45 {ADORAGemm} : bf16
              %61 = arith.addf %60, %59 {ADORAGemm} : bf16
              %62 = arith.mulf %36#0, %37 {ADORAGemm} : bf16
              %63 = arith.mulf %36#1, %39 {ADORAGemm} : bf16
              %64 = arith.addf %63, %62 {ADORAGemm} : bf16
              %65 = arith.mulf %36#2, %42 {ADORAGemm} : bf16
              %66 = arith.addf %65, %64 {ADORAGemm} : bf16
              %67 = arith.mulf %36#3, %45 {ADORAGemm} : bf16
              %68 = arith.addf %67, %66 {ADORAGemm} : bf16
              %69 = affine.load %21[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %70 = arith.addf %47, %69 {ADORAGemm} : bf16
              affine.store %70, %27[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %71 = affine.load %22[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %72 = arith.addf %54, %71 {ADORAGemm} : bf16
              affine.store %72, %26[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %73 = affine.load %23[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %74 = arith.addf %61, %73 {ADORAGemm} : bf16
              affine.store %74, %25[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %75 = affine.load %24[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %76 = arith.addf %68, %75 {ADORAGemm} : bf16
              affine.store %76, %28[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
            } {ADORAGemm}
          } {ADORAGemm}
          ADORA.terminator {ADORAGemm}
        } {ADORAGemm, KernelName = "GEMMIS"}
        ADORA.BlockStore %28, %16 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "11", KernelName = "GEMMIS"}
        ADORA.BlockStore %27, %10 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "10", KernelName = "GEMMIS"}
        ADORA.BlockStore %26, %12 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "9", KernelName = "GEMMIS"}
        ADORA.BlockStore %25, %14 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "8", KernelName = "GEMMIS"}
        ADORA.BlockStore %10, %arg2 [%arg4, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %12, %arg2 [%arg4 + 1, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %14, %arg2 [%arg4 + 2, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %16, %arg2 [%arg4 + 3, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
      } {ADORAGemm}
    } {ADORAGemm}
    %alloc_0 = memref.alloc() : memref<32x64xbf16>
    memref.copy %arg2, %alloc_0 : memref<32x64xbf16> to memref<32x64xbf16>
    affine.for %arg3 = 0 to 128 step 4 {
      affine.for %arg4 = 0 to 32 step 16 {
        %c0 = arith.constant 0 : index
        %1 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
        %2 = ADORA.BlockLoad %arg0 [%arg4 + 1, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
        %3 = ADORA.BlockLoad %arg0 [%arg4 + 2, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
        %4 = ADORA.BlockLoad %arg0 [%arg4 + 3, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
        %5 = ADORA.BlockLoad %arg1 [%arg3, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
        %6 = ADORA.BlockLoad %arg1 [%arg3 + 1, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
        %7 = ADORA.BlockLoad %arg1 [%arg3 + 2, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
        %8 = ADORA.BlockLoad %arg1 [%arg3 + 3, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
        %9 = ADORA.BlockLoad %arg2 [%arg4, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
        %10 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
        %11 = ADORA.BlockLoad %arg2 [%arg4 + 1, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
        %12 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
        %13 = ADORA.BlockLoad %arg2 [%arg4 + 2, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
        %14 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
        %15 = ADORA.BlockLoad %arg2 [%arg4 + 3, %c0] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
        %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
        %17 = ADORA.BlockLoad %5 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "0", KernelName = "GEMMIS"}
        %18 = ADORA.BlockLoad %6 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "1", KernelName = "GEMMIS"}
        %19 = ADORA.BlockLoad %7 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "2", KernelName = "GEMMIS"}
        %20 = ADORA.BlockLoad %8 [0, 0] : memref<1x64xbf16> -> memref<1x64xbf16>  {Id = "3", KernelName = "GEMMIS"}
        %21 = ADORA.BlockLoad %9 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "4", KernelName = "GEMMIS"}
        %22 = ADORA.BlockLoad %11 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "5", KernelName = "GEMMIS"}
        %23 = ADORA.BlockLoad %13 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "6", KernelName = "GEMMIS"}
        %24 = ADORA.BlockLoad %15 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "7", KernelName = "GEMMIS"}
        %25 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "8", KernelName = "GEMMIS"}
        %26 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "9", KernelName = "GEMMIS"}
        %27 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "10", KernelName = "GEMMIS"}
        %28 = ADORA.LocalMemAlloc memref<4x64xbf16>  {Id = "11", KernelName = "GEMMIS"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 4 {
            %29 = affine.vector_load %1[%arg5, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %30:4 = ADORA.deinterleaver %29 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %31 = affine.vector_load %2[%arg5 + 1, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %32:4 = ADORA.deinterleaver %31 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %33 = affine.vector_load %3[%arg5 + 2, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %34:4 = ADORA.deinterleaver %33 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            %35 = affine.vector_load %4[%arg5 + 3, 0] {ADORAGemm, Pingpong} : memref<4x4xbf16>, vector<4xbf16>
            %36:4 = ADORA.deinterleaver %35 : vector<4xbf16> -> (bf16, bf16, bf16, bf16) {ADORAGemm}
            affine.for %arg6 = 0 to 64 {
              %37 = affine.load %17[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %38 = arith.mulf %30#0, %37 {ADORAGemm} : bf16
              %39 = affine.load %18[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %40 = arith.mulf %30#1, %39 {ADORAGemm} : bf16
              %41 = arith.addf %40, %38 {ADORAGemm} : bf16
              %42 = affine.load %19[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %43 = arith.mulf %30#2, %42 {ADORAGemm} : bf16
              %44 = arith.addf %43, %41 {ADORAGemm} : bf16
              %45 = affine.load %20[0, %arg6] {ADORAGemm, Pingpong} : memref<1x64xbf16>
              %46 = arith.mulf %30#3, %45 {ADORAGemm} : bf16
              %47 = arith.addf %46, %44 {ADORAGemm} : bf16
              %48 = arith.mulf %32#0, %37 {ADORAGemm} : bf16
              %49 = arith.mulf %32#1, %39 {ADORAGemm} : bf16
              %50 = arith.addf %49, %48 {ADORAGemm} : bf16
              %51 = arith.mulf %32#2, %42 {ADORAGemm} : bf16
              %52 = arith.addf %51, %50 {ADORAGemm} : bf16
              %53 = arith.mulf %32#3, %45 {ADORAGemm} : bf16
              %54 = arith.addf %53, %52 {ADORAGemm} : bf16
              %55 = arith.mulf %34#0, %37 {ADORAGemm} : bf16
              %56 = arith.mulf %34#1, %39 {ADORAGemm} : bf16
              %57 = arith.addf %56, %55 {ADORAGemm} : bf16
              %58 = arith.mulf %34#2, %42 {ADORAGemm} : bf16
              %59 = arith.addf %58, %57 {ADORAGemm} : bf16
              %60 = arith.mulf %34#3, %45 {ADORAGemm} : bf16
              %61 = arith.addf %60, %59 {ADORAGemm} : bf16
              %62 = arith.mulf %36#0, %37 {ADORAGemm} : bf16
              %63 = arith.mulf %36#1, %39 {ADORAGemm} : bf16
              %64 = arith.addf %63, %62 {ADORAGemm} : bf16
              %65 = arith.mulf %36#2, %42 {ADORAGemm} : bf16
              %66 = arith.addf %65, %64 {ADORAGemm} : bf16
              %67 = arith.mulf %36#3, %45 {ADORAGemm} : bf16
              %68 = arith.addf %67, %66 {ADORAGemm} : bf16
              %69 = affine.load %21[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %70 = arith.addf %47, %69 {ADORAGemm} : bf16
              affine.store %70, %28[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %71 = affine.load %22[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %72 = arith.addf %54, %71 {ADORAGemm} : bf16
              affine.store %72, %27[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %73 = affine.load %23[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %74 = arith.addf %61, %73 {ADORAGemm} : bf16
              affine.store %74, %25[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %75 = affine.load %24[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
              %76 = arith.addf %68, %75 {ADORAGemm} : bf16
              affine.store %76, %26[%arg5, %arg6] {ADORAGemm, Pingpong} : memref<4x64xbf16>
            } {ADORAGemm}
          } {ADORAGemm}
          ADORA.terminator {ADORAGemm}
        } {ADORAGemm, KernelName = "GEMMIS"}
        ADORA.BlockStore %28, %10 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "11", KernelName = "GEMMIS"}
        ADORA.BlockStore %27, %12 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "10", KernelName = "GEMMIS"}
        ADORA.BlockStore %26, %16 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "9", KernelName = "GEMMIS"}
        ADORA.BlockStore %25, %14 [0, 0] : memref<4x64xbf16> -> memref<4x64xbf16>  {Id = "8", KernelName = "GEMMIS"}
        ADORA.BlockStore %10, %arg2 [%arg4, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %12, %arg2 [%arg4 + 1, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %14, %arg2 [%arg4 + 2, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
        ADORA.BlockStore %16, %arg2 [%arg4 + 3, %c0] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
      } {ADORAGemm}
    } {ADORAGemm}
    return %alloc_0 : memref<32x64xbf16>
  }
}


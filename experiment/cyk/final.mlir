#map = affine_map<(d0, d1) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0 + d1 * 4)>
#map2 = affine_map<(d0, d1) -> (d1)>
#map3 = affine_map<(d0, d1) -> (d0 + d1)>
#map4 = affine_map<(d0) -> (d0 floordiv 1024)>
#map5 = affine_map<(d0) -> (d0 mod 1024)>
#map6 = affine_map<(d0) -> (d0 floordiv 32)>
#map7 = affine_map<(d0) -> (d0 mod 32)>
#map8 = affine_map<(d0) -> (d0 floordiv 9)>
#map9 = affine_map<(d0) -> (d0 mod 9)>
#map10 = affine_map<(d0) -> (d0 floordiv 3)>
#map11 = affine_map<(d0) -> (d0 mod 3)>
#map12 = affine_map<(d0, d1) -> (d1 + 4)>
#map13 = affine_map<(d0, d1) -> (d1 + 8)>
#map14 = affine_map<(d0, d1) -> (d0 + 1)>
#map15 = affine_map<(d0, d1) -> (d0 + 2)>
#map16 = affine_map<(d0, d1) -> (d0 + 3)>
#map17 = affine_map<(d0, d1) -> (d0 + 4)>
#map18 = affine_map<(d0, d1) -> (d0 + 5)>
#map19 = affine_map<(d0, d1) -> (d0 + 6)>
#map20 = affine_map<(d0, d1) -> (d0 + 7)>
#map21 = affine_map<(d0, d1) -> (d0 + 8)>
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
    %c0 = arith.constant 0 : index
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 34 {
        affine.for %arg5 = 0 to 34 {
          affine.store %cst, %alloc[%c0, %arg3, %arg4, %arg5] : memref<1x3x34x34xf32>
        }
      }
    }
    %c0_0 = arith.constant 0 : index
    affine.for %arg3 = 0 to 3 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 32 {
          %0 = affine.load %arg0[%c0_0, %arg3, %arg4, %arg5] : memref<1x3x32x32xf32>
          affine.store %0, %alloc[%c0_0, %arg3, %arg4 + 1, %arg5 + 1] : memref<1x3x34x34xf32>
        }
      }
    }
    %alloc_1 = memref.alloc() : memref<1x16x32x32xf32>
    %c0_2 = arith.constant 0 : index
    affine.for %arg3 = 0 to 16 {
      affine.for %arg4 = 0 to 32 {
        affine.for %arg5 = 0 to 32 {
          %0 = affine.load %arg2[%arg3] : memref<16xf32>
          affine.store %0, %alloc_1[%c0_2, %arg3, %arg4, %arg5] : memref<1x16x32x32xf32>
        }
      }
    }
    affine.for %arg3 = 0 to 27 step 9 {
      affine.for %arg4 = 0 to 1024 step 256 {
        %c0_3 = arith.constant 0 : index
        %0 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %0[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %1 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map12(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %1[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %2 = ADORA.LocalMemAlloc memref<64x1xf32>  {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          %c0_13 = arith.constant 0 : index
          %50 = affine.apply #map(%arg4, %arg3) {ADORAGemm}
          %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
          %52 = affine.apply #map13(%arg4, %arg3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %c0_13) {ADORAGemm}
          %54 = affine.apply #map4(%51) {ADORAGemm}
          %55 = affine.apply #map5(%51) {ADORAGemm}
          %56 = affine.apply #map6(%55) {ADORAGemm}
          %57 = affine.apply #map7(%55) {ADORAGemm}
          %58 = affine.apply #map8(%53) {ADORAGemm}
          %59 = affine.apply #map9(%53) {ADORAGemm}
          %60 = affine.apply #map10(%59) {ADORAGemm}
          %61 = affine.apply #map11(%59) {ADORAGemm}
          %62 = affine.apply #map3(%56, %60) {ADORAGemm}
          %63 = affine.apply #map3(%57, %61) {ADORAGemm}
          %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
          affine.store %64, %2[%arg5, %c0_13] {ADORAGemm} : memref<64x1xf32>
        } {ADORAGemm}
        %3 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map14(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %3[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %4 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map14(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map12(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %4[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %5 = ADORA.LocalMemAlloc memref<64x1xf32>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          %c0_13 = arith.constant 0 : index
          %50 = affine.apply #map14(%arg4, %arg3) {ADORAGemm}
          %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
          %52 = affine.apply #map13(%arg4, %arg3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %c0_13) {ADORAGemm}
          %54 = affine.apply #map4(%51) {ADORAGemm}
          %55 = affine.apply #map5(%51) {ADORAGemm}
          %56 = affine.apply #map6(%55) {ADORAGemm}
          %57 = affine.apply #map7(%55) {ADORAGemm}
          %58 = affine.apply #map8(%53) {ADORAGemm}
          %59 = affine.apply #map9(%53) {ADORAGemm}
          %60 = affine.apply #map10(%59) {ADORAGemm}
          %61 = affine.apply #map11(%59) {ADORAGemm}
          %62 = affine.apply #map3(%56, %60) {ADORAGemm}
          %63 = affine.apply #map3(%57, %61) {ADORAGemm}
          %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
          affine.store %64, %5[%arg5, %c0_13] {ADORAGemm} : memref<64x1xf32>
        } {ADORAGemm}
        %6 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map15(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %6[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %7 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map15(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map12(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %7[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %8 = ADORA.LocalMemAlloc memref<64x1xf32>  {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          %c0_13 = arith.constant 0 : index
          %50 = affine.apply #map15(%arg4, %arg3) {ADORAGemm}
          %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
          %52 = affine.apply #map13(%arg4, %arg3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %c0_13) {ADORAGemm}
          %54 = affine.apply #map4(%51) {ADORAGemm}
          %55 = affine.apply #map5(%51) {ADORAGemm}
          %56 = affine.apply #map6(%55) {ADORAGemm}
          %57 = affine.apply #map7(%55) {ADORAGemm}
          %58 = affine.apply #map8(%53) {ADORAGemm}
          %59 = affine.apply #map9(%53) {ADORAGemm}
          %60 = affine.apply #map10(%59) {ADORAGemm}
          %61 = affine.apply #map11(%59) {ADORAGemm}
          %62 = affine.apply #map3(%56, %60) {ADORAGemm}
          %63 = affine.apply #map3(%57, %61) {ADORAGemm}
          %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
          affine.store %64, %8[%arg5, %c0_13] {ADORAGemm} : memref<64x1xf32>
        } {ADORAGemm}
        %9 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map16(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %9[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %10 = ADORA.LocalMemAlloc memref<64x4xf32>  {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 4 {
            %50 = affine.apply #map16(%arg4, %arg3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map12(%arg4, %arg3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.apply #map8(%53) {ADORAGemm}
            %59 = affine.apply #map9(%53) {ADORAGemm}
            %60 = affine.apply #map10(%59) {ADORAGemm}
            %61 = affine.apply #map11(%59) {ADORAGemm}
            %62 = affine.apply #map3(%56, %60) {ADORAGemm}
            %63 = affine.apply #map3(%57, %61) {ADORAGemm}
            %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
            affine.store %64, %10[%arg5, %arg6] {ADORAGemm} : memref<64x4xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %11 = ADORA.LocalMemAlloc memref<64x1xf32>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          %c0_13 = arith.constant 0 : index
          %50 = affine.apply #map16(%arg4, %arg3) {ADORAGemm}
          %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
          %52 = affine.apply #map13(%arg4, %arg3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %c0_13) {ADORAGemm}
          %54 = affine.apply #map4(%51) {ADORAGemm}
          %55 = affine.apply #map5(%51) {ADORAGemm}
          %56 = affine.apply #map6(%55) {ADORAGemm}
          %57 = affine.apply #map7(%55) {ADORAGemm}
          %58 = affine.apply #map8(%53) {ADORAGemm}
          %59 = affine.apply #map9(%53) {ADORAGemm}
          %60 = affine.apply #map10(%59) {ADORAGemm}
          %61 = affine.apply #map11(%59) {ADORAGemm}
          %62 = affine.apply #map3(%56, %60) {ADORAGemm}
          %63 = affine.apply #map3(%57, %61) {ADORAGemm}
          %64 = affine.load %alloc[%54, %58, %62, %63] {ADORAGemm} : memref<1x3x34x34xf32>
          affine.store %64, %11[%arg5, %c0_13] {ADORAGemm} : memref<64x1xf32>
        } {ADORAGemm}
        %12 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
        %c0_4 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_4) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %12[%c0_4, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %13 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
        %c0_5 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map14(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_5) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %13[%c0_5, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %14 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
        %c0_6 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map15(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_6) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %14[%c0_6, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %15 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
        %c0_7 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map16(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_7) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %15[%c0_7, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %16 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "16", KernelName = "GEMMIS", Pingpong}
        %c0_8 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map17(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_8) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %16[%c0_8, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %17 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "17", KernelName = "GEMMIS", Pingpong}
        %c0_9 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map18(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_9) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %17[%c0_9, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %18 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "18", KernelName = "GEMMIS", Pingpong}
        %c0_10 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map19(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_10) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %18[%c0_10, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %19 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "19", KernelName = "GEMMIS", Pingpong}
        %c0_11 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map20(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_11) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %19[%c0_11, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %20 = ADORA.LocalMemAlloc memref<1x16xf32>  {ADORAGemm, Id = "20", KernelName = "GEMMIS", Pingpong}
        %c0_12 = arith.constant 0 : index
        affine.for %arg5 = 0 to 16 {
          %50 = affine.apply #map21(%arg3, %c0_3) {ADORAGemm}
          %51 = affine.apply #map3(%50, %c0_12) {ADORAGemm}
          %52 = affine.apply #map2(%arg3, %c0_3) {ADORAGemm}
          %53 = affine.apply #map3(%52, %arg5) {ADORAGemm}
          %54 = affine.apply #map8(%51) {ADORAGemm}
          %55 = affine.apply #map9(%51) {ADORAGemm}
          %56 = affine.apply #map10(%55) {ADORAGemm}
          %57 = affine.apply #map11(%55) {ADORAGemm}
          %58 = affine.load %arg1[%53, %54, %56, %57] {ADORAGemm} : memref<16x3x3x3xf32>
          affine.store %58, %20[%c0_12, %arg5] {ADORAGemm} : memref<1x16xf32>
        } {ADORAGemm}
        %21 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "21", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
            affine.store %58, %21[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %22 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "22", KernelName = "GEMMIS", Pingpong}
        %23 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "23", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map14(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
            affine.store %58, %23[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %24 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "24", KernelName = "GEMMIS", Pingpong}
        %25 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "25", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map15(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
            affine.store %58, %25[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %26 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "26", KernelName = "GEMMIS", Pingpong}
        %27 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "27", KernelName = "GEMMIS", Pingpong}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map16(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
            affine.store %58, %27[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        %28 = ADORA.LocalMemAlloc memref<64x16xf32>  {ADORAGemm, Id = "28", KernelName = "GEMMIS", Pingpong}
        %29 = ADORA.BlockLoad %2 [0, 8] : memref<64x1xf32> -> memref<64x1xf32>  {Id = "0", KernelName = "GEMMIS"}
        %30 = ADORA.BlockLoad %5 [1, 8] : memref<64x1xf32> -> memref<64x1xf32>  {Id = "1", KernelName = "GEMMIS"}
        %31 = ADORA.BlockLoad %8 [2, 8] : memref<64x1xf32> -> memref<64x1xf32>  {Id = "2", KernelName = "GEMMIS"}
        %32 = ADORA.BlockLoad %11 [3, 8] : memref<64x1xf32> -> memref<64x1xf32>  {Id = "3", KernelName = "GEMMIS"}
        %33 = ADORA.BlockLoad %12 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "4", KernelName = "GEMMIS"}
        %34 = ADORA.BlockLoad %13 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "5", KernelName = "GEMMIS"}
        %35 = ADORA.BlockLoad %14 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "6", KernelName = "GEMMIS"}
        %36 = ADORA.BlockLoad %15 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "7", KernelName = "GEMMIS"}
        %37 = ADORA.BlockLoad %16 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "8", KernelName = "GEMMIS"}
        %38 = ADORA.BlockLoad %17 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "9", KernelName = "GEMMIS"}
        %39 = ADORA.BlockLoad %18 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "10", KernelName = "GEMMIS"}
        %40 = ADORA.BlockLoad %19 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "11", KernelName = "GEMMIS"}
        %41 = ADORA.BlockLoad %20 [0, 0] : memref<1x16xf32> -> memref<1x16xf32>  {Id = "12", KernelName = "GEMMIS"}
        %42 = ADORA.BlockLoad %21 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "13", KernelName = "GEMMIS"}
        %43 = ADORA.BlockLoad %23 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "14", KernelName = "GEMMIS"}
        %44 = ADORA.BlockLoad %25 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "15", KernelName = "GEMMIS"}
        %45 = ADORA.BlockLoad %27 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "16", KernelName = "GEMMIS"}
        %46 = ADORA.LocalMemAlloc memref<64x16xf32>  {Id = "17", KernelName = "GEMMIS"}
        %47 = ADORA.LocalMemAlloc memref<64x16xf32>  {Id = "18", KernelName = "GEMMIS"}
        %48 = ADORA.LocalMemAlloc memref<64x16xf32>  {Id = "19", KernelName = "GEMMIS"}
        %49 = ADORA.LocalMemAlloc memref<64x16xf32>  {Id = "20", KernelName = "GEMMIS"}
        ADORA.kernel {
          affine.for %arg5 = 0 to 64 {
            affine.for %arg6 = 0 to 16 {
              %50 = affine.vector_load %0[%arg5, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %51:4 = ADORA.deinterleaver %50 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %52 = affine.vector_load %1[%arg5, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %53:4 = ADORA.deinterleaver %52 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %54 = affine.load %29[%arg5, 0] {ADORAGemm, Pingpong} : memref<64x1xf32>
              %55 = affine.vector_load %3[%arg5 + 1, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %56:4 = ADORA.deinterleaver %55 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %57 = affine.vector_load %4[%arg5 + 1, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %58:4 = ADORA.deinterleaver %57 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %59 = affine.load %30[%arg5, 0] {ADORAGemm, Pingpong} : memref<64x1xf32>
              %60 = affine.vector_load %6[%arg5 + 2, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %61:4 = ADORA.deinterleaver %60 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %62 = affine.vector_load %7[%arg5 + 2, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %63:4 = ADORA.deinterleaver %62 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %64 = affine.load %31[%arg5, 0] {ADORAGemm, Pingpong} : memref<64x1xf32>
              %65 = affine.vector_load %9[%arg5 + 3, 0] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %66:4 = ADORA.deinterleaver %65 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %67 = affine.vector_load %10[%arg5 + 3, 4] {ADORAGemm, Pingpong} : memref<64x4xf32>, vector<4xf32>
              %68:4 = ADORA.deinterleaver %67 : vector<4xf32> -> (f32, f32, f32, f32) {ADORAGemm}
              %69 = affine.load %32[%arg5, 0] {ADORAGemm, Pingpong} : memref<64x1xf32>
              affine.for %arg7 = 0 to 16 {
                %70 = affine.load %33[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %71 = arith.mulf %51#0, %70 {ADORAGemm} : f32
                %72 = affine.load %34[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %73 = arith.mulf %51#1, %72 {ADORAGemm} : f32
                %74 = arith.addf %73, %71 {ADORAGemm} : f32
                %75 = affine.load %35[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %76 = arith.mulf %51#2, %75 {ADORAGemm} : f32
                %77 = arith.addf %76, %74 {ADORAGemm} : f32
                %78 = affine.load %36[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %79 = arith.mulf %51#3, %78 {ADORAGemm} : f32
                %80 = arith.addf %79, %77 {ADORAGemm} : f32
                %81 = affine.load %37[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %82 = arith.mulf %53#0, %81 {ADORAGemm} : f32
                %83 = arith.addf %82, %80 {ADORAGemm} : f32
                %84 = affine.load %38[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %85 = arith.mulf %53#1, %84 {ADORAGemm} : f32
                %86 = arith.addf %85, %83 {ADORAGemm} : f32
                %87 = affine.load %39[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %88 = arith.mulf %53#2, %87 {ADORAGemm} : f32
                %89 = arith.addf %88, %86 {ADORAGemm} : f32
                %90 = affine.load %40[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %91 = arith.mulf %53#3, %90 {ADORAGemm} : f32
                %92 = arith.addf %91, %89 {ADORAGemm} : f32
                %93 = affine.load %41[0, %arg7] {ADORAGemm, Pingpong} : memref<1x16xf32>
                %94 = arith.mulf %54, %93 {ADORAGemm} : f32
                %95 = arith.addf %94, %92 {ADORAGemm} : f32
                %96 = arith.mulf %56#0, %70 {ADORAGemm} : f32
                %97 = arith.mulf %56#1, %72 {ADORAGemm} : f32
                %98 = arith.addf %97, %96 {ADORAGemm} : f32
                %99 = arith.mulf %56#2, %75 {ADORAGemm} : f32
                %100 = arith.addf %99, %98 {ADORAGemm} : f32
                %101 = arith.mulf %56#3, %78 {ADORAGemm} : f32
                %102 = arith.addf %101, %100 {ADORAGemm} : f32
                %103 = arith.mulf %58#0, %81 {ADORAGemm} : f32
                %104 = arith.addf %103, %102 {ADORAGemm} : f32
                %105 = arith.mulf %58#1, %84 {ADORAGemm} : f32
                %106 = arith.addf %105, %104 {ADORAGemm} : f32
                %107 = arith.mulf %58#2, %87 {ADORAGemm} : f32
                %108 = arith.addf %107, %106 {ADORAGemm} : f32
                %109 = arith.mulf %58#3, %90 {ADORAGemm} : f32
                %110 = arith.addf %109, %108 {ADORAGemm} : f32
                %111 = arith.mulf %59, %93 {ADORAGemm} : f32
                %112 = arith.addf %111, %110 {ADORAGemm} : f32
                %113 = arith.mulf %61#0, %70 {ADORAGemm} : f32
                %114 = arith.mulf %61#1, %72 {ADORAGemm} : f32
                %115 = arith.addf %114, %113 {ADORAGemm} : f32
                %116 = arith.mulf %61#2, %75 {ADORAGemm} : f32
                %117 = arith.addf %116, %115 {ADORAGemm} : f32
                %118 = arith.mulf %61#3, %78 {ADORAGemm} : f32
                %119 = arith.addf %118, %117 {ADORAGemm} : f32
                %120 = arith.mulf %63#0, %81 {ADORAGemm} : f32
                %121 = arith.addf %120, %119 {ADORAGemm} : f32
                %122 = arith.mulf %63#1, %84 {ADORAGemm} : f32
                %123 = arith.addf %122, %121 {ADORAGemm} : f32
                %124 = arith.mulf %63#2, %87 {ADORAGemm} : f32
                %125 = arith.addf %124, %123 {ADORAGemm} : f32
                %126 = arith.mulf %63#3, %90 {ADORAGemm} : f32
                %127 = arith.addf %126, %125 {ADORAGemm} : f32
                %128 = arith.mulf %64, %93 {ADORAGemm} : f32
                %129 = arith.addf %128, %127 {ADORAGemm} : f32
                %130 = arith.mulf %66#0, %70 {ADORAGemm} : f32
                %131 = arith.mulf %66#1, %72 {ADORAGemm} : f32
                %132 = arith.addf %131, %130 {ADORAGemm} : f32
                %133 = arith.mulf %66#2, %75 {ADORAGemm} : f32
                %134 = arith.addf %133, %132 {ADORAGemm} : f32
                %135 = arith.mulf %66#3, %78 {ADORAGemm} : f32
                %136 = arith.addf %135, %134 {ADORAGemm} : f32
                %137 = arith.mulf %68#0, %81 {ADORAGemm} : f32
                %138 = arith.addf %137, %136 {ADORAGemm} : f32
                %139 = arith.mulf %68#1, %84 {ADORAGemm} : f32
                %140 = arith.addf %139, %138 {ADORAGemm} : f32
                %141 = arith.mulf %68#2, %87 {ADORAGemm} : f32
                %142 = arith.addf %141, %140 {ADORAGemm} : f32
                %143 = arith.mulf %68#3, %90 {ADORAGemm} : f32
                %144 = arith.addf %143, %142 {ADORAGemm} : f32
                %145 = arith.mulf %69, %93 {ADORAGemm} : f32
                %146 = arith.addf %145, %144 {ADORAGemm} : f32
                %147 = affine.load %42[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %148 = arith.addf %95, %147 {ADORAGemm} : f32
                affine.store %148, %49[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %149 = affine.load %43[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %150 = arith.addf %112, %149 {ADORAGemm} : f32
                affine.store %150, %47[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %151 = affine.load %44[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %152 = arith.addf %129, %151 {ADORAGemm} : f32
                affine.store %152, %46[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %153 = affine.load %45[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
                %154 = arith.addf %146, %153 {ADORAGemm} : f32
                affine.store %154, %48[%arg5, %arg7] {ADORAGemm, Pingpong} : memref<64x16xf32>
              } {ADORAGemm}
            } {ADORAGemm}
          } {ADORAGemm}
          ADORA.terminator {ADORAGemm}
        } {ADORAGemm, KernelName = "GEMMIS"}
        ADORA.BlockStore %49, %22 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "20", KernelName = "GEMMIS"}
        ADORA.BlockStore %48, %28 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "19", KernelName = "GEMMIS"}
        ADORA.BlockStore %47, %24 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "18", KernelName = "GEMMIS"}
        ADORA.BlockStore %46, %26 [0, 0] : memref<64x16xf32> -> memref<64x16xf32>  {Id = "17", KernelName = "GEMMIS"}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %22[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
            affine.store %58, %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map14(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %24[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
            affine.store %58, %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map15(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %26[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
            affine.store %58, %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
          } {ADORAGemm}
        } {ADORAGemm}
        affine.for %arg5 = 0 to 64 {
          affine.for %arg6 = 0 to 16 {
            %50 = affine.apply #map16(%arg4, %c0_3) {ADORAGemm}
            %51 = affine.apply #map1(%50, %arg5) {ADORAGemm}
            %52 = affine.apply #map2(%arg4, %c0_3) {ADORAGemm}
            %53 = affine.apply #map3(%52, %arg6) {ADORAGemm}
            %54 = affine.apply #map4(%51) {ADORAGemm}
            %55 = affine.apply #map5(%51) {ADORAGemm}
            %56 = affine.apply #map6(%55) {ADORAGemm}
            %57 = affine.apply #map7(%55) {ADORAGemm}
            %58 = affine.load %28[%arg5, %arg6] {ADORAGemm} : memref<64x16xf32>
            affine.store %58, %alloc_1[%54, %53, %56, %57] {ADORAGemm} : memref<1x16x32x32xf32>
          } {ADORAGemm}
        } {ADORAGemm}
      } {ADORAGemm}
    } {ADORAGemm}
    return %alloc_1 : memref<1x16x32x32xf32>
  }
}


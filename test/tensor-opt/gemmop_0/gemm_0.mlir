// RUN: tensor-opt \
// RUN:   --adora-tensor-op-strategy-decision="adg-fn=%S/../../spec/cgra_bf16/vitra_cgra_adg.json bus-bandwidth=16" \
// RUN:   %s | %FileCheck %s

// CHECK: module {
// CHECK:   func.func @matmul_0(%arg0: memref<8x3072xbf16>, %arg1: memref<3072x768xbf16>, %arg2: memref<8x768xbf16>) -> memref<8x768xbf16> {
// CHECK:     %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {algorithm = "GEMM_Standard", stationary_kind = "InputStationary", tile_size = array<i64: 2, 768, 4, 12>} : (memref<8x3072xbf16>, memref<3072x768xbf16>, memref<8x768xbf16>) -> memref<8x768xbf16>
// CHECK:     return %0 : memref<8x768xbf16>
// CHECK:   }
// CHECK: }

func.func @matmul_0(%arg0: memref<8x3072xbf16>, %arg1: memref<3072x768xbf16>, %arg2: memref<8x768xbf16>) -> memref<8x768xbf16> {
  %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) : (memref<8x3072xbf16>, memref<3072x768xbf16>, memref<8x768xbf16>) -> memref<8x768xbf16>
  return %0 : memref<8x768xbf16>
}
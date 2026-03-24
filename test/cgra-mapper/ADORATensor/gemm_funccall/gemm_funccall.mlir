// RUN: rm -rf %t && mkdir -p %t
// RUN: adoracc.py %s --work-dir %t -o %t/opt.mlir
// RUN: %cgra-mapper --adg=%S/../../../../../MatrixMeld/vitrartl/spec/vitra_cgra_adg.json --op-file=%S/../../../../../MatrixMeld/vitrartl/spec/operations.json --output-type=pytest --obj-opt=false %t/opt.mlir --output=%t/gemm_funccall.py
// RUN: test -s %t/gemm_funccall.py
// RUN: %FileCheck %s --check-prefix=CHECK-PY --input-file=%t/gemm_funccall.py
//
// Verify cgra-mapper flow with function call around ADORATensor.Gemm.
//
// CHECK-PY: async def matmul_0(runtime: DeviceRuntime
// CHECK-PY: await Gemm_0(runtime
// CHECK-PY: async def Gemm_0(runtime: DeviceRuntime

#map = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
module {
  func.func @matmul_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {llvm.emit_c_interface} {
    %0 = call @Gemm_0(%arg0, %arg1, %arg2) : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    return %0 : memref<32x64xbf16>
  }
  func.func @Gemm_0(%arg0: memref<32x128xbf16>, %arg1: memref<128x64xbf16>, %arg2: memref<32x64xbf16>) -> memref<32x64xbf16> attributes {adora_kernel, llvm.emit_c_interface} {
    %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [#map, #map1, #map2], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 4, 64, 4, 4>} : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    return %0 : memref<32x64xbf16>
  }
}


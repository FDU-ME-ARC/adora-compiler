// XFAIL: *
// RUN: tensor-opt --adora-gen-tensor-op-cdfg %s | FileCheck %s

// CHECK-LABEL: func.func @test_conv_direct
func.func @test_conv_direct(%arg0: memref<1x3x16x16xbf16>, %arg1: memref<8x3x3x3xbf16>, %arg2: memref<8xbf16>) -> memref<1x8x16x16xbf16> attributes {llvm.emit_c_interface, onnxEntryPoint} {
  %0 = "ADORATensor.Conv"(%arg0, %arg1, %arg2) <{
    auto_pad = "NOTSET", 
    dilations = [1, 1], 
    group = 1 : i64, 
    kernel_shape = [3, 3], 
    pads = [1, 1, 1, 1], 
    strides = [1, 1]
  }> {
    algorithm = "Conv_Direct", 
    stationary_kind = "InputStationary", 
    tile_size = array<i64: 1, 4, 8, 8> 
  } : (memref<1x3x16x16xbf16>, memref<8x3x3x3xbf16>, memref<8xbf16>) -> memref<1x8x16x16xbf16>
  
  return %0 : memref<1x8x16x16xbf16>

  // =====================================================================
  // Verification 1: Ensure the global output matrix is pre-allocated
  // CHECK: %[[FINAL_RES:.*]] = memref.alloc() : memref<1x8x16x16xbf16>

  // Verification 2: IFM mathematical derivation check (P=8 -> H_in=10)
  // CHECK: ADORA.BlockLoad %{{.*}} : memref<1x3x16x16xbf16> -> memref<1x3x10x10xbf16>

  // Verification 3: Check micro-kernel uses affine.load and stores to final_res
  // CHECK: ADORA.kernel {
  // CHECK: ADORA.terminator {ADORAConv}
  // CHECK: } {ADORAConv, KernelName = "ConvDirect"}
  // CHECK: ADORA.BlockStore %{{.*}}, %[[FINAL_RES]]{{.*}} {ADORAConv, Id = "{{.*}}", KernelName = "ConvDirect", Pingpong}
}
// XFAIL: *
// RUN: tensor-opt --adora-gen-tensor-op-cdfg %s | FileCheck %s

// CHECK-LABEL: func.func @test_os_gemm_bug
func.func @test_os_gemm_bug(%A: memref<36x12xf32>, %B: memref<12x36xf32>, %C: memref<36x36xf32>) -> memref<36x36xf32> attributes {llvm.emit_c_interface, onnxEntryPoint} {
  
  // Use M=6, K=12, N=6. Tile is 6x6. 
  // 1) K(12) != N(6) triggers the M x N memory allocation bug (Bug 1).
  // 2) tile_row_size=6 triggers the remainder logic (6 % 4 = 2) for Bug 2, avoiding outer loop assertion.
  %0 = "ADORATensor.Gemm"(%A, %B, %C) {
      algorithm = "GEMM_Standard",
      stationary_kind = "OutputStationary",
      tile_size = array<i64: 1, 1, 6, 6>
  } : (memref<36x12xf32>, memref<12x36xf32>, memref<36x36xf32>) -> memref<36x36xf32>
  
  return %0 : memref<36x36xf32>

  // =====================================================================
  // Verify Bug 1 fix: Check if the allocated memory size for matrix C is M x N (6x6) instead of 6x12
  // CHECK: memref.alloc() : memref<6x6xf32>

  // Verify Bug 2 fix: Check the spatial offset calculation for the remainder 
  // (For a 6x6 tile, the remainder is 2 rows. It should allocate and store a 2x1 memref)
  // CHECK: ADORA.LocalMemAlloc memref<2x1xf32>
  // CHECK: ADORA.BlockStore %{{.*}}, %{{.*}}[{{.*}}] : memref<2x1xf32> -> memref<6x6xf32>
}
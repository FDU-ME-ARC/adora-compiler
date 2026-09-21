// RUN: rm -rf %t && mkdir -p %t
// RUN: %cgra-mapper --help | %FileCheck %s --check-prefix=HELP
// RUN: python3 %S/test_cstore_alignment.py %t %cgra-opt %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %s %S/normal_store.mlir.in %S/parallel_kernels.mlir.in

// HELP: --seed=<uint>
module {
  func.func @cstore_route(%raw_index: i32, %index_delta: i32,
                          %value: i32, %bias: i32,
                          %condition: i1, %invert: i1,
                          %gate: i1, %mask: i1,
                          %output: memref<16xi32>) {
    ADORA.kernel {
      %stored_value = arith.addi %value, %bias : i32
      %adjusted_index = arith.addi %raw_index, %index_delta : i32
      %index = arith.index_cast %adjusted_index : i32 to index
      %inverted = arith.addi %condition, %invert : i1
      %gated = arith.ori %inverted, %gate : i1
      %predicate = arith.andi %gated, %mask : i1
      ADORA.cond_store %stored_value, %output[%index] if %predicate : memref<16xi32>
      ADORA.terminator
    } {KernelName = "cstore_route"}
    return
  }
}

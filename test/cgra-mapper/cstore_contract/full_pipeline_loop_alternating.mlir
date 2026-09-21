// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_full_pipeline_loop_alternating.py %t %adoracc %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/../../spec/cgra_cstore_vitra/artifacts/loop_index_contract.json %S %s

// Keep the alternating predicate as real per-iteration data: a normal LOAD
// reads [1, 0, 1, 0], SLT converts each element to predicate bit 0, and the
// same physical loop-index ACC drives the byte-scaled CSTORE address.
module {
  func.func @full_pipeline_loop_alternating(
      %value: i16, %predicates: memref<4xi16>, %output: memref<4xi16>) {
    ADORA.kernel {
      %zero = arith.constant 0 : i16
      affine.for %i = 0 to 4 {
        %predicate_value = affine.load %predicates[%i] : memref<4xi16>
        %enable = arith.cmpi slt, %zero, %predicate_value : i16
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "full_pipeline_loop_alternating"}
    return
  }
}

// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_full_pipeline_loop_index_acc.py %t %adoracc %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/../../spec/cgra_cstore_vitra/artifacts/loop_index_contract.json %S %s

// Exercise the production compiler pipeline rather than feeding the mapper an
// already-normalized loop. affine-loop-normalize represents this logical IV as
// a 0..5 step-1 loop plus affine.apply (d0 * 2 + 3); CDFG generation must fold
// that canonical linear apply back into the physical loop-index ACC contract.
module {
  func.func @full_pipeline_loop_index_b(
      %value: i16, %enable: i1, %output: memref<16xi16>) {
    ADORA.kernel {
      affine.for %i = 3 to 13 step 2 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<16xi16>
      }
      ADORA.terminator
    } {KernelName = "full_pipeline_loop_index_b"}
    return
  }
}

// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_full_pipeline_real_if_store.py %t %adoracc %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/../../spec/cgra_cstore_vitra/artifacts/loop_index_contract.json %S %s

// This is the original real loop-bearing if_store workload.  In addition to
// proving that FOR no longer reaches the physical mapper, lock the physical
// materialization of the constant store value: a CSTORE IOB has no local
// immediate operand, so 7 must be produced by a routed GPE operand 0.
module attributes {} {
  func.func @if_store(%a: i1, %output: memref<16xi32>) {
    %c7_i32 = arith.constant 7 : i32
    affine.for %i = 0 to 16 {
      scf.if %a {
        affine.store %c7_i32, %output[%i] : memref<16xi32>
      }
    }
    return
  }
}

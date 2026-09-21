// RUN: rm -rf %t && mkdir -p %t
// RUN: python3 %S/test_loop_index_acc_mapping.py %t %cgra-opt %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S/../../spec/cgra_cstore_vitra/artifacts/loop_index_contract.json %S

// The checker reads the pinned ADG's configuration IDs and ranges and decodes
// the mapper's emitted config.bit.  Case B deliberately has a non-zero lower
// bound, non-unit step, and 2-byte elements.
module {
  func.func @loop_index_a(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_a"}
    return
  }
}

// -----

module {
  func.func @loop_index_b(%value: i16, %enable: i1, %output: memref<16xi16>) {
    ADORA.kernel {
      affine.for %i = 3 to 13 step 2 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<16xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_b"}
    return
  }
}

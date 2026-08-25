// RUN: python3 %S/test_cstore_loop_index_acc.py %cgra-opt %cgra-mapper %S/../../spec/cgra_cstore_vitra/artifacts/adg.json %S/../../spec/cgra_cstore_vitra/artifacts/operations.json %S

// Supported loop-index CSTORE lowering must replace the physical affine.for
// node with an ACC whose routed operand 0 is the positive induction step.
// The checker also exercises an unsupported empty loop, existing generic ACC,
// and existing CSTORE/STORE regressions so this focused test documents the
// boundary without widening it.

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

// -----

module {
  func.func @loop_index_empty(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 4 to 4 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_empty"}
    return
  }
}

// -----

// Loop results/iter_args are control semantics that this v1 physical lowering
// cannot represent. The result is intentionally unused.
module {
  func.func @loop_index_iter_arg(%value: i16, %enable: i1, %output: memref<4xi16>) {
    %zero = arith.constant 0 : i16
    ADORA.kernel {
      %unused = affine.for %i = 0 to 4 iter_args(%carry = %zero) -> (i16) {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
        affine.yield %carry : i16
      }
      ADORA.terminator
    } {KernelName = "loop_index_iter_arg"}
    return
  }
}

// -----

// An affine.apply is an indirect IV-to-address use and is outside the exact
// direct-use subset accepted by v1.
module {
  func.func @loop_index_indirect(%value: i16, %enable: i1, %scale: index,
                                 %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %address = affine.apply affine_map<(d0) -> (d0)>(%i)
        %dynamic_address = arith.muli %address, %scale : index
        ADORA.cond_store %value, %output[%dynamic_address] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_indirect"}
    return
  }
}

// -----

// The IV has an additional CSTORE-data path as well as the direct address
// path. A loop-index ACC may only be used at the exact direct address port.
module {
  func.func @loop_index_wrong_port(%enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %data = arith.index_cast %i : index to i16
        ADORA.cond_store %data, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_wrong_port"}
    return
  }
}

// -----

// This bound/step pair overflows the old rounded-up division numerator. It
// must still fail deterministically on the trip-count contract.
module {
  func.func @loop_index_extreme(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 9223372036854775807 step 2 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_extreme"}
    return
  }
}

// -----

// affine.for rejects non-positive static steps before CDFG generation. Keep a
// parser-level negative so this value can never reach signed trip arithmetic.
module {
  func.func @loop_index_negative_step(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 4 to 0 step -1 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_negative_step"}
    return
  }
}

// -----

// affine-loop-normalize emits this canonical one-dimensional linear apply.
// Its scale/offset must be folded into the physical ACC rather than silently
// losing CSTORE address connectivity.
module {
  func.func @loop_index_static_apply(%value: i16, %enable: i1, %output: memref<16xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %address = affine.apply affine_map<(d0) -> (d0 * 2 + 1)>(%i)
        ADORA.cond_store %value, %output[%address] if %enable : memref<16xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_static_apply"}
    return
  }
}

// -----

// A CSTORE-related nested loop would lose the outer execution count if only
// the inner IV were changed to ACC. v1 must reject before mutation.
module {
  func.func @loop_index_nested(%value: i16, %enable: i1, %output: memref<16xi16>) {
    ADORA.kernel {
      affine.for %outer = 0 to 2 {
        affine.for %inner = 0 to 4 {
          ADORA.cond_store %value, %output[%inner] if %enable : memref<16xi16>
        }
      }
      ADORA.terminator
    } {KernelName = "loop_index_nested"}
    return
  }
}

// -----

// A direct live affine.store address is an additional physical IV consumer.
module {
  func.func @loop_index_direct_extra(%value: i16, %enable: i1,
                                     %output: memref<4xi16>, %other: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
        affine.store %value, %other[%i] : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_direct_extra"}
    return
  }
}

// -----

// The same additional live use one arithmetic transform away is unsupported.
module {
  func.func @loop_index_transformed_extra(%value: i16, %enable: i1,
                                          %output: memref<4xi16>, %other: memref<8xi16>) {
    %one = arith.constant 1 : index
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %other_address = arith.addi %i, %one : index
        ADORA.cond_store %value, %output[%i] if %enable : memref<4xi16>
        memref.store %value, %other[%other_address] : memref<8xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_transformed_extra"}
    return
  }
}

// -----

// Division/modulo affine expressions are not v1 materialization inputs.
module {
  func.func @loop_index_division(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %address = affine.apply affine_map<(d0) -> (d0 floordiv 2)>(%i)
        ADORA.cond_store %value, %output[%address] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_division"}
    return
  }
}

// -----

module {
  func.func @loop_index_modulo(%value: i16, %enable: i1, %output: memref<4xi16>) {
    ADORA.kernel {
      affine.for %i = 0 to 4 {
        %address = affine.apply affine_map<(d0) -> (d0 mod 2)>(%i)
        ADORA.cond_store %value, %output[%address] if %enable : memref<4xi16>
      }
      ADORA.terminator
    } {KernelName = "loop_index_modulo"}
    return
  }
}

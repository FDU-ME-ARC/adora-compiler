// PR6.3 Pass 5 — verify --adora-lower-async-tokens strips !ADORA.token
// iter_args / affine.yield operands emitted by PR6.2 on affine.for ops.
//
// RUN: %cgra-opt %s --adora-schedule-tasks --adora-assign-streams \
// RUN:           --adora-lower-async-tokens 2>/dev/null | FileCheck %s

// The loop must be rewritten to a plain affine.for with no iter_args and
// no results, and its body must contain no !ADORA.token-typed SSA values
// other than those produced by adora.event.* ops (events themselves are
// typed !ADORA.token and are the lowered carrier of async deps).

// CHECK-LABEL: func.func @lc
// CHECK-NOT: iter_args
// CHECK: affine.for %{{.*}} = 0 to
// CHECK-NOT: affine.yield %{{.*}} : !ADORA.token

#map = affine_map<(d0) -> (d0, 0)>

module {
  func.func @lc(%A: memref<64x16xf32>, %B: memref<16x16xf32>,
                %C: memref<64x16xf32>) {
    affine.for %tk = 0 to 4 {
      %a = "ADORA.BlockLoad"(%A, %tk)
           <{operandSegmentSizes = array<i32: 1, 1, 0>}>
           {Id = "0", KernelName = "k0", map = #map}
           : (memref<64x16xf32>, index) -> memref<16x16xf32>
      %b = "ADORA.BlockLoad"(%B) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
           {Id = "1", KernelName = "k0",
            map = affine_map<() -> (0, 0)>}
           : (memref<16x16xf32>) -> memref<16x16xf32>
      %c = "ADORA.BlockLoad"(%C, %tk)
           <{operandSegmentSizes = array<i32: 1, 1, 0>}>
           {Id = "2", KernelName = "k0", map = #map}
           : (memref<64x16xf32>, index) -> memref<16x16xf32>

      "ADORA.kernel"() ({
        "ADORA.terminator"() : () -> ()
      }) {KernelName = "k0"} : () -> ()

      "ADORA.BlockStore"(%c, %C, %tk)
           <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}>
           {Id = "3", KernelName = "k0", map = #map}
           : (memref<16x16xf32>, memref<64x16xf32>, index) -> ()
    }
    return
  }
}

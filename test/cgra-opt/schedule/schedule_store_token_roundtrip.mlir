// Regression: a BlockStore that CONSUMES a token but does NOT produce one
// (a terminal write-back, e.g. storing to a function argument that no later op
// reads) must still print its `async [...]` dependency list, so the dependency
// survives a textual print -> parse round-trip. Previously the printer gated
// the whole `async [...]` clause on `getAsyncToken()`, so a non-producing store
// hid its real dependency; round-tripping the IR then dropped it, which would
// let the store be issued before the kernel it depends on.
//
// Chain: BlockLoad -> kernel -> BlockStore(%arg0). The store reads the kernel's
// output and writes back to %arg0; nothing consumes the store's token, so the
// store consumes a token without producing one.
//
// Uses generic-form input to bypass adora-adjust-kernel-mem-footprint.
//
// 1) Schedule and check the terminal store shows its dependency but no token result.
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null \
// RUN:   | %FileCheck %s --check-prefix=SCHED
//
// 2) Round-trip the scheduled IR through parse+print; the dependency must be
//    preserved (idempotent / convergent).
// RUN: %cgra-opt %s --adora-schedule-tasks 2>/dev/null > %t.0
// RUN: %cgra-opt %t.0 2>/dev/null > %t.1
// RUN: %cgra-opt %t.1 2>/dev/null > %t.2
// RUN: diff %t.1 %t.2
// RUN: %FileCheck %s --check-prefix=ROUNDTRIP < %t.1

// The terminal store consumes the kernel token (%N) via `async [...]` but does
// NOT carry a `-> !ADORA.token` result suffix (it produces no token).
// SCHED:     ADORA.BlockStore async [
// SCHED-NOT: ADORA.BlockStore async {{.*}}-> !ADORA.token

// After round-tripping, the dependency is still present (not silently dropped).
// ROUNDTRIP:     ADORA.BlockStore async [
// ROUNDTRIP-NOT: ADORA.BlockStore async {{.*}}-> !ADORA.token

module {
  func.func @chain(%arg0: memref<?x25xf32>, %arg1: memref<?x25xf32>, %arg2: memref<?x25xf32>) {
    %0 = "ADORA.BlockLoad"(%arg0) <{operandSegmentSizes = array<i32: 1, 0, 0>}>
         {Id = "0", KernelName = "chain_0", map = affine_map<() -> (0, 0)>}
         : (memref<?x25xf32>) -> memref<20x25xf32>

    "ADORA.kernel"() ({
      "ADORA.terminator"() : () -> ()
    }) {KernelName = "chain_0"} : () -> ()

    "ADORA.BlockStore"(%0, %arg0) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>}>
         {Id = "1", KernelName = "chain_0", map = affine_map<() -> (0, 0)>}
         : (memref<20x25xf32>, memref<?x25xf32>) -> ()

    return
  }
}

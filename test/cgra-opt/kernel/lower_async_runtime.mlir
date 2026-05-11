// PR3 commit C — verify --adora-to-llvm-async-runtime lowers the four ADORA
// async event ops to llvm.call into the runtime ABI:
//
//   ADORA.event.create  -> llvm.call @adoraEventCreate() -> !llvm.ptr
//   ADORA.event.destroy -> llvm.call @adoraEventDestroy(!llvm.ptr)
//   ADORA.signal        -> llvm.call @adoraEventRecord(!llvm.ptr, i64)
//   ADORA.wait          -> llvm.call @adoraEventWait(!llvm.ptr, i64)
//
// Runs the full three-pass pipeline:
//   1. adora-schedule-tasks (emit-token=true)
//   2. adora-lower-async-tokens
//   3. adora-to-llvm-async-runtime
//
// RUN: cgra-opt %s \
// RUN:   --adora-schedule-tasks="emit-token=true" \
// RUN:   --adora-lower-async-tokens \
// RUN:   --adora-to-llvm-async-runtime \
// RUN:   2>/dev/null | FileCheck %s

// Extern runtime declarations must appear at module scope.
// CHECK-DAG: llvm.func @adoraEventCreate() -> !llvm.ptr
// CHECK-DAG: llvm.func @adoraEventDestroy(!llvm.ptr)
// CHECK-DAG: llvm.func @adoraEventRecord(!llvm.ptr, i64)
// CHECK-DAG: llvm.func @adoraEventWait(!llvm.ptr, i64)

// CHECK-LABEL: func.func @chain
// create → token ptr
// CHECK:         %[[EV:.+]] = llvm.call @adoraEventCreate() : () -> !llvm.ptr
// sync BlockLoad (no !ADORA.token result)
// CHECK:         %{{.+}} = ADORA.BlockLoad
// CHECK-NOT:     -> !ADORA.token
// signal
// CHECK:         llvm.call @adoraEventRecord(%[[EV]], %{{.+}}) : (!llvm.ptr, i64)
// wait before consumer
// CHECK:         llvm.call @adoraEventWait(%[[EV]], %{{.+}}) : (!llvm.ptr, i64)
// destroy after last use
// CHECK:         llvm.call @adoraEventDestroy(%[[EV]]) : (!llvm.ptr)
// No raw !ADORA.token values remain in the output.
// CHECK-NOT:     !ADORA.token

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

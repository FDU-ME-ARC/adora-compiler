// 05_loop_carried — minimal reduction example.
//
// One scf.for / affine.for whose body:
//   1) BlockLoad  %arg0 [0]     — no IV in map       ⇒ LC across iterations
//   2) BlockStore %arg0 [0]     — same tile as above ⇒ LC
//
// Expected analyzer output (in adora.lc_dep_summary):
//   LC-RAW : store(k) → load(k+1)    (same tile)
//   LC-WAR : load(k)  → store(k+1)   (same tile)
//   LC-WAW : store(k) → store(k+1)
//   LC-RAR : load(k)  → load(k+1)    (conservative — candidate for PR6.6
//                                     load-after-load elimination)
//
// RUN: %cgra-opt %s --adora-schedule-tasks="emit-token=true" 2>/dev/null \
// RUN:   | FileCheck %s

// CHECK: "adora.lc_dep_summary"
// CHECK-SAME: LC-RAW
// CHECK-SAME: LC-WAW

module {
  func.func @loop_carried_min(%arg0 : memref<16xf32>) {
    affine.for %tk = 0 to 4 {
      ADORA.kernel ins() outs(%arg0 : memref<16xf32>) {
        %c = ADORA.BlockLoad %arg0 [] : memref<16xf32> -> memref<16xf32>
            {Id = "c_load", KernelName = "loop_carried_min_kernel"}
        ADORA.BlockStore %c, %arg0 [] : memref<16xf32> -> memref<16xf32>
            {Id = "c_store", KernelName = "loop_carried_min_kernel"}
      } {KernelName = "loop_carried_min_kernel"}
    }
    return
  }
}

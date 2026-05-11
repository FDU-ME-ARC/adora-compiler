// PR6.2 loop-carried token threading on 05_loop_carried's affine.for.
//
// CHECK-LABEL: func.func @loop_carried_min
// CHECK: ADORA.event.create -> !ADORA.token
// CHECK: affine.for {{.*}} iter_args
// CHECK-SAME: !ADORA.token
// CHECK: ADORA.BlockLoad async
// CHECK: ADORA.kernel async
// CHECK: ADORA.BlockStore async {{.*}}-> !ADORA.token
// CHECK: affine.yield {{.*}} : !ADORA.token

// PR6.2 loop-carried token threading on the innermost affine.for of
// 04_gemm_tiled (the tk-reduction loop carrying the C-tile accumulator).
//
// CHECK-LABEL: func.func @gemm_tiled
// CHECK: ADORA.event.create -> !ADORA.token
// CHECK: affine.for {{.*}} iter_args
// CHECK-SAME: !ADORA.token
// CHECK: ADORA.BlockLoad async
// CHECK: ADORA.kernel async
// CHECK: %{{.*}} = ADORA.BlockStore async [%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}] %{{.*}}, %arg2
// CHECK: affine.yield {{.*}} : !ADORA.token, !ADORA.token, !ADORA.token

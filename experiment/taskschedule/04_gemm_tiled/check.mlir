// FileCheck patterns for 04_gemm_tiled
// C-tile load acquires WAR token
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg2 {{.*}} -> !ADORA.token
// A-tile load also produces token
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg0 {{.*}} -> !ADORA.token
// B-tile load also produces token
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg1 {{.*}} -> !ADORA.token
// kernel consumes all load tokens
// CHECK: ADORA.kernel async [%{{.*}}
// BlockStore consumes kernel token
// CHECK: ADORA.BlockStore async [%{{.*}}] {{.*}}, %arg2
// PR6 TODO: loop-carried token not yet wired
// (no scf.for iter_args here until PR6)

// FileCheck patterns for 03_3mm
//
// kernel_3mm_0: two loads → kernel → store
// CHECK: ADORA.BlockLoad %arg1 {{.*}} -> !ADORA.token
// CHECK: ADORA.BlockLoad %arg2 {{.*}} -> !ADORA.token
// CHECK: ADORA.kernel async [%{{.*}}, %{{.*}}]
// CHECK: ADORA.BlockStore async [%{{.*}}] {{.*}}, %arg0
//
// kernel_3mm_1: two loads → kernel → store
// CHECK: ADORA.BlockLoad %arg4 {{.*}} -> !ADORA.token
// CHECK: ADORA.BlockLoad %arg5 {{.*}} -> !ADORA.token
// CHECK: ADORA.kernel async [%{{.*}}, %{{.*}}]
// CHECK: ADORA.BlockStore async [%{{.*}}] {{.*}}, %arg3
//
// kernel_3mm_2: buffer reuse eliminates Load(%arg0) and Load(%arg3)
// kernel uses on-chip buffers directly, no load token deps needed
// CHECK: ADORA.kernel async
// CHECK: ADORA.BlockStore async [%{{.*}}] {{.*}}, %arg6
//
// Verify redundant loads are gone
// CHECK-NOT: ADORA.BlockLoad %arg0
// CHECK-NOT: ADORA.BlockLoad %arg3

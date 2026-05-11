// FileCheck patterns for 01_linear_chain
// BlockLoad produces token
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg0 {{.*}} -> !ADORA.token
// kernel consumes it, produces kernel token
// CHECK: %{{.*}} = ADORA.kernel async [%{{.*}}]
// BlockStore consumes kernel token
// CHECK: ADORA.BlockStore async [%{{.*}}]
// No async BlockLoad (no RAW pred on A)
// CHECK-NOT: ADORA.BlockLoad async

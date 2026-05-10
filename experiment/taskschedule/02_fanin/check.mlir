// FileCheck patterns for 02_fanin
// Both loads produce tokens (independent → parallel DMA possible)
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg0 {{.*}} -> !ADORA.token
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad %arg1 {{.*}} -> !ADORA.token
// kernel consumes both tokens (fan-in)
// CHECK: %{{.*}} = ADORA.kernel async [%{{.*}}, %{{.*}}]
// BlockStore consumes kernel token
// CHECK: ADORA.BlockStore async [%{{.*}}]
// No async BlockLoad (no RAW pred)
// CHECK-NOT: ADORA.BlockLoad async

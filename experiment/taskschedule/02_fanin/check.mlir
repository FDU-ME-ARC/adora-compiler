// FileCheck patterns for 02_fanin
// Both loads are independent (no RAW pred) → empty async list → parallel DMA possible
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad async [] %arg0 {{.*}}{Id = "0"
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad async [] %arg1 {{.*}}{Id = "1"
// kernel consumes both load tokens (fan-in)
// CHECK: %{{.*}} = ADORA.kernel async [%{{.*}}, %{{.*}}]
// store writes back the kernel result
// CHECK: ADORA.BlockStore %{{.*}}, %arg2

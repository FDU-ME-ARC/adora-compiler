// FileCheck patterns for 03_3mm
//
// kernel_3mm_0: two loads → kernel → store
// CHECK: ADORA.BlockLoad async [] %arg1 {{.*}}{Id = "0"
// CHECK: ADORA.BlockLoad async [] %arg2 {{.*}}{Id = "1"
// CHECK: ADORA.kernel async [%{{.*}}, %{{.*}}]
// CHECK: ADORA.BlockStore async [%{{.*}}] %{{.*}}, %arg0
//
// kernel_3mm_1: two loads → kernel → store
// CHECK: ADORA.BlockLoad async [] %arg4 {{.*}}{Id = "0"
// CHECK: ADORA.BlockLoad async [] %arg5 {{.*}}{Id = "1"
// CHECK: ADORA.kernel async [%{{.*}}, %{{.*}}]
// CHECK: ADORA.BlockStore async [%{{.*}}] %{{.*}}, %arg3
//
// kernel_3mm_2: buffer reuse eliminates Load(%arg0) and Load(%arg3)
// kernel waits on the two upstream store tokens, no fresh loads needed
// CHECK: ADORA.kernel async [%{{.*}}, %{{.*}}]
// store of the final result carries no async list
// CHECK: ADORA.BlockStore %{{.*}}, %arg6
//
// Verify redundant loads are gone
// CHECK-NOT: ADORA.BlockLoad async [] %arg0
// CHECK-NOT: ADORA.BlockLoad async [] %arg3

// FileCheck patterns for 01_linear_chain
// BlockLoad of A has no predecessor → empty async list, returns (result, token)
// CHECK: %{{.*}}, %{{.*}} = ADORA.BlockLoad async [] %arg0 {{.*}}{Id = "0"
// kernel consumes the load token, produces a kernel token
// CHECK: %{{.*}} = ADORA.kernel async [%{{.*}}]
// store writes the kernel result back (no async list on this store)
// CHECK: ADORA.BlockStore %{{.*}}, %arg1

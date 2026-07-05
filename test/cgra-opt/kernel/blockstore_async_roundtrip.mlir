// RUN: %cgra-opt %s | %cgra-opt | %FileCheck %s

// Regression for the BlockStore async print/parse round-trip. A token-producing
// store must print the explicit "-> !ADORA.token" suffix so the printed result
// name (%tok) can be reparsed; without it the parser rejects the op with
// "cannot name an operation with no results". A store that only consumes
// dependencies (or none) must NOT print a result name.

// CHECK-LABEL: func.func @blockstore_roundtrip
func.func @blockstore_roundtrip(%s: memref<1x8xi32>, %d: memref<1x8xi32>, %dep: !ADORA.token) {
  // Token-producing store: consumes %dep, produces a token.
  // CHECK: %{{.+}} = ADORA.BlockStore async [%{{.+}}] %{{.+}}, %{{.+}} [0, 0] : memref<1x8xi32> -> memref<1x8xi32> {{.*}}-> !ADORA.token
  %tok = ADORA.BlockStore async [%dep] %s, %d [0, 0] : memref<1x8xi32> -> memref<1x8xi32> -> !ADORA.token {Id = "3"}

  // Dependency-consuming terminal store: no token result, no result name.
  // CHECK: ADORA.BlockStore async [%{{.+}}] %{{.+}}, %{{.+}} [0, 0] : memref<1x8xi32> -> memref<1x8xi32>
  // CHECK-NOT: {{%.+}} = ADORA.BlockStore async [%{{.+}}] %{{.+}}, %{{.+}} [0, 0] : memref<1x8xi32> -> memref<1x8xi32> {{.*}}-> !ADORA.token
  ADORA.BlockStore async [%tok] %s, %d [0, 0] : memref<1x8xi32> -> memref<1x8xi32> {Id = "2a"}

  // Plain terminal store: no async, no token result.
  // CHECK: ADORA.BlockStore %{{.+}}, %{{.+}} [0, 0] : memref<1x8xi32> -> memref<1x8xi32>
  ADORA.BlockStore %s, %d [0, 0] : memref<1x8xi32> -> memref<1x8xi32> {Id = "2b"}
  return
}

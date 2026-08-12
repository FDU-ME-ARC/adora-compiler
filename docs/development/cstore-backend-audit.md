# CSTORE backend contract audit

**Scope.** This is a source and repository-history audit of the conditional
store (`CSTORE`) backend contract. It records what the mapper can represent
today and the separate hardware-spec gap that prevents an end-to-end CSTORE
claim. It does not change mapper code or hardware specifications.

## Mapper contract

The mapper recognises `CSTORE` as an I/O operation in both JSON-DFG and
LLVM-CDFG parsing. In each case it creates a `DFGIONode` and registers the
node as I/O ([`mapper/src/ir/dfg_ir.cpp`](../../mapper/src/ir/dfg_ir.cpp),
lines 300--335 and the corresponding LLVM-CDFG path at lines 472--475).
`DFGIONode` itself documents the I/O operation family, including `CSTORE`
([`mapper/include/dfg/dfg_node.h`](../../mapper/include/dfg/dfg_node.h),
lines 75--76).

`CSTORE` is an output/store node. `DFG::getOutNodes()` includes it alongside
`OUTPUT` and `STORE`, so the configuration paths select store mode for a
mapped CSTORE I/O node ([`mapper/src/dfg/dfg.cpp`](../../mapper/src/dfg/dfg.cpp),
lines 125--135; [`mapper/src/mapper/configuration/configuration.cpp`](../../mapper/src/mapper/configuration/configuration.cpp),
lines 315--323).

The mapper's source-level logical signature is:

```
CSTORE: void store(data, addr, en)
```

as documented in [`mapper/include/dfg/dfg_node.h`](../../mapper/include/dfg/dfg_node.h),
line 18. The JSON DFG parser transfers each edge's `operand` value (or
`headport`'s `inN` suffix) directly to the destination port
([`mapper/src/ir/dfg_ir.cpp`](../../mapper/src/ir/dfg_ir.cpp), lines 371--405).
Therefore the frontend/CDFG contract must normalize a CSTORE's operands as
`data=0`, `address=1`, and `enable=2`; a frontend must emit those destination
port indices rather than rely on edge order.

## Configuration contract

Both normal and ping-pong I/O configuration paths program CSTORE as follows:

- If the ADG exposes `UseAddr`, the mapper writes `1` for `CSTORE` (as it does
  for `LOAD`, `STORE`, and `CLOAD`).
- If the ADG exposes `UseEn`, the mapper writes `1` only for `CLOAD` and
  `CSTORE`; it writes `0` for other operations.

The existence checks are important: neither field is assumed to be present.
See [`mapper/src/mapper/configuration/configuration.cpp`](../../mapper/src/mapper/configuration/configuration.cpp),
lines 425--447, and [`mapper/src/mapper/configuration/pingpongCfg.cpp`](../../mapper/src/mapper/configuration/pingpongCfg.cpp),
lines 142--164.

## Available operation and hardware artifacts

The three checked operation catalogs contain a `STORE` entry but no `CSTORE`
entry:

| Catalog | `STORE` | `CSTORE` |
| --- | --- | --- |
| [`test/spec/cgra_fp32/operations_fp32.json`](../../test/spec/cgra_fp32/operations_fp32.json) | line 187 | absent |
| [`test/spec/cgra_bf16/operations.json`](../../test/spec/cgra_bf16/operations.json) | line 251 | absent |
| [`lib/DFG/Documents/operations20241118.json`](../../lib/DFG/Documents/operations20241118.json) | line 187 | absent |

The fp32, bf16, and documented ADGs each describe the I/O block with
`iob_mode=1`, `num_operands=2`, and a configuration map containing `UseAddr`
but no `UseEn`:

| ADG | Evidence |
| --- | --- |
| [`test/spec/cgra_fp32/cgra_adg_fp32.json`](../../test/spec/cgra_fp32/cgra_adg_fp32.json) | lines 1771--1824, 1838 |
| [`test/spec/cgra_bf16/vitra_cgra_adg.json`](../../test/spec/cgra_bf16/vitra_cgra_adg.json) | lines 8782--8839, 8853 |
| [`lib/DFG/Documents/cgra_adg20241118.json`](../../lib/DFG/Documents/cgra_adg20241118.json) | lines 1771--1824, 1838 |

Consequently, the current hardware artifacts cannot validate conditional-store
suppression: their I/O block only accepts two operands and has no `UseEn`
configuration bit for the enable input.

## Repository-history check

This audit enumerated all 17 local heads, remote-tracking refs, and tags with
`git for-each-ref refs/heads refs/remotes refs/tags`, then searched every tip
for the following hardware/fixture artifacts:

- a JSON operation entry named `CSTORE`;
- a JSON `iob_mode` equal to `2`;
- a JSON `UseEn` field; and
- `CSTORE` in test DFG-style `.dot`, `.json`, or `.mlir` fixtures (excluding
  operation and ADG catalog hits).

All four searches returned no result. The mapper source does contain dormant
CSTORE/UseEn handling described above; the negative result is specifically
that no local or remote ref supplies a usable CSTORE operation entry, a
three-input (`iob_mode=2`) I/O block with `UseEn`, or a CSTORE DFG fixture to
exercise it.

## Delivery boundary

**Stage A may proceed** with ADORA IR support, control-flow lowering,
normalized CDFG ports, and regression tests. These are frontend and mapper
contract work that can be checked without claiming an enabled hardware store.

**Stage B is blocked** until the repository has all of the following:

1. A real `CSTORE` operation specification.
2. A three-input I/O block (`iob_mode=2`) exposing `UseEn`.
3. Mapper placement/configuration against that ADG.
4. Simulation demonstrating that `enable=0` suppresses the store and
   `enable=1` performs it.

Until then, mapping or compiling a CSTORE must not be treated as proof of
conditional-store hardware behavior.

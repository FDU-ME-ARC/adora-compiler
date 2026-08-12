# CSTORE backend contract audit

**Scope.** This is a source and repository-history audit of the conditional
store (`CSTORE`) backend contract. It records what the mapper can represent
today and the separate hardware-spec gap that prevents an end-to-end CSTORE
claim. It does not add a CSTORE operation specification or change the hardware
interface.

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

## CDFG I/O metadata contract

A CSTORE is serialized as an I/O node as well as a three-input operation. Its
DOT/LLVM-CDFG record carries the same memory identity and footprint fields used
by other memory I/O nodes:

- `ref_name` identifies the target (`KernelName:argN` for function block
  arguments, or the established `BlockLoad`/`LocalMemAlloc` ID for local
  buffers);
- `size` is the full memref size in bytes;
- `offset` is initialized as `0,0`; and
- `pattern` is `0,1`, denoting one explicit scalar address per invocation.

The pattern is metadata, not an implicit affine address. CSTORE address port 1
therefore remains connected and byte-scaled in the CDFG. The direct
LLVM-CDFG-to-mapper parser consumes access-pattern fields only in complete,
non-empty pairs, so malformed or absent pairs are never indexed past the end.
CDFG generation also verifies, before writing a success DOT, that every CSTORE
has exactly one connected data, byte-address, and enable input (ports 0, 1,
and 2 respectively) and complete memory metadata.

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
normalized CDFG ports and metadata, and regression tests. These are frontend
and mapper-contract checks that do not claim an enabled hardware store. The
current lowering deliberately fails closed at these boundaries:

- branch-local loads are unsupported because conditional loads are outside
  Stage A; a read is not speculatively moved across a write;
- branch operations other than supported stores must be both memory-effect
  free and speculatable; calls, copies, nested loops/regions, and other effects
  are rejected before any `scf.if` rewrite;
- store-bearing `scf.if` and pre-authored `ADORA.cond_store` under `scf.for`
  are rejected before lowering because their execution/address contract cannot
  be represented safely by this CDFG path; `affine.for` remains supported;
- Stage A CSTORE targets must be statically shaped rank-one memrefs. Dynamic
  rank-one and all higher-rank targets are rejected rather than serialized with
  incomplete size/address metadata;
- nested `affine.apply` address expressions are fully composed before
  arithmetic expansion; any address that still cannot be represented fails
  closed through the CDFG port postcondition;
- when a CSTORE participates in a kernel, mapped leaf memory operations are
  conservatively chained in structured lexical order, including intervening
  accesses to other memrefs and the entry/exit boundaries of nested
  `affine.for` regions;
- the optimized CDFG clone bypasses affine load/store-pair hoisting whenever it
  contains a CSTORE, because that transform does not model CSTORE as a memory
  ordering barrier; kernels without CSTORE retain the existing hoist path;
- tensor mapper execution failure propagates to the tool before tensor-op
  erasure or configuration/execution emission. The failed in-memory module may
  still contain temporary lowered loop IR beside the unerased tensor op, but
  the caller exits without serializing or emitting it; and
- every rejection diagnoses the unsupported operation, fails the DFG pass,
  and emits no success DOT for that kernel. Validation precedes normalization,
  and lowering plus both optimized/fallback CDFG attempts run on temporary
  kernels, so a failure leaves the original kernel unchanged.

**Stage B is blocked** until the repository has all of the following:

1. A real `CSTORE` operation specification.
2. A three-input I/O block (`iob_mode=2`) exposing `UseEn`.
3. Mapper placement/configuration against that ADG.
4. Simulation demonstrating that `enable=0` suppresses the store and
   `enable=1` performs it.

Until then, mapping or compiling a CSTORE must not be treated as proof of
conditional-store hardware behavior.

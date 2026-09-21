# CSTORE backend contract audit

**Scope.** This is a source, fixture, and regression audit of the conditional
store (`CSTORE`) backend contract. Its detailed direct loop-free placement,
routing, and input-alignment results preserve the historical Stage A baseline.
The branch has since completed loop-index ACC lowering and package-only
full-CGRA execution; the authoritative current status and final acceptance
evidence are recorded in
[`cstore-loop-index-e2e.md`](cstore-loop-index-e2e.md).

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

## Validated configuration packet contract

The normal and ping-pong I/O configuration paths contain the following CSTORE
branches:

- If the ADG exposes `UseAddr`, the mapper writes `1` for `CSTORE` (as it does
  for `LOAD`, `STORE`, and `CLOAD`).
- If the ADG exposes `UseEn`, the mapper writes `1` only for `CLOAD` and
  `CSTORE`; it writes `0` for other operations.

The existence checks mean neither field is assumed to be present for IOBs that
do not advertise CSTORE. A CSTORE-capable IOB is validated separately and must
provide a well-formed `UseEn` field. See
[`mapper/src/mapper/configuration/configuration.cpp`](../../mapper/src/mapper/configuration/configuration.cpp),
and
[`mapper/src/mapper/configuration/pingpongCfg.cpp`](../../mapper/src/mapper/configuration/pingpongCfg.cpp),
for the two encoders. The current mapper dispatches IOB configuration through
`getIobPingpongCfgData(..., false)`, so the emitted single-phase values exercise
the latter path; runtime phase switching is not claimed.

The focused black-box regression decodes the pinned ADG's configuration IDs,
bit ranges, and packet addresses rather than duplicating numeric offsets. For
each successful mapping it checks that every address/value pair in `config.bit`
matches the generated `cfgbit_<kernel>` array. It then proves the following
decoded values:

- a mapped CSTORE has `IsStore=1`, `UseAddr=1`, and `UseEn=1`;
- a mapped normal STORE has `IsStore=1`, `UseAddr=1`, and `UseEn=0`;
- mapped INPUT nodes keep all three controller bits clear;
- each IOB input mux selects the physical input recorded in
  `mapped_routes.tsv`; and
- the DelayPipe field packs every routed input's recorded alignment delay into
  its ADG-defined lane.

The checks run for CSTORE seeds 7, 19, and 101 and a repeated seed 7. The two
seed-7 runs produce byte-identical route manifests and configuration packets.

## Available operation and hardware artifacts

The legacy checked operation catalogs contain a `STORE` entry but no `CSTORE`
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

Consequently, the legacy default artifacts cannot validate conditional-store
suppression: their I/O block only accepts two operands and has no `UseEn`
configuration bit for the enable input. They remain the untouched defaults for
the existing fp32/bf16 flows.

The ingestion phase additionally introduced an opt-in, tracked regression fixture at
[`test/spec/cgra_cstore_vitra/`](../../test/spec/cgra_cstore_vitra/). It is a
minimal regression subset of the final VITRA handoff generated from clean
`MIONkb/VITRA-CGRA@da03f4ab0cf696466147ac9210518e7ead6c9589`, not a replacement
for any legacy catalog. Its canonical clean RTL SHA-256 is
`3dfdfe954ae2b6614d7e3a7e3b08c3feb9d899f0fef9d4f4e1eee230d24fbbc0`.
The tracked semantic payload hashes are:

| Payload | SHA-256 |
| --- | --- |
| `artifacts/operations.json` | `0eee215afdbed65fed6bd7773f40a783189d67a66accb6e7bc86aaac582bae8d` |
| `artifacts/adg.json` | `e34fcef5b15f47718762812513d3cd7db35f06e60a17097e50429dfc41e26cf7` |
| `artifacts/loop_index_contract.json` | `c3d358639ef9470f5abb5c3a8146e75c6920c6055e28dfba780868073312489e` |

The fixture advertises exactly `INPUT`, `OUTPUT`, `LOAD`, `STORE`, and
`CSTORE`; it deliberately does not advertise `CLOAD`. When an IOB JSON object
has `attributes.operations`, the ADG parser treats that field as authoritative:
only its string entries are capabilities, and an explicit empty or malformed
value contributes no mode-derived fallback. The legacy FIFO/SRAM/conditional
mode fallback runs only when the field is absent. Thus the fixture's explicit
list neither gains `CLOAD` by inference nor changes the untouched legacy
defaults, whose operation lists are absent.

## Repository-history check

At the start of Stage A, this audit enumerated the available local heads,
remote-tracking refs, and tags with `git for-each-ref refs/heads refs/remotes
refs/tags`, then searched the pre-feature refs for the following
hardware/fixture artifacts:

- a JSON operation entry named `CSTORE`;
- a JSON `iob_mode` equal to `2`;
- a JSON `UseEn` field; and
- `CSTORE` in test DFG-style `.dot`, `.json`, or `.mlir` fixtures (excluding
  operation and ADG catalog hits).

All four searches returned no result at that point. The repository now contains
the opt-in tracked VITRA fixture above, alongside the compiler/CDFG MLIR
fixtures introduced during Stage A; the legacy catalogs remain unchanged. The
direct loop-free regression established placement, three-port routing, input
alignment, and configuration decoding. Subsequent work lowered supported
`affine.for` induction values to physical ACCs, so the original real
`if_store` workload no longer presents a physical FOR to the mapper. The final
compiler-generated configurations have also executed successfully on the full
CGRA through the immutable package replay.

## Historical Stage A delivery boundary

Stage A delivered ADORA IR support, control-flow lowering, normalized CDFG
ports and metadata, and mapper-contract regressions. At that checkpoint the
evidence did not yet claim enabled hardware execution. Several safety
boundaries established there still apply:

- branch-local loads are unsupported because conditional loads are outside
  Stage A; a read is not speculatively moved across a write;
- branch operations other than supported stores must be both memory-effect
  free and speculatable; calls, copies, nested loops/regions, and other effects
  are rejected before any `scf.if` rewrite;
- store-bearing `scf.if` and pre-authored `ADORA.cond_store` under `scf.for`
  are rejected before lowering because their execution/address contract cannot
  be represented safely by this CDFG path; the supported subset uses
  `affine.for`;
- CSTORE targets must be statically shaped, identity-layout rank-one
  memrefs. Dynamic rank-one, non-identity-layout, and all higher-rank targets
  are rejected rather than serialized with incomplete size/address metadata;
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

## Historical direct mapping evidence

The focused mapper regression uses a direct loop-free CSTORE whose data,
byte-address, and enable values are all dynamic. Their CDFG producer depths are
respectively one, two, and three supported operations. Before each mapping
series, the regression requires exactly one CSTORE and exactly one incoming
logical edge at each of ports 0, 1, and 2.

The pinned VITRA ADG and operation catalog map that graph with seeds 7, 19,
and 101, always using `--obj-opt=false --max-iters=30 --timeout=30000`. Every
successful route manifest contains exactly three CSTORE rows. Logical ports
0, 1, and 2 use the disjoint physical input sets `{0,1}`, `{2,3}`, and `{4,5}`
respectively; every arrival latency equals its target latency. The runs include
nonzero RDU delay, proving that the alignment check is not limited to
equal-latency routes. Repeating seed 7 produces a byte-identical manifest.

A direct loop-free normal STORE maps with the same fixture and has no logical
operand 2 route. Removing CSTORE from a temporary ADG copy still permits that
STORE while rejecting the CSTORE with the explicit required/available
capability diagnostic. The hash-pinned fixture itself is never modified.

This direct-case evidence proves placement, routing, mapper-level latency
alignment, and the emitted controller, mux, and DelayPipe packet values. It was
the Stage A mapper boundary, not the final project boundary.

## Current executable boundary

The supported loop-bearing path now lowers a logical `affine.for` induction
value to a mapper-visible physical ACC, routes its positive step as operand 0,
and inserts explicit element-byte multiplication before CSTORE address port 1.
Physical FOR remains intentionally unsupported as a hardware operation; it is
absent from the supported mapper-visible DFG.

The mapper decodes the loop-index ACC independently of generic accumulation and
emits `InitVal=lower_bound`, `WI=1`, `Latency=0`,
`Cycles=trip_count`, `Repeats=1`, and `SkipFirst=1`. The CSTORE controller
remains `IsStore/UseAddr/UseEn=1/1/1`; normal STORE remains `1/1/0`.

Compiler-generated configurations for canonical and nontrivial loop bounds,
alternating predicates, and the original real `if_store` workload passed
package-only execution on the final full-CGRA RTL. This proves true writes,
false suppression, completion, and final SRAM side effects for the supported
v1 subset. It does not add physical FOR, CLOAD, negative-step, dynamic-bound,
or generalized nested-loop support, and it does not claim runtime ping-pong
phase switching. The exact final evidence is in
[`cstore-loop-index-e2e.md`](cstore-loop-index-e2e.md).

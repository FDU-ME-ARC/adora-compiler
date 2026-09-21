# Loop-index CSTORE full-CGRA validation

## Status and source identities

The supported ADORA loop-bearing conditional-store path is complete and was
validated end to end from production compiler input through full-CGRA SRAM
side effects.

| Source | Identity |
| --- | --- |
| ADORA | `cfb0053e67a5a214637a710a9b6be2fe4f46ab82` |
| VITRA-CGRA | `da03f4ab0cf696466147ac9210518e7ead6c9589` |
| Canonical clean `CGRAWithAXI.v` SHA-256 | `3dfdfe954ae2b6614d7e3a7e3b08c3feb9d899f0fef9d4f4e1eee230d24fbbc0` |
| VITRA loop-index contract SHA-256 | `c3d358639ef9470f5abb5c3a8146e75c6920c6055e28dfba780868073312489e` |

The tracked fixture at
[`test/spec/cgra_cstore_vitra/`](../../test/spec/cgra_cstore_vitra/) is the
minimal hash-pinned regression subset of the final VITRA package. Its manifest,
ADG, operation catalog, generator input, loop-index contract, and hardware
validation summary all pass `SHA256SUMS`. The full immutable package, rather
than a patched fixture or hand-edited RTL, was used for heavyweight execution.
The full package manifest SHA-256 was
`67fb7c89ec427555a16c12cc0b97c13b1c9b5a6df341137191e49aa124ca6786`,
and the package `SHA256SUMS` file SHA-256 was
`cfc9dc8c4065e7186e7729d486a9db50be28cdc9a3df722b2411fdde9b8eb333`.

Fresh compiler output used mapper seed 7 with `--obj-opt=false`,
`--max-iters=30`, `--timeout=30000`, and pytest output. The pinned ADG and
operation catalog SHA-256 values were respectively
`e34fcef5b15f47718762812513d3cd7db35f06e60a17097e50429dfc41e26cf7`
and `0eee215afdbed65fed6bd7773f40a783189d67a66accb6e7bc86aaac582bae8d`.

## Compiler-to-hardware path

For the supported subset, the compiler preserves `affine.for` as a logical
control construct and lowers the induction value before physical mapping:

```text
affine.for induction value
  -> physical loop-index ACC
  -> explicit MUL(elementBytes)
  -> CSTORE address port 1
```

No physical FOR is introduced or retained in the mapper-visible DFG. VITRA
still intentionally does not advertise FOR as a hardware operation; the
compiler lowering is what makes the supported loop-bearing path executable.
The static positive step is materialized by a real routable producer and reaches
ACC operand 0. It is not represented by a fake runtime function argument.

The CSTORE logical ports and controller fields remain:

| Meaning | Contract |
| --- | --- |
| data | operand port 0 |
| byte address | operand port 1 |
| predicate | operand port 2, truth in bit 0 |
| CSTORE controller | `IsStore/UseAddr/UseEn = 1/1/1` |
| normal STORE controller | `IsStore/UseAddr/UseEn = 1/1/0` |

The loop-index specialization does not change the configuration semantics of
ordinary accumulation ACC nodes.

## Exact ACC contract

The emitted configuration is decoded using configuration IDs and bit ranges
from the pinned ADG, and the routed step is checked against the mapped route.
For Case B (`LB=3`, `UB=13`, `STEP=2`, two-byte elements), the result is:

| Field | Decoded value |
| --- | ---: |
| `InitVal` | 3 |
| routed operand 0 | 2 |
| `WI` | 1 |
| `Latency` | 0 |
| `Cycles` | 5 |
| `Repeats` | 1 |
| `SkipFirst` | 1 |

The ACC sequence is `3, 5, 7, 9, 11`. The explicit multiply by two produces
byte addresses `6, 10, 14, 18, 22`, and the MUL output is routed to CSTORE
address port 1.

## Supported v1 subset and fail-closed boundary

The supported loop-index subset has constant `affine.for` bounds, no
loop-carried values, one or more and at most 4095 iterations, an unsigned
16-bit lower bound, a positive nonzero unsigned 16-bit step, and no logical
16-bit wrap before the final iteration. The induction value may feed the
supported CSTORE address form directly or through the recognized canonical
linear `affine.apply` normalization. CSTORE memory targets remain statically
shaped, identity-layout rank-one memrefs.

The compiler fails closed instead of emitting a malformed ACC for empty,
negative-step, dynamic-bound, wrapping, nested CSTORE-related, or generalized
multidimensional loops; loop-carried values; division/modulo and other
nonlinear address transforms; or additional live induction-value uses. The
`scf.for` conditional-store path also remains unsupported. Physical FOR and
CLOAD remain unadvertised and out of scope.

## Permanent compiler and mapper regressions

The following lit-discovered tests exercise the production compiler and mapper
path without making the normal test suite depend on a heavyweight RTL build:

- [`full_pipeline_loop_index_acc.mlir`](../../test/cgra-mapper/cstore_contract/full_pipeline_loop_index_acc.mlir)
  locks Case B, the ACC configuration, routed step, explicit byte scaling, and
  CSTORE ports.
- [`full_pipeline_loop_alternating.mlir`](../../test/cgra-mapper/cstore_contract/full_pipeline_loop_alternating.mlir)
  locks a four-iteration `1,0,1,0` predicate data path.
- [`full_pipeline_real_if_store.mlir`](../../test/cgra-mapper/cstore_contract/full_pipeline_real_if_store.mlir)
  locks the original control-flow workload and rejects any residual physical
  FOR diagnostic.
- [`loop_index_acc_mapping.mlir`](../../test/cgra-mapper/cstore_contract/loop_index_acc_mapping.mlir)
  decodes Case A and Case B configurations and preserves generic ACC and normal
  STORE behavior.
- [`cstore_loop_index_acc.mlir`](../../test/cgra-opt/cdfggen/cstore_loop_index_acc.mlir)
  covers supported lowering and the fail-closed subset at CDFG generation.

At the final ADORA source identity above, `check-adora` reports:

```text
Total Discovered Tests: 64
Unsupported: 8
Passed: 56
Failed: 0
```

## Full-CGRA execution evidence

All hardware cases used fresh compiler-generated CDFG, mapped routes, and
`config.bit` output. None of those artifacts was hand-edited. The incoming
package checksum was verified before package-only replay, and execution used
the canonical clean RTL hash recorded above.

| Case | Compiler/configuration result | Full-CGRA SRAM result |
| --- | --- | --- |
| Case A: `0..4 step 1`, two-byte elements, alternating predicate `1,0,1,0` | ACC indices `0,1,2,3`; byte addresses `0,2,4,6`; config SHA-256 `8e5fe194d55afb3689acb903a66cde7da2912cd8f7e779ecfaf1a9a9c0b414b8` | writes only at byte offsets 0 and 4; suppressed iterations perform no target write; PASS |
| Case B true: `3..13 step 2`, two-byte elements | exact ACC contract above; config SHA-256 `27f42a3016111a1b488fa83cca2990490e9842463fadedefa9c8dc11518ab7f6` | five writes at byte offsets `6,10,14,18,22`, each storing `0x1234`; PASS |
| Case B false | exact same compiler configuration as true, predicate bit 0 clear | zero target writes; all five OLD `0xbeef` values preserved; PASS |
| real `if_store`: `0..16 step 1`, four-byte elements | physical ACC plus byte-scaling MUL; config SHA-256 `554c73ac172796d0465cd3619198c636bbf5f4e2cf1ba6b1c1bf10c53e7b7751` | 16 writes at byte offsets 0 through 60 in steps of 4, each storing 7; PASS |

Each replay loaded configuration, started execution, reached normal completion,
and matched the expected final memory. The real workload progressed through
normalization, CDFG generation, mapping, configuration, and full-CGRA execution;
it no longer stops at the historical mapper diagnostic `FOR is not supported!`.

## Remaining limitations

This validation does not broaden the v1 contract to CLOAD, physical FOR,
negative or zero step, dynamic bounds, empty iteration spaces, logical index
wrap, generalized nested loops, multidimensional/non-identity targets,
nonlinear index transforms, or extra live induction-value consumers. Runtime
ping-pong phase switching is also not part of this proof. These cases remain
explicitly unsupported or fail closed.

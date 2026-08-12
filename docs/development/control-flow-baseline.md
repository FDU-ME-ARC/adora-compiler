# ADORA Control-Flow Baseline

## Baseline

- Date: 2026-08-12
- Compiler commit: `e88f070`
- Test entrypoint: `experiment/jyhu/control-flow/run.sh all`
- Fixed input: hand-maintained MLIR paired with equivalent minimal C sources
- Frontend status: `cgeist` is not installed in the current environment, so C-to-MLIR is recorded as `SKIP`
- Mapper configuration: `test/spec/cgra_fp32/{cgra_adg_fp32.json,operations_fp32.json}` with `--max-iters=1 --timeout=10000`
- Generated IR, DOT, and logs: `experiment/jyhu/control-flow/out/` (ignored by Git)

The fixed MLIR path is authoritative for this baseline. C sources record the intended source semantics and can be compared with frontend output when `cgeist` becomes available.

## Existing implementation

The normal `adoracc.py` path normalizes the input, extracts affine loops into `ADORA.kernel`, optimizes block access, and finally invokes `--adora-kernel-dfg-gen`.

Control-flow conversion is not a standalone pass. `generateCDFGfromKernel()` in `lib/DFG/DFGgen.cpp` mutates the kernel immediately before graph construction:

1. `lowerSCFIfToSelect()` collects every `scf.if` in the kernel.
2. Result-producing `scf.if` regions have their non-yield operations moved before the if and each yielded result becomes an `arith.select`.
3. A result-less if containing stores is rewritten by store sinking:
   - a one-sided store becomes an old-value load, select, and unconditional store;
   - matching stores in both branches become one select and one unconditional store.
4. `InsertIselForLoopCarry()` subsequently inserts `ADORA.isel` for non-accumulation loop-carried values.
5. `GeneralOpName.txt` maps `arith.select` to `SEL` and `ADORA.isel` to `ISEL`.

Important boundaries found by code inspection:

- Only `scf.if` has explicit if-conversion in the CDFG path.
- `affine.if` may be carried through kernel extraction, but has no corresponding conversion or operation-name mapping in CDFG generation.
- `cf.cond_br` is registered with the driver through the standard SCF-to-CF pass, but the `adoracc`/CDFG pipeline does not run that conversion and has no `cf.cond_br` graph mapping.
- `arith.select` is directly representable as `SEL`.
- `ADORA.isel` represents loop-carried state selection; it is not the general branch predicate representation.

## Results

`Pre-DFG IR` is the final MLIR exported by `adoracc`. `Post-DFG rewrite` is the in-memory form produced by running `--adora-kernel-dfg-gen`; this distinction matters because the DFG pass performs the if-conversion itself.

| Case | Pre-DFG IR | Post-DFG rewrite / CDFG | Mapper | Assessment |
|---|---|---|---|---|
| `if_simple` | one-sided, result-less `scf.if`; memory-footprint optimization separates the external `BlockLoad` from an uninitialized local output allocation | local old-value `Input + SEL + Output`; predicate port 0 has no edge | PASS | Incorrect: the false path reads the uninitialized local output allocation instead of preserving the external value; the first semantic loss occurs during memory-footprint optimization, before DFG rewriting |
| `if_else` | already canonicalized to one `arith.select` | `SEL + Output`; predicate port 0 has no edge | PASS | Partial: value selection shape is correct, predicate data is absent from CDFG |
| `if_elseif_else` | outer `scf.if`, inner branch already `arith.select` | `select(a, 11, select(b, 22, 33))`; both predicate edges are absent | PASS | Partial: MLIR encodes `a`, `!a && b`, `!a && !b` correctly, but the CDFG cannot receive `a` or `b` |
| `nested_if` | canonicalization changes nested ifs to `arith.andi a, b` plus one `scf.if` | `AND + old-value Input + SEL + Output`; `AND` has no input edges | FAIL: `AND is not supported!` | Partial: MLIR path condition is `a && b`; CDFG loses both function-argument inputs and fp32 mapper spec lacks `AND` |
| `if_compute` | result-producing `scf.if` | both pure calculation branches are hoisted and feed one `SEL`; predicate edge absent | PASS | Partial: selected value is correct for pure operations, but both branches execute unconditionally |
| `if_load` | the local `affine.load` remains in the true region, but memory-footprint optimization has already emitted an unconditional external `ADORA.BlockLoad` | the local load is then hoisted before `SEL`; predicate edge absent | PASS | Incomplete: external memory traffic first becomes unconditional during memory-footprint optimization, and DFG rewriting also removes the conditional local-load behavior |
| `if_store` | one-sided store | local old-value `Input + SEL + Output`; predicate edge absent | PASS | Incorrect for general output buffers: memory-footprint optimization creates `LocalMemAlloc` without loading the external old value, so false does not reliably preserve memory |
| `if_else_store` | same-address store in both branches | one `SEL + Output`, with no old-value load; predicate edge absent | PASS | Partial: store merging is correct, but the condition is disconnected |
| `loop_if` | `cmpi + scf.if` inside an `affine.for iter_args` | `SLT + ISEL + ADD + SEL + Output`; predicate and loop backedge are present | FAIL: `SLT is not supported!` | Compiler/CDFG path works; mapper fp32 operation spec lacks `SLT` |

Observed CDFG opcode sets:

| Case | Opcodes |
|---|---|
| `if_simple` | `Input×2, ADD, CONST, SEL, Output` |
| `if_else` | `CONST×2, SEL, Output` |
| `if_elseif_else` | `CONST×3, SEL×2, Output` |
| `nested_if` | `AND, Input, CONST, SEL, Output` |
| `if_compute` | `Input, MUL, ADD×2, CONST×3, SEL, Output` |
| `if_load` | `Input, CONST, SEL, Output` |
| `if_store` | `Input, CONST, SEL, Output` |
| `if_else_store` | `CONST×2, SEL, Output` |
| `loop_if` | `ISEL, Input, SLT, ADD, CONST, SEL, Output` |

### Path-condition conclusions

For the else-if case, the post-conversion MLIR is algebraically correct:

```text
select(a, S1, select(b, S2, S3))
S1 -> a
S2 -> !a && b
S3 -> !a && !b
```

For the nested case, canonicalization explicitly forms `a && b` before DFG generation. However, function arguments of type `i1` are not materialized as CDFG input nodes, so the `SEL` predicate ports—or the inputs of the nested `AND`—remain disconnected. Mapper PASS on the simple cases therefore means only that the remaining graph can be placed; it does not prove executable predicate semantics.

The loop case computes its predicate from an in-kernel load. That comparison is represented and connected correctly, demonstrating that the missing predicate edges are specifically associated with values entering the kernel as non-index block arguments.

## Known gaps and follow-up ownership

### Task two: path conditions and control-flow correctness

- Define how kernel scalar/i1 arguments become CDFG inputs and connect them to predicate operand 0.
- Preserve or explicitly compose path predicates for nested and else-if structures.
- Process nested `scf.if` operations in a mutation-safe order.
- Restrict speculative hoisting to operations proven safe; loads and other side effects must not be unconditionally moved merely because the if returns a value.
- Decide and test the intended handling or rejection of `affine.if` and `cf.cond_br`.

### Task three: conditional memory operations

- Replace one-sided store emulation with the planned `ADORA.cond_store`/`CSTORE` chain rather than reading an uninitialized local output buffer and issuing an unconditional store.
- Preserve address, value, and enable operand ports explicitly.
- Verify the mapper/IOB operation specs include the required `CSTORE` and predicate operations.

### Mapper/spec limitations

- The selected fp32 operation file lacks `AND` and `SLT`; those failures are separate from successful compiler/CDFG generation.
- A future mapper baseline should distinguish unsupported opcodes from malformed or disconnected graphs before reporting overall success.

## Reproduction

```bash
# Run all cases. Stage failures are recorded without stopping later cases.
experiment/jyhu/control-flow/run.sh all

# Run one case and replace summary.tsv with that single result.
experiment/jyhu/control-flow/run.sh if_elseif_else

# Existing simple-if CDFG regression.
cmake --build build --target check-adora-cgra-opt-cdfggen-gettanh
```

Each failure is reproducible from the command log under its case output directory. The runner copies `input.mlir` before invoking `adoracc.py` because that driver cleans its MLIR input in place.

The runner is report-oriented: case-level `FAIL`/`SKIP` results are written to `summary.tsv`, but the command still exits zero after completing the requested cases. A nonzero exit is reserved for invocation errors or missing required infrastructure. Missing `cgeist` is a deliberate frontend `SKIP` and does not downgrade the fixed-MLIR Overall result; an installed frontend that fails does make Overall fail.

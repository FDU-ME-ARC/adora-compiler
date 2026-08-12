# ADORA Control-Flow Baseline

## Baseline

- Date: 2026-08-12
- Compiler commit: `4ef8fc6` (`DFG: tighten scalar input and DOT checks`)
- Experimental entrypoint: `experiment/jyhu/control-flow/run.sh all`
- Formal CDFG regressions: `check-adora-cgra-opt-cdfggen-control_flow_paths`, `check-adora-cgra-opt-cdfggen-gettanh`, and `check-adora-cgra-opt-cdfggen`
- Fixed input: hand-maintained MLIR paired with equivalent minimal C sources
- Frontend status: `cgeist` is not installed in the current environment, so C-to-MLIR is recorded as `SKIP`
- Mapper configuration: `test/spec/cgra_fp32/{cgra_adg_fp32.json,operations_fp32.json}` with `--max-iters=1 --timeout=10000`
- Generated IR, DOT, and logs: `experiment/jyhu/control-flow/out/` (ignored by Git)

The fixed MLIR path is authoritative for this baseline. C sources record the intended source semantics and can be compared with frontend output when `cgeist` becomes available.

## Existing implementation

The normal `adoracc.py` path normalizes the input, extracts affine loops into `ADORA.kernel`, optimizes block access, and finally invokes `--adora-kernel-dfg-gen`.

Control-flow conversion is not a standalone pass. `generateCDFGfromKernel()` in `lib/DFG/DFGgen.cpp` mutates the kernel immediately before graph construction:

1. `lowerSCFIfToSelect()` collects every `scf.if` in the kernel.
2. Result-producing `scf.if` regions are lowered in postorder: non-yield operations are moved before the if and each yielded result becomes an `arith.select`. This keeps an inner select available before its enclosing select.
3. A result-less if containing stores is rewritten by store sinking:
   - a one-sided store becomes an old-value load, select, and unconditional store;
   - matching stores in both branches become one select and one unconditional store.
4. `InsertIselForLoopCarry()` subsequently inserts `ADORA.isel` for non-accumulation loop-carried values.
5. Captured integer and floating-point function-entry arguments, including `i1` predicates, are materialized once per function argument as synthetic CDFG `Input` nodes. Their stable metadata uses `<kernel>:arg<index>`, byte size `max(1, ceil(bitwidth / 8))`, offset `0,0`, and pattern `0,1`.
6. `GeneralOpName.txt` maps `arith.select` to `SEL` and `ADORA.isel` to `ISEL`. Every final `SEL` has false value at port 0, true value at port 1, and condition at port 2.

Important boundaries found by code inspection:

- Only `scf.if` has explicit if-conversion in the CDFG path.
- `affine.if` may be carried through kernel extraction, but has no corresponding conversion or operation-name mapping in CDFG generation.
- `cf.cond_br` is registered with the driver through the standard SCF-to-CF pass, but the `adoracc`/CDFG pipeline does not run that conversion and has no `cf.cond_br` graph mapping.
- `arith.select` is directly representable as `SEL`; nested SEL value-commit structure carries structured path semantics. It does not add a per-operation predicate annotation, and pure branch calculations may still execute speculatively.
- `ADORA.isel` represents loop-carried state selection; it is not the general branch predicate representation.

## Task 2 results

The formal `control_flow_paths` lit regressions cover the four Task 2 completion cases. Each checks that the rewritten kernel has no remaining `scf.if`, has the expected SEL count, and has connected captured predicate Inputs at SEL condition port 2. The DOT checks also reject undefined and CTRL opcodes.

| Case | CDFG path-condition evidence | Result |
|---|---|---|
| one-sided `if` | One SEL; old value -> port 0, true value -> port 1, captured `i1` -> port 2 | PASS |
| `if-else` | One SEL; captured predicate and scalar value Inputs, with false/true/condition ports 0/1/2 | PASS |
| `if-else if-else` | Two SELs; `b` feeds the inner condition, `a` the outer condition, and the inner result is the outer false value (port 0) | PASS |
| two-level nested `if` | Two SELs; `b` feeds the inner condition, `a` the outer condition, and the inner result is the outer true value (port 1) | PASS |

The focused target reported 4/4 passing tests. The required `gettanh` regression reported 1/1 passing, and the complete CDFG test group reported 9/9 passing. These are compiler/CDFG regressions, not mapper-placement claims.

### Path-condition conclusions

For the else-if case, the nested SEL structure is algebraically correct:

```text
select(a, S1, select(b, S2, S3))
S1 -> a
S2 -> !a && b
S3 -> !a && !b
```

For the nested case, the inner result is committed through the outer true arm, which represents `a && b` without requiring a separate branch-predicate dialect or an explicit AND node. Captured `i1` function arguments now reach the CDFG as reusable synthetic Inputs, so the required predicate edges are no longer disconnected.

The nesting expresses which value commits on each path; it does not make pure calculations control-dependent. Branch calculations that are safe to speculate can run before the SELs, while conditional memory operations require separate handling.

## Known gaps and follow-up ownership

### Task two: path conditions and control-flow correctness

- Structured `scf.if` result paths are represented by nested SEL commits, with captured scalar and `i1` function arguments feeding SEL condition port 2. Postorder lowering preserves the required inner-to-outer ordering.
- Conditional load is still a known limitation: existing lowering/memory-footprint processing can make a load unconditional, and Task 2 did not add a conditional-load representation.
- `affine.if`, `cf.cond_br`, switch, break, continue, and unstructured CFG remain out of scope and have no equivalent CDFG control-flow implementation.

### Task three: conditional memory operations

- Conditional store remains Task 3: replace one-sided store emulation with the planned `ADORA.cond_store`/`CSTORE` chain rather than reading an uninitialized local output buffer and issuing an unconditional store.
- Preserve address, value, and enable operand ports explicitly.
- Verify the mapper/IOB operation specs include the required `CSTORE` and predicate operations.

### Mapper/spec limitations

- The selected fp32 operation file lacks `AND` and `SLT`; these mapper capability gaps remain recorded and are not bypassed by hardware-spec changes.
- A future mapper baseline should distinguish unsupported opcodes from malformed or disconnected graphs before reporting overall success.

## Reproduction

```bash
# Run all experimental cases. Stage failures are recorded without stopping later cases.
experiment/jyhu/control-flow/run.sh all

# Run one experimental case and replace summary.tsv with that single result.
experiment/jyhu/control-flow/run.sh if_elseif_else

# Formal Task 2 CDFG regressions.
cmake --build build --target check-adora-cgra-opt-cdfggen-control_flow_paths -- -j1
cmake --build build --target check-adora-cgra-opt-cdfggen-gettanh -- -j1
cmake --build build --target check-adora-cgra-opt-cdfggen -- -j1
```

Each experimental failure is reproducible from the command log under its case output directory. The runner copies `input.mlir` before invoking `adoracc.py` because that driver cleans its MLIR input in place.

The runner is report-oriented: case-level `FAIL`/`SKIP` results are written to `summary.tsv`, but the command still exits zero after completing the requested cases. A nonzero exit is reserved for invocation errors or missing required infrastructure. Missing `cgeist` is a deliberate frontend `SKIP` and does not downgrade the fixed-MLIR Overall result; an installed frontend that fails does make Overall fail.

# ADORA Cycle Estimator

A fast cycle-count estimator for ADORA-dialect MLIR. It reuses the structured
CDFG dot produced by the compiler's `adora-kernel-dfg-gen` pass to obtain the
operator graph and loop-carried distances, reads loop trip-counts **strong-typed
via the upstream MLIR Python bindings**, and combines them with ADG hardware
parameters and a latency table to estimate a kernel's execution cycles on the
CGRA — **without running RTL simulation**.

- Design details: [`docs/DESIGN.md`](docs/DESIGN.md)
- Implementation status / current accuracy / roadmap: [`docs/STATUS.md`](docs/STATUS.md)

## Data sources (no regex parsing)

| Data | Source |
|---|---|
| Node opcode / edges / loop-carried `iterdist` / memory bytes | `_CDFG.dot` from cgra-opt (`core/dot_parser.py`) |
| Loop trip-count / element width | strong-typed `affine.for` bounds (`extract/loop_info.py` + `adora_mlir`) |
| PE array / SPAD / cfg width | ADG json (`arch/adg.py`) |
| Per-op latency | hard-coded latency table, sourced from `Operations.scala` (`core/latency_table.py`) |

> The old `mlir_loop_info.py` (regex parsing of MLIR text) and `cdfg_native.py`
> (the segfaulting `_adora_cdfg` pybind) have been removed.

## Dependencies

- Python 3
- Upstream MLIR Python bindings (`mlir.ir` / `mlir.dialects`): produced by the
  LLVM build; the path is given via the `ADORA_MLIR_CORE` environment variable
  (defaults to this host's LLVM build `python_packages/mlir_core`)
- The `adora_mlir` package (shipped with this tool, includes the
  `_adoraDialectsRegister` extension)

## Install & usage

After `ninja install`, the tool is installed under the install prefix:

- `bin/cycle-estimator`: an executable launcher with PYTHONPATH / ADORA_MLIR_CORE
  preset
- `bin/pypack/cycle_estimator/`: the importable Python package

Use it directly as a command (recommended, no environment variables needed):

```bash
cycle-estimator --mlir kernel.mlir --adg adg.json
```

Or invoke it explicitly as a package:

```bash
PYTHONPATH=<prefix>/bin/pypack \
ADORA_MLIR_CORE=<llvm-build>/python_packages/mlir_core \
python3 -m cycle_estimator --mlir kernel.mlir --adg adg.json
```

Run directly from the source tree:

```bash
ADORA_MLIR_CORE=<...>/mlir_core python3 run.py --mlir kernel.mlir --adg adg.json
```

### Options

| Option | Meaning |
|---|---|
| `--mlir` | kernel MLIR: read trip-counts; or use as the source for dot generation |
| `--dot` | an existing `_CDFG.dot` (skip cgra-opt) |
| `--adg` | ADG json: PE count / SPAD / cfg width (overrides hard-coded `num_alus`) |
| `--num-alus` | array ALU count (ignored if `--adg` is given) |
| `--num-tiles` | kernel tile count (for tile-based cfgNum, default 1) |
| `--no-overlap` | disable mem/compute overlap, fall back to serial sum |
| `--route-lat` | per-edge routing latency (0 = optimistic lower bound) |
| `--cfg-num` / `--load-bytes` / `--store-bytes` | manual overrides |

### Example output

```
kernel        : kernel_fir
II            : 1  (RecMII=1, ResMII=1)
inner_trip    : 100
outer_trip    : 1
pipeline_drain: 7
config        : 96
block_load    : 800
block_store   : 2
mem/compute   : overlap (max)
---------------------------------
TOTAL cycles  : 898  (optimistic; routeLat=0)
```

Cycle formula (overlap by default):

```
total = config_overhead + max(block_load + block_store, outer×(II×inner + drain))
```

With `--no-overlap` it falls back to serial:
`config + load + outer×(II×inner + drain) + store`.

## Module layout

```
run.py            sole top-level entry point
core/             estimation engine: dot_parser / cycle_model / ii_model / latency_table
extract/          strong-typed MLIR trip extraction: loop_info
arch/             ADG reader: adg
scripts/          standalone tools: adora_analyze / calibrate / cfgnum
adora_mlir/       official Python bindings for the ADORA dialect
docs/  tests/
```

## Tests

```bash
ADORA_MLIR_CORE=<...>/mlir_core python3 -m unittest discover -s tests
```

## Calibration (TODO, needs a simulation environment)

The current numbers are an **optimistic estimate**; absolute values are not yet
calibrated against ground truth. The calibration chain is in place
(`scripts/calibrate.py`), with ground truth from
`CGRA-Cocotb-Sim/server/test_runif.py` (RTL-level, CLOCKPERIOD=2 ns):

```bash
python3 scripts/calibrate.py --log sim.log --dot-dir <dot dir> --mlir kernel.mlir
```

See the "Roadmap" section in [`docs/STATUS.md`](docs/STATUS.md) for details.

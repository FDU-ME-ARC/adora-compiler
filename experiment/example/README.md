# ADORA Compiler — Example Kernels

Three self-contained examples that exercise the full `adoracc` pipeline:
normalize → kernel-extract → kernel-opt → task-schedule → DFG generation.

| Example | Source | Description |
|---------|--------|-------------|
| `mvt/`  | PolyBench | Matrix-Vector Transpose — two gemv loops sharing one matrix |
| `attn/` | Transformer | Scaled dot-product attention (Q·K^T → scale → ReLU·V) |
| `ffn/`  | Transformer | Feed-forward network: FC1 + ReLU + FC2 (two matmul kernels) |

---

## Prerequisites

```bash
# Build the compiler first (from project root):
cmake -S . -B build && cmake --build build -j$(nproc)
```

`adoracc.py` is located at `build/bin/adoracc.py` and requires `cgra-opt`
in the same directory.

---

## Run a single example

```bash
# MVT
adoracc.py experiment/example/mvt/mvt.mlir \
  --work-dir /tmp/mvt_out \
  -o /tmp/mvt_out/result.mlir

# Attention
adoracc.py experiment/example/attn/attn.mlir \
  --work-dir /tmp/attn_out \
  -o /tmp/attn_out/result.mlir

# FFN
adoracc.py experiment/example/ffn/ffn.mlir \
  --work-dir /tmp/ffn_out \
  -o /tmp/ffn_out/result.mlir
```

---

## Output directory layout

After a successful run each `--work-dir` contains:

```
<work-dir>/
├── result.mlir                  ← final scheduled MLIR (pipeline output)
└── adora-cc-ir/
    ├── 1_frontend/              ← cgeist output  (C input only)
    ├── 2_normalize/             ← affine-normalize pass + CDFG dot files
    ├── 3_kernel-extract/        ← ADORA.kernel extraction
    ├── 4_kernel-opt/            ← memory-footprint & math rewrites
    ├── 5_task-schedule/         ← <name>.pre.mlir / <name>.post.mlir
    │                               <name>.token_graph.dot
    ├── 6_dfg/                   ← final CDFG dot files (per kernel)
    └── pipeline.log             ← full command log
```

---

## Example: inspect the DFG

```bash
# After running MVT:
dot -Tpng /tmp/mvt_out/adora-cc-ir/6_dfg/mvt_normalized_CDFG.dot \
    -o /tmp/mvt_dfg.png
```

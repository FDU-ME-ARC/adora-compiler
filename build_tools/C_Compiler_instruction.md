# ADORA Quick Start: adoracc.py and cgra-mapper

## 1. Toolchain Overview

| Tool           | Function                              | Input       | Output                                  |
|----------------|----------------------------------------|-------------|-----------------------------------------|
| **adoracc**    | C/MLIR → optimized kernel MLIR + CDFG | .c / .mlir | Optimized MLIR, CDFG                    |
| **cgra-mapper**| CDFG/MLIR → CGRA mapping               | MLIR       | pytest (cocotb) / SDK emit / RISC-V C   |

**Repositories:**
- CGRA RTL generation (depends on Chipyard): https://github.com/MIONkb/VITRA-CGRA
- ADORA compiler (depends on MLIR; see GitHub repo for install): https://github.com/FDU-ME-ARC/adora-compiler
- Cocotb-based test environment (internal use): https://github.com/FDU-ME-ARC/MatrixMeld

**Contributors:** @楼佳杭 @zzw @陈奕宽 @张荐荣 @胡家耀

---

## 2. adoracc Usage

### 2.1 Overview

adora-cc compiles C source or MLIR input into optimized kernel IR suitable for CGRA mapping and generates CDFG (Control-Data Flow Graph).

It internally invokes `cgeist` (C→MLIR), `cgra-opt`, and other tools for normalization, kernel extraction, and optimization.

### 2.2 Invocation

```bash
adoracc.py <input> [options]
```

### 2.3 Parameters

| Parameter         | Type   | Default   | Description                                                       |
|-------------------|--------|-----------|-------------------------------------------------------------------|
| `input`           | required | -       | Input file path; supports `.c` or `.mlir`                         |
| `--work-dir`      | Path   | current dir | Working directory for IR output; creates `adora-cc-ir/` and subdirs |
| `--enable-unroll` | flag   | false    | Enable automatic loop unrolling                                   |
| `--adg-path`      | Path   | -        | CGRA architecture description file (.adg); required when unroll is enabled |
| `-o` / `--output` | Path   | stdout   | Output path for optimized kernel MLIR; omit to print to stdout    |

### 2.4 Output Directory Layout

adora-cc creates `adora-cc-ir/` under the work directory. Typical structure:

```
adora-cc-ir/
├── 0_kernels/        # Kernel MLIR
├── 1_kernels_opt/    # Optimized kernel MLIR (for cgra-mapper)
├── 2_dfgs/           # CDFG .dot files
└── tempfiles/DFGs/   # Temporary DFG files
```

### 2.5 Examples

```bash
# Compile from C source
adoracc.py mvt.c -o mvt_opt.mlir

# Compile from existing MLIR, specify work directory
adoracc.py kernel.mlir --work-dir ./build -o kernel_opt.mlir

# Enable auto-unroll (requires ADG)
adoracc.py kernel.c --enable-unroll --adg-path rtl/spec/vitra_cgra_adg.json -o kernel_opt.mlir
```

---

## 3. cgra-mapper Usage

cgra-mapper maps kernels in MLIR onto the CGRA architecture description (ADG) and generates executable configuration or call code. It supports two main output types: **pytest (cocotb)** and **sdk emit**.

### 3.1 General Invocation

```bash
cgra-mapper [options] <input.mlir> \
  --adg path/to/adg.json --op-file path/to/operation.json \
  --output-type [pytest/sdk/c]
```

### 3.2 Parameters

| Parameter         | Type  | Default   | Description                                      |
|-------------------|-------|-----------|--------------------------------------------------|
| `--adg`           | required | -      | Architecture description file path (.json)       |
| `--op-file`       | required | -      | Operation definition file path (operations.json)|
| `--output`        | Path  | -        | Output file path                                |
| `--output-type`   | enum  | pytest   | Output type: `pytest` / `sdk` / `c`             |
| `--obj-opt`       | bool  | true     | Enable objective optimization (better mapping)  |
| `--max-iters`     | int   | 2000     | Maximum mapping algorithm iterations            |
| `--timeout`       | int   | 360000   | Timeout in milliseconds                         |
| `--dump-mapped-viz` | flag | true   | Dump mapping visualization                      |
| `--verbose`       | flag  | false    | Verbose output                                  |
| `--tile`          | int   | 9999999  | Max number of tiles to use (default: all)       |
| `--parallel-cores`| int   | 1        | Number of cores for parallel mapping            |

### 3.3 Output Type 1: pytest (cocotb)

*To be superseded by c-cocotb simulation.*

For **AXI-CGRA simulation**, generates cocotb-based pytest scripts for RTL simulation.

**Features:**
- Outputs `.py` files
- Used with cocotb / pytest
- For AXI-interface CGRA simulation

**Example:**

```bash
cgra-mapper \
  --adg="../../rtl/spec/vitra_cgra_adg.json" \
  --op-file="../../rtl/spec/operations.json" \
  --output="mmul_relu.py" \
  --output-type="pytest" \
  --obj-opt=true \
  --max-iters=2000 \
  mmul_relu_opt.mlir
```

### 3.4 Output Type 2: sdk emit

For **Vitis SDK** and similar SoC flows; generates C-style CGRA call functions for Rocket+CGRA SoC or Vitis integration.

**Features:**
- Outputs C call functions
- For Vitis SDK / SoC integration
- Can be used with RISC-V host code

**Example:**

```bash
cgra-mapper \
  --adg="path/to/cgra_adg.json" \
  --op-file="path/to/operations.json" \
  --output="kernel_sdk.c" \
  --output-type="sdk" \
  --obj-opt=true \
  kernel_opt.mlir
```

### ~~3.5 Output Type 3: C (RISC-V)~~

When the default type is `c`, it generates C call functions for RISC-V executables for Rocket+CGRA SoC simulation or board runs.

---

## 4. Reference: Full Compilation Flow

### 4.1 Scenario: C kernel (e.g. MVT) → adora-cc → cgra-mapper (sdk)

Example with Polybench MVT:

```bash
# 1. Generate optimized kernel MLIR from C with adora-cc
adora-cc mvt.c --work-dir . -o adora-cc-ir/1_kernels_opt/mvt_opt.mlir

# 2. Generate pytest (cocotb) test script with cgra-mapper
cgra-mapper \
  --adg="../../rtl/spec/vitra_cgra_adg.json" \
  --op-file="../../rtl/spec/operations.json" \
  --output="mvt_test.c" \
  --output-type="sdk" \
  --obj-opt=true \
  --max-iters=2000 \
  adora-cc-ir/1_kernels_opt/mvt_opt.mlir
```

---

## 5. File Path Reference

| File                           | Description                              |
|--------------------------------|------------------------------------------|
| `vitra_cgra_adg.json`          | CGRA architecture description (ADG)     |
| `operations.json`              | CGRA operation definitions               |
| `adora-cc-ir/1_kernels_opt/*.mlir` | Optimized kernel MLIR from adora-cc  |
| `*.py`                         | pytest (cocotb) scripts from cgra-mapper |

---

## 6. Notes

1. **ADG and op-file:** `--adg` and `--op-file` must match the target CGRA version.
2. **obj-opt:** `true` gives better mapping quality but is slower; for debugging, set to `false` or reduce `--max-iters`.
3. **pytest vs sdk:** pytest is for cocotb simulation; sdk is for Vitis and other SoC toolchains.
4. **adoracc vs tensor-opt:** adoracc is for C/general MLIR kernels; tensor-opt is for tensor ops (e.g. Gemm, Matmul) dataflow optimization. tensor-opt can be omitted for now; it is still being extended.

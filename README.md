```
      __   __   __           __   __   __           __   __         __          ___  __  
 /\  |  \ /  \ |__)  /\     /  ` / _` |__)  /\     /  ` /  \  |\/| |__) | |    |__  |__) 
/~~\ |__/ \__/ |  \ /~~\    \__, \__> |  \ /~~\    \__, \__/  |  | |    | |___ |___ |  \ 
                                                                                           
```
## ADORA: Adaptive Dataflow Optimization for Reconfigurable Architectures.

An MLIR project for CGRA SoC ([FDRA Repository](https://github.com/MIONkb/FDRA)).
ADORA includes three compilers designed for the CGRA SoC:

- **`tensor-opt`**

·High-level tensor transformation tool for CGRA-based DNN acceleration.  

·Converts commonly used tensor operations to ADORATensor built-in operations (e.g., `linalg.matmul`-> `ADORATensor.Gemm`).  

·Focuses on operator-level dataflow optimizations before hardware mapping.  

- **`cgra-opt`**

·Mid-level automated lowering and transformation framework.  

·Performs affine optimizations for C kernels and tensor operations not directly supported in the ADORATensor dialect.  

·Generates Data-Flow Graphs (DFGs) for subsequent mapping.  

·Can be executed independently of specific hardware information.  

- **`cgra-mapper`**

·Low-level mapper that maps DFGs to the Architecture Description Graph (ADG).  

·Handles hardware-specific scheduling, placement, and resource allocation.  

·Generates RISC-V execution files for the Rocket+CGRA SoC.

·Or generates pytest files for the AXI-CGRA simulation environment.


### Related repositories

External projects that complement this tree:

| Project | Role |
|--------|------|
| [VITRA-CGRA](https://github.com/MIONkb/VITRA-CGRA) | CGRA RTL / hardware generation (Chipyard-based). |
| [CGRA-Cocotb-Sim](https://github.com/theElysia/CGRA-Cocotb-Sim) | Cocotb-based simulation environment for AXI-CGRA. |

#### If you have any issues related to this repository, please don't hesitate to get in touch!

## Directories:
- **`tools`** : Contains the main functions for `tensor-opt`, `cgra-opt` and `cgra-mapper`

- **`include` / `lib`** : C++ headers and source files for `cgra-opt` and `tensor-opt`

- **`mapper`** : C++ source files for `cgra-mapper`

- **`test`** : LLVM **lit** regression suite (`check-adora`): MLIR inputs and FileCheck expectations for `tensor-opt`, `cgra-opt`, `cgra-mapper`, and `adoracc` kernel flows; `test/spec/` holds JSON architecture / operation specs used by mapper tests.

- **`frontend`** : Checkout location for the optional **adora-onnx-mlir** submodule (ONNX → MLIR entry path). Initialize and build it as in [Build § adora-onnx-mlir](#3-adora-onnx-mlir-optional) below.

- **`build_tools`** : Bash scripts for building LLVM and Adora. See [build_tools/C_Compiler_instruction.md](build_tools/C_Compiler_instruction.md) for how to run the C compiler (adoracc + cgra-mapper) end-to-end.

- **`experiment`** : Includes ML benchmarks and C benchmarks (e.g., Polybench). Follow the instructions below to run them. More benchmarks will be added soon.

- **`env.sh`** : Change the environment variables to your own, and source it.
---

# Build

### 1 LLVM-18  (Unskippable)

<!-- ~~LLVM Commit: `26eb4285b56edd8c897642078d91f16ff0fd3472` (Same as [Polygeist GitHub Repository](https://github.com/llvm/Polygeist))~~ -->

LLVM Commit: `b270525f730be6e7196667925f5a9bfa153262e9` (Same as [ONNX-MLIR v5.0.0](https://github.com/onnx/onnx-mlir/tree/v0.5.0.0))

You can download the specified version from the following link:  [LLVM GitHub Repository](https://github.com/llvm/llvm-project/tree/b270525f730be6e7196667925f5a9bfa153262e9)

To install LLVM, follow script:  
```bash
./build_tools/build_llvm.sh
```
### 2 Adora (This project)

Update the LLVM installation path in the following files to your own path:
- `build_adora.sh`
<!-- - `CMakeLists.txt` -->

After making the changes, run `build_adora.sh`.  
**Tip:** Run `build_adora.sh` in small steps (copy one command at a time) so you can catch errors early.

After a successful CMake build, you can run the regression suite from the build directory:

```bash
cmake --build <your-build-dir> --target check-adora
```

The validated loop-bearing conditional-store path is documented in
[`docs/development/cstore-loop-index-e2e.md`](docs/development/cstore-loop-index-e2e.md).

### 3 adora-onnx-mlir (Optional)

A self-hosted **onnx-mlir** fork lives under [`frontend/adora-onnx-mlir`](frontend/adora-onnx-mlir) (git submodule). From the repo root, initialize and update it with:
```bash
git submodule update --init --recursive
```
Then follow the installation script [`build_tools/build_onnxmlir.sh`](build_tools/build_onnxmlir.sh).

If you only need to run small existing models in experiments, or only use C kernels, you can skip adora-onnx-mlir.

### 4 CGRA (Optional)

You can download and install the CGRA from the appropriate repository:  
[CGRA Repository](https://github.com/MIONkb/VITRA-CGRA)

A lightweight Python API (cocotb) will be released soon.

### 5 Other dependencies you may need

#### For AI benchmarks: `torch-mlir`

Download and install `torch-mlir`.  
Our supported version is **`torch-mlir_20250127.357`**. (Corresponding **torchvision** version: `0.1.6.dev0`.)

You can find the official repository here:  
[Torch MLIR GitHub Repository](https://github.com/llvm/torch-mlir)

#### For C benchmarks: Polygeist

Download and install **Polygeist** to run C benchmarks.  
You can find the Polygeist repository here:  
[Polygeist GitHub Repository](https://github.com/llvm/Polygeist)

---

# Run an Example

## Quick Start — three built-in examples

Three ready-to-run MLIR kernels live in [`experiment/example/`](experiment/example/):

| Kernel | Description |
|--------|-------------|
| `mvt/mvt.mlir`   | Matrix-Vector Transpose (PolyBench) |
| `attn/attn.mlir` | Scaled dot-product attention (Transformer) |
| `ffn/ffn.mlir`   | Feed-forward network: FC1 + ReLU + FC2 (Transformer) |

Run any of them with `adoracc` (replace `/path/to/adora-compiler` with the actual root):

```bash
ADORACC=/path/to/adora-compiler/build/bin/adoracc.py

# MVT
$ADORACC experiment/example/mvt/mvt.mlir --work-dir . -o ./mvt_result.mlir

# Attention
$ADORACC experiment/example/attn/attn.mlir --work-dir . -o ./attn_result.mlir

# FFN
$ADORACC experiment/example/ffn/ffn.mlir --work-dir . -o ./ffn_result.mlir
```

After a successful run (using `mvt` as example), the current directory will contain:

```
./
├── mvt_result.mlir                      ← final scheduled MLIR
└── adora-cc-ir/
    ├── 1_frontend/                      ← cgeist output (.c input only; empty for .mlir)
    ├── 2_kernel-opt/
    │   └── mvt_opt.mlir
    ├── 3_task-schedule/
    │   ├── mvt.pre.mlir                 ← input snapshot to scheduler
    │   ├── mvt.final.mlir               ← scheduler output
    │   └── mvt.token_graph.dot          ← task dependency graph
    ├── temp/
    │   ├── normalize/
    │   │   ├── mvt_normalized.mlir
    │   │   ├── kernel_mvt_0_CDFG.dot
    │   │   └── kernel_mvt_1_CDFG.dot
    │   ├── kernel-extract/
    │   │   └── mvt_kernel.mlir
    │   └── dfg/
    │       ├── kernel_mvt_0_CDFG.dot    ← final DFG (copied from temp/normalize)
    │       └── kernel_mvt_1_CDFG.dot
    └── pipeline.log
```

---

## Full C-to-Executable Flow

You can run the full C compilation flow (C → adoracc → cgra-mapper) by following **[build_tools/C_Compiler_instruction.md](build_tools/C_Compiler_instruction.md)**.

The `experiment` directory contains pre-transformed MLIR files ready to use.

### Step 1: Set Up Environment Variables
Before running an example, you need to update the environment paths in `env.sh`:

```bash
######################
# Set your own environment paths in env.sh:
# - CGRVOPT_PROJECT_PATH: Path to your `cgra-opt` project (e.g., xxxx/cgra-opt)
# - CGRA_ADG_PATH: Path to the hardware information generated by the Chisel hardware generator
# - CHIPYARD_DIR: Path to Chipyard for final RISC-V compilation and linking
#
# Note:
# - If you only want to generate the DFG and do not need to simulate the benchmark,
#   the CHIPYARD_DIR path is not required.
######################
source env.sh  # Update paths in env.sh as described above.
```
### Step 2: Run 'deriche' example:  C -> mlir -> DFG -> mapping result -> executable file

To run 'deriche' example from Polybench benchmark, you can go with:
```bash
cd experiment/Cbenchmarks/Polybench/medley/deriche/deriche_mini

### From C source file to MLIR. This step can be skipped since the kernel is already in MLIR format.
bash scripts/0_compileCtoMLIR.sh

### For hardware-independent compilation
bash scripts/1_kernel_opt.sh
bash scripts/2_kernel_dfgs.sh

### For mapping and riscv link (Chipyard must be installed)
bash scripts/3_kernel_map.sh
bash scripts/4_get_all_asms.sh
bash scripts/5_compile_and_link.sh

### The final linked RISC-V executable file can now be simulated using tools such as VCS or Verilator.
```


## Related publications

```bibtex
@inproceedings{lou2025adora,
  title={Adora Compiler: End-to-End Optimization for High-Efficiency Dataflow Acceleration and Task Pipelining on CGRAs},
  author={Lou, Jiahang and Zhu, Qilong and Dai, Yuan and Zhong, Zewei and Yin, Wenbo and Wang, Lingli},
  booktitle={2025 62nd ACM/IEEE Design Automation Conference (DAC)},
  year={2025},
  organization={IEEE}
}
```

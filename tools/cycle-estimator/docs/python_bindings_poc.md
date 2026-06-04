# adora-compiler Python 绑定：调研结论与 PoC 计划

## 0. 背景与目标演进

最初目标是给 cycle-estimator 接上 pybind 拿 CDFG 数据。讨论中目标升级为：
**让 adora-compiler 具备"用 Python 写 pass / 操作 IR"的能力**（对标 Triton /
torch-mlir）。estimator 只是这套基建的第一个消费者。

## 1. 关键技术区分（决定路线）

| 能力 | 底座 | 是否需重编 LLVM |
|---|---|---|
| 只读暴露 CDFG / cost model 数据 | pybind11 手写绑定（现有 CDFGBindings.cpp） | 否 |
| autotuner 批量读结果 | 同上 + cost model 暴露 | 否 |
| **用 Python 写 pass、操作 IR** | **MLIR 官方 CAPI + Python 绑定** | **是，必须** |

核心结论：**现有 pybind11 手写绑定永远长不成"用 Python 写 pass"**，天花板是
只读数据暴露。要操作 IR，必须走 MLIR 官方绑定（`mlir.ir` / `mlir.dialects`
/ `mlir.passmanager`），其前提是 LLVM/MLIR 以 `MLIR_ENABLE_BINDINGS_PYTHON=ON`
编译。

## 2. 已查证的事实

- **主干树**：`adora-compiler-cycleestimator/`，含 `lib/DFG`（CDFG + pybind）、
  标准 TableGen dialect、`tools/`。
- **dialect 是标准 TableGen**：`include/ADORA/Dialect/ADORA/IR/ADORAOps.td`、
  `ADORABase.td`、`ADORAKernelOp.td`；C++ 头 `ADORA/Dialect/ADORA/IR/ADORA.h`，
  cgra-opt.cpp:15 引用。→ 官方 `declare_mlir_dialect_python_bindings` 可套用。
- **CMake 已预埋绑定钩子**：CMakeLists.txt:23,27-30 已有 `MLIRDetectPythonEnv`
  / `if(MLIR_ENABLE_BINDINGS_PYTHON)`。
- **硬门槛（决定性）**：链接的 MLIR（Polygeist 的 llvm-project）是
  `MLIR_ENABLE_BINDINGS_PYTHON:BOOL=0` 编的
  （`adora-compiler/frontend/Polygeist/llvm-project/build/CMakeCache.txt:1576`）。
  → 官方绑定必须重编 LLVM 才能开。
- **现有 pybind 是独立土法绑定**：`lib/DFG/python/CDFGBindings.cpp` 用 pybind11
  直接绑 CDFG 只读结构，不经 MLIR 官方基建，与"Python 写 pass"不在一条路上。

## 3. 修订后的 pybind 路线（替代之前的三阶段）

之前的"先修 CDFGBindings 字段"被降级——因为它与"Python 写 pass"终点不在一条
技术路线。新路线直奔官方绑定：

### 阶段 A（当前）：官方绑定最小 PoC
证明"Python import ADORA dialect → parse kernel → 改 op → 打印回 MLIR"可行。

### 阶段 B（后续）：把关键 pass / cost model 暴露，形成 Python 可编程层

### 现有 pybind11（CDFGBindings.cpp）的处置
保留不投入。estimator 短期仍可用它（或纯 Python+正则）拿数据，但不再为它做
字段修补的大投入——资源转向官方绑定。

## 4. 阶段 A PoC 步骤

1. **重编 LLVM/MLIR 开 Python 绑定**（独立 build 目录，不覆盖现有 build）
   - `-DMLIR_ENABLE_BINDINGS_PYTHON=ON -DPython3_EXECUTABLE=<py>`，保持原
     LLVM_ENABLE_PROJECTS
   - 验证：`python -c "import mlir.ir"` 成功
   - 风险：耗时（数十分钟~数小时）、磁盘（数十 GB）、Polygeist 兼容性
2. **ADORA dialect 入官方绑定**
   - 写 ADORA CAPI 注册（`mlirGetDialectHandle__adora__` 模式）
   - ADORA CMake 加 `declare_mlir_dialect_python_bindings`，吃 ADORAOps.td 生成
   - 验证：`from mlir.dialects import adora` 成功
3. **Python parse + 遍历**：parse 含 `ADORA.kernel` 的 MLIR，walk 到 ADORA op
4. **改一个 op 打印回 MLIR**：改 attribute 或插 op，输出合法 MLIR

### 执行策略
- 先单独验证步骤 1（纯上游 MLIR import），通过再投入步骤 2 的 CAPI
- 重编在独立 build 目录进行，不破坏现有 build；动重编前再次确认

## 5. 环境确认结论（已实测）

- **真正链接的 MLIR**：`/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build`
  （由 `adora-compiler-cycleestimator/build/CMakeCache.txt` 的 `MLIR_DIR` /
  `LLVM_DIR` 指向）。**不是** Polygeist 那份——重编目标是 onnx 这份。
- **LLVM 版本**：18.0.0（`llvmorg-18-init-7134-g26eb4285b56e`）。两份 llvm 同版本，
  绑定均为 `MLIR_ENABLE_BINDINGS_PYTHON:BOOL=0`。
- **绑定后端**：nanobind（`onnx/mlir/python/CMakeLists.txt:
  PYTHON_BINDINGS_LIBRARY nanobind`）。**当前 Python 环境未装 nanobind**，重编前
  需 `pip install nanobind`。
- **工具链**：Python 3.10.12（`/usr/bin/python3.10`，与 onnx build 一致）、
  ninja 1.13、cmake 3.22.1、磁盘剩 263G（够）。
- **onnx build 的 projects**：`LLVM_ENABLE_PROJECTS=mlir;clang`，重编保持不变，
  只加 `-DMLIR_ENABLE_BINDINGS_PYTHON=ON`。
- `python_packages` 当前未生成（绑定关着，符合预期）。

## 5b. PoC 执行结果（已跑通）

策略：原地 reconfigure onnx-llvm build，增量编 `MLIRPythonModules`。

- **步骤 1 成功**：`ninja MLIRPythonModules` 零 error，生成
  `_mlir.cpython-310-x86_64-linux-gnu.so`（`[302/302]`）。
  `PYTHONPATH=<onnx>/build/tools/mlir/python_packages/mlir_core` 下
  `import mlir.ir` 通过，`Module.parse('module {}')` OK。
- **步骤 3 成功**：Python parse 真实 fir kernel，walk 遍历到全部 11 个 op：
  `ADORA.BlockLoad ×2, ADORA.LocalMemAlloc, ADORA.kernel, affine.for,
  affine.load ×2, affine.yield, affine.store, ADORA.terminator, ADORA.BlockStore`。
- **步骤 4 成功**：从 Python 给 module 加 attr、给 `affine.for` 加
  `poc.estimated_trip` attr，`print(module)` 输出含新 attr 的合法 MLIR。
  → **"用 Python 操作 IR"在 adora-compiler 上成立。**

### 关键限制（决定下一步）
`Context.allow_unregistered_dialects=True` **只能 parse generic 格式**
（`"builtin.module"()` 那种）。直接 parse 含 `ADORA.BlockLoad` 自定义 assembly
的 .mlir 会报 `Dialect 'ADORA' not found for custom op`。本次 PoC 是先用
`cgra-opt --mlir-print-op-generic` 把 kernel 转 generic 再喂给 Python 绕过的。

→ 要让 Python **直接**读项目里的真实 ADORA .mlir（自定义语法），必须把 ADORA
dialect 真正注册进绑定，即写 CAPI（`mlirGetDialectHandle__adora__`）+
`declare_mlir_dialect_python_bindings`。这是阶段 A 的收尾工作（步骤 2）。

## 6.（原 5）待补 / 待确认
- 重编选项：是在 onnx 现有 build 目录原地加开关 reconfigure（快，但动现有
  build），还是开独立 build 目录（安全，但要全量重编 LLVM 18 ~ 数小时 + 数十 GB）。
  待与用户确认。

## 6. 布局清理项（与本工程并行）
- 删旧残留嵌套树 `tools/cycle-estimator/adora-compiler/`（挪仓库残留，整套重复
  dialect/build/tools；删前 diff 确认无独有改动）
- 修 `tools/cycle-estimator/cdfg_native.py:30,34` 写死的 `.so`/op-name 相对路径

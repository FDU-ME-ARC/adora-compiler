# Cycle Estimator — 实现状态 / 现状 / 后续规划

> 本文记录 2026-06 这一轮重建：从"含正则栈、不可安装"的状态，重建为
> MLIR-native、含访存/计算交叠建模、可 import、CMake 可安装的工具。

## 1. 这一轮实现了什么

### 1.1 走 dot 路径，彻底去正则
- 算子级 CDFG（节点 opcode、边、operand、loop-carried `iterdist`、访存 size）
  来自 cgra-opt `--adora-kernel-dfg-gen` 产出的 `_CDFG.dot`，Python 端解析
  （`core/dot_parser.py`）。dot 里的 opcode 已是硬件名（如 `FMUL32`），直接查
  `core/latency_table.py`（权威源 `Operations.scala`），**不需要**旧的
  `_OpNameCovertMap` / `operations20241118.json` 映射。
- 循环 trip-count / 元素位宽改用 **MLIR 官方 Python 绑定强类型**读取：walk
  `affine.for`，从 `lowerBoundMap` / `upperBoundMap` 的常量 affine_map 算 trip
  （`extract/loop_info.py` + `adora_mlir`）。**零正则**。
- 删除旧的 `mlir_loop_info.py`（正则解析 MLIR 文本）和 `cdfg_native.py`
  （依赖会段错误的 `_adora_cdfg` pybind）。

### 1.2 整个 kernel 时间模型（不止 CDFG）
```
total = config_overhead + max(block_load + block_store, compute) + (prologue/epilogue, 暂未建模)
compute = outer_trip × (II × inner_trip + pipeline_drain)
II = max(RecMII, ResMII)
```
- **访存/计算交叠（overlap）**：默认 `max(访存, 计算)`，建模异构 token 硬件下
  数据搬运与计算并发；`--no-overlap` 退回串行求和。
- **ADG 接入**（`arch/adg.py`）：从 ADG json 读 `num_alus`(= num_row×num_colum)、
  `cfg_data_width`、`cfg_spad_size`、`max_cfg_data_num` 等，替掉硬编码的
  `num_alus=16`；`config_overhead` 用 tile-based 估算（tile 数 × max_cfg_data_num）。

### 1.3 工程化
- 整理成包：顶层只留 `run.py`，引擎进 `core/`，trip 提取 `extract/`，ADG
  `arch/`，独立工具 `scripts/`。
- CMake install（`tools/cycle-estimator/CMakeLists.txt`）：装 `bin/pypack/
  cycle_estimator/` 包 + `bin/cycle-estimator` 启动器（自动设 PYTHONPATH /
  ADORA_MLIR_CORE），支持 `python -m cycle_estimator` 与直接命令两种调用。
- 测试 19 个全绿（`tests/test_estimator.py`），覆盖 dot 解析、II、关键路径、
  io 字节、overlap 与串行两种总公式、trip 提取。

### 1.4 顺带修的 C++（非主线）
`lib/DFG/DFGgen.cpp` 中 5 处 affine load/store 处理对 block-argument / 非
DataBlock memref 的空指针解引用 / `assert(0)`，已加保护——这是早先"复用
`_adora_cdfg` pybind"路线遗留的修复，当前 dot 路径不依赖它，`lib/DFG/python`
（`_adora_cdfg`）已随该路线放弃而删除。

## 2. 现状（准不准）

**结构正确，绝对值未经校准——目前是乐观下界，不可作绝对周期数引用。**
未校准 / 理想化的来源：

| 项 | 现状 | 影响 |
|---|---|---|
| DMA 带宽 | `DEFAULT_DMA_BYTES_PER_CYCLE = 4`（占位 TBD） | 访存 cycle 直接随之缩放；overlap 下常是大头 |
| overlap | 取 `max`，假设零气泡完美交叠 | 忽略 prologue/epilogue、token 同步、buffer stall → 偏乐观 |
| II | `max(RecMII, ResMII)` 调度下界，`route_lat=0` | 真实 II ≥ 此值，偏乐观 |
| config | `cfg_num // 3` 系数未核实 | config_overhead 绝对值待定 |

可信的部分：相对趋势（kernel 间快慢、改循环边界的变化方向）大概率正确；
绝对 cycle 数待校准。report 里的 `optimistic` 标注属实。

## 3. 后续规划

### 3.1 校准（最优先，便宜且能定位误差）
Ground truth 已有：`CGRA-Cocotb-Sim/server/test_runif.py`（RTL 级，
CLOCKPERIOD=2 ns）。校准链已就绪（`scripts/calibrate.py`）：
1. 跑 cocotb sim 拿含 `EXE.F ... time {T} ns` 的日志（ground-truth cycles = T/2）。
2. `python3 scripts/calibrate.py --log sim.log --dot-dir <dir> --mlir kernel.mlir`
   对比估算 vs 真值、出误差。
3. 据误差拟合：DMA 带宽、route_lat、config 系数。

### 3.2 模型增强（视校准结果）
- prologue/epilogue：overlap 下首尾不可隐藏的搬运，补进公式。
- `num_tiles` 自动推导：当前默认 1，应从 kernel tiling 决策（`getTileSize` /
  `OpStrategyDecision`）或 ADG 推算。
- 非常量循环 bound：`extract/loop_info.py` 目前只处理常量 affine_map，符号/参数化
  bound 待支持。

### 3.3 事件级仿真器（条件触发，非默认要做）
是否自写一个事件级（比 RTL 快、比解析模型准）仿真器，取决于校准结果：
- **若**校准后解析模型误差可接受 → 不必写，解析模型够用。
- **仅当**满足以下之一才值得写：(a) cocotb RTL sim 太慢，无法支撑编译器内
  design-space exploration；(b) 校准发现系统性结构误差（overlap 气泡、token
  同步、buffer stall 等动态行为，调常量无法消除）。
- 注意悖论：自写仿真器的参数同样要靠 cocotb sim 校准，所以"为了更准"不是写它的
  理由；"为了在 RTL 太慢时做快速 DSE"才是。

## 4. 可移植性遗留
- `ADORA_MLIR_CORE` 默认值、`run.py` 中 cgra-opt / 编译器根路径为本机绝对路径，
  换环境需经环境变量 / 参数覆盖；后续可做成 CMake 配置注入。

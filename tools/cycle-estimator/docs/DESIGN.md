# ADORA MLIR 周期数预估器 — 设计文档

## 1. 目标

基于 ADORA 方言的 MLIR，快速预估 kernel 在 CGRA 上的执行周期数，**无需跑完整 mapper 映射、无需跑 RTL 仿真**。

复用编译器已有的 DFG 生成能力（`adora-kernel-dfg-gen` pass）拿到结构化数据流图，外层用 Python 补充硬件延迟知识、II 估计公式和周期模型。

设计原则：**不重造 DFG，不与编译器逻辑漂移，硬件常数以唯一事实源为准**。

---

## 2. 数据来源（已查证）

### 2.1 入口：MLIR 直出结构 dot，不依赖 mapper

cgra-opt pass 名 **`adora-kernel-dfg-gen`**（`include/ADORA/Dialect/ADORA/Transforms/Passes.td:19`）。
对模块内每个 `ADORA.KernelOp` 调用 `generateCDFGfromKernel()` 后**无条件**输出 `<kernelName>_CDFG.dot`：

```cpp
// lib/Dialect/ADORA/Transforms/DFGgenPass.cpp:78-86
m->walk([&](ADORA::KernelOp kernel) {
  std::string kernelName = kernel.getKernelName();
  if (kernelName.empty()) kernelName = "kernel_" + std::to_string(kernel_cnt);
  LLVMCDFG *CDFG = new LLVMCDFG(kernelName, GeneralOpNameFile_str);
  generateCDFGfromKernel(CDFG, kernel, /*verbose=*/Verbose);
  CDFG->CDFGtoDOT(kernelName + "_CDFG.dot");   // 默认就出 dot，无需 verbose
  kernel_cnt++;
});
```

选项：`verbose`（默认 false，仅多出中间态 dot，主 dot 不依赖它）。

调用命令（待用真实二进制名核实路径）：
```
cgra-opt <kernel.mlir> --adora-kernel-dfg-gen   # 产出 <kernel>_CDFG.dot
```

### 2.2 dot 内容：纯结构图（无 latency / 无 II）

`LLVMCDFG::CDFGtoDOT` 实现于 `lib/DFG/src/mlir_cdfg.cpp:175-308`。dot 携带：

- **节点名** = `TypeName + id`（如 `add5`、`load12`，id 需从名字尾部数字解析）
- **节点属性**：`opcode="<op>"` 恒有；load/store 额外带 `ref_name / size / offset / pattern`；常量带 `value`；`color` 编码 loop level
- **边属性**：`operand=<idx>`；回边（loop-carried）带 `iterdist=<N>`；依赖类型仅用**颜色**编码（红=CTRL，蓝=MEM 循环依赖，黑=DATA）
- **不含**：latency、II、stride、loop bounds（stride/bounds 的 emit 行被注释掉，`mlir_cdfg.cpp:197-209`）

### 2.3 延迟事实源：硬件 Scala（唯一权威）

`VITRA-CGRA/cgra-mg/src/main/scala/op/Operations.scala:71` 起 `BasicOpInfoMap`，注释 "latency including the register outside ALU"：

| op 类 | latency |
|---|---|
| 整数 ALU（ADD/SUB/MUL/MIN/MAX/AND/OR/XOR/移位/比较/SEL） | 1 |
| UDIV/SDIV/UREM/SREM | `DIV_LATENCY` = **6**（`Common.scala:37`） |
| MulAdd / MAC | 2 |
| FADD32 / FSUB32 | 1 |
| FMUL32 | 3 |
| FDIV32 | 7 |
| FSQRT | 17 |
| FMA32 / FMAC32 | 4 |
| BFADD16 | 1 |
| BFMUL16 | 3 |
| BFDIV16 / BFMA16 | 4 |
| DEINTLV4 / 3 / 2 | 4 / 3 / 2 |
| 访存 LSOpInfoMap：INPUT / LOAD / CLOAD | 2 |
| 访存 LSOpInfoMap：OUTPUT / STORE / CSTORE | 1 |

> ⚠️ **不要**用 `lib/DFG/Documents/operations20241118.json`，那是过时副本（其中 LOAD=3、FMUL32=2、FDIV32=2，与硬件冲突）。

### 2.4 II 公式：mapper 的 RecMII（Python 侧复刻）

```cpp
// mapper/src/mapper/mapping.cpp:1685  Mapping::evaluateII()
int II = 1;
for (auto& elem : _dfg->backEdgeLoops()) {            // 遍历回边
    int dstInportLat = dfgNodeAttr(dstId).lat - node(dstId)->opLatency();
    int srclat   = dfgNodeAttr(srcId).lat;
    int routeLat = _dfgEdgeAttr[elem.first].lat;       // 布线延迟（dot 中无此信息）
    int iterDist = std::max(edge->iterDist(), 1);
    int newII = (srclat + routeLat - dstInportLat + iterDist - 1) / iterDist;  // 向上取整
    II = std::max(newII, II);
}
return std::max(II, _II);   // _II = 资源下界 ResMII
```

即 RecMII：对每条回边 `II ≥ ⌈(srclat + routeLat − dstInportLat) / iterDist⌉`，取最大，再与 ResMII 取 max。

---

## 3. 周期模型

总周期（雏形）：

```
Cycles ≈ config_overhead(cfgNum)
       + BlockLoad_latency(bytes)
       + II × trip_count
       + pipeline_drain(关键路径延迟 = DFG 最长路径)
       + BlockStore_latency(bytes)
```

各项来源：
- `config_overhead`：仿真器中 `enable_config` 延迟 = `config_ptr.size // 3`（`CGRA-Cocotb-Sim/server/test_runif.py:1209-1211`）。cfgNum 由 mapper `EmitPytest.cpp:2196` 按 `cdp.data.size()*32/cfgDataWidth` 累加。
- `II`：见 §2.4。
- `trip_count`：从 MLIR 的 `affine.for` 上下界静态读出（KernelOp body 恒为单顶层 affine.for）。
- `pipeline_drain`：DFG 关键路径上各节点 latency 之和（对应 mapper `latencyBound()` / `mapping.cpp:1214` 的 maxLatDfg）。
- `BlockLoad/Store_latency`：按 memref 字节数与 DMA 带宽换算（待补带宽常数）。

---

## 4. 实现结构

```
cycle-estimator/
├── docs/
│   └── DESIGN.md            # 本文档
├── run.py                   # 入口：调 cgra-opt 出 dot → 解析 → 估算 → 打印/写报告
├── dot_parser.py            # 解析 _CDFG.dot → 内存图（节点 op/id/loop-level, 边 operand/iterdist/类型）
├── latency_table.py         # 硬编 Operations.scala 延迟表 + DIV_LATENCY=6（唯一事实源）
├── ii_model.py              # 复刻 evaluateII 的 RecMII + 估 ResMII=⌈节点数/PE阵列⌉
├── cycle_model.py           # §3 总周期公式
├── mlir_loop_info.py        # 从 MLIR 读 affine.for trip_count / load-store 字节
└── tests/
    └── test_intvecadd.py    # 用已知 kernel 端到端校准
```

### 模块职责

- **dot_parser.py**：纯文本解析 Graphviz dot。提取节点 `opcode`、id、`color`（loop level）、load/store 的 `size`；边的 `operand`、`iterdist`、颜色（依赖类型）。构建有向图 + 回边集合。
- **latency_table.py**：`op_latency(op_name) -> int`，照搬 §2.3 表。op 名归一化（去掉尾部数字 id）。
- **ii_model.py**：
  - `rec_mii(graph)`：在回边上套 §2.4 公式。**routeLat 在 dot 中缺失，先取 0**（得乐观下界）。
  - `res_mii(graph, pe_array)`：≈ ⌈计算节点数 / ALU 数⌉ 与 ⌈访存节点数 / IO 单元数⌉ 取 max（对齐 Passes.td:242 的 Util 公式）。
  - `estimate_ii = max(rec_mii, res_mii)`。
- **cycle_model.py**：组合 §3 公式。
- **mlir_loop_info.py**：用正则或轻量 MLIR 解析从 kernel.mlir 拿 affine.for 边界 → trip_count。

---

## 5. 实现步骤

1. **dot_parser.py** — 先解析 `_CDFG.dot`，打印节点/边统计，肉眼对一个 IntVecAdd dot 验证解析正确。
2. **latency_table.py** — 照搬硬件延迟表。
3. **mlir_loop_info.py** — 从 MLIR 读 trip_count 与 load/store 字节。
4. **ii_model.py** — RecMII（routeLat=0）+ ResMII。
5. **cycle_model.py** — 总周期公式。
6. **run.py** — 串联：调 cgra-opt 出 dot → 解析 → 估算 → 报告。
7. **端到端校准** — 用 IntVecAdd / gemm，与仿真器真实周期对比，记录误差，必要时引入布线延迟修正系数。

---

## 6. 已知精度上限与风险

| 风险 | 说明 | 缓解 |
|---|---|---|
| dot 无布线延迟 routeLat | RecMII 公式依赖它，缺失则 II 偏乐观（低估） | 先取 routeLat=0 得下界；后续用仿真器校准修正系数 |
| ResMII 需要 PE 阵列规模 | dot 不含硬件配置 | 从 cgra-mg 配置或 ADG 读阵列尺寸（ALU 数 / IO 单元数） |
| DMA 带宽常数未定 | BlockLoad/Store 周期换算需要 | 待从 test_runif.py / 硬件读出 |
| mapper 运行时实际加载哪份 latency JSON 未核实 | 影响"与真实映射一致性"的判断（本方案不读该 JSON，以 Scala 为准，风险低） | 实现后用一个已映射 kernel 反查确认 |

---

## 7. 待确认（实现阶段）

1. cgra-opt 二进制的实际路径与调用方式（`adora-kernel-dfg-gen` flag 已确认）。
2. II 精度策略：先用"routeLat=0 的 RecMII 乐观下界"，还是一开始就预留校准系数接口。
3. PE 阵列规模（ALU 数 / IO 单元数）从哪里读（cgra-mg 配置 or ADG json）。

---

## 8. 实现状态（v1 已落地）

已实现并端到端跑通的模块：
- `dot_parser.py` — 解析 `_CDFG.dot`（节点 opcode/id/size，边 operand/iterdist/deptype）。已用 gemm 真实 dot 验证。
- `latency_table.py` — 照搬 `Operations.scala` 全表（含 ACC/MAC/FP32/BF16，`DIV_LATENCY=6`）。
- `mlir_loop_info.py` — 从 kernel MLIR 提取嵌套 affine.for trip-count 与 memref 元素字节。
- `ii_model.py` — RecMII（routeLat=0 乐观下界）+ ResMII（⌈compute/num_alus⌉）。
- `cycle_model.py` — 关键路径排空 + `outer × (load + II×inner + drain + store)`。
- `run.py` — 模式 2（已有 dot + MLIR）已验证；模式 1（MLIR→dot）见下方限制。

验证结果（gemm 全部 15 个 dot）：II=1，周期随 trip 单调增长，同名 kernel 各优化阶段估计一致。例：`gemm_1` (25×30) → 975 cyc；`gemm_opt_1` (20×25×30) → 19500 cyc。

### 模式 1（MLIR → dot）已打通

完整 pass 流水（run.py 自动执行）：
```
1. 剥掉模块属性头  module attributes {dlti.dl_spec ...} { → module {
   （这个 cgra-opt 构建无法解析 Polygeist 的 dlti.dl_spec，是模块头而非 ADORA dialect 问题）
2. cgra-opt <stripped.mlir> \
     --adora-extract-affine-for-to-kernel \   # affine.for → ADORA.KernelOp
     --adora-kernel-dfg-gen                    # KernelOp → <kernel>_CDFG.dot
   （须从编译器根目录运行：OpName 文件 lib/DFG/Documents/GeneralOpName.txt 是相对路径）
3. dot 收集到临时目录后解析、估算
```
注意：pass 在写完 dot 后于 teardown 阶段 abort（core dump），但 dot 已落盘，run.py 据"是否产出 dot"判断成功，忽略非零退出码。

验证：`run.py --mlir 1_kernels_opt/gemm_kernel_opt.mlir` → gemm_0=81、gemm_1=1855 周期。

### 自动数据抽取（已实现）

`cycle_model.io_bytes_from_dot()` 从 dot 的 `Input`(load)/`Output`(store) 节点 `size` 属性自动累加搬运字节，无需手传 `--load-bytes/--store-bytes`（仍可覆盖）。

### 校准框架（已就绪，待仿真环境）

`calibrate.py` 解析仿真日志的 `EXE.F ... time {T} ns` 行（test_runif.py:1430），按 `CLOCKPERIOD=2 ns`（:1469）换算 **ground-truth cycles = T / 2**，与预估器对比算误差。日志解析已用模拟数据验证。

⚠️ 当前环境**未装 cocotb / verilator / iverilog**，无法实跑 RTL 仿真。校准需在有仿真环境的机器上：开 DEBUG 日志跑 cocotb，把日志喂 `calibrate.py --log sim.log --dot-dir <dir> --mlir <kernel.mlir>`。

### 剩余已知限制

| 限制 | 说明 |
|---|---|
| II 偏乐观 | routeLat=0，未含布线延迟，对长回环 II 会低估。`--route-lat` 接口待仿真标定。 |
| ResMII 阵列规模 | 默认 `num_alus=16`，需按真实 ADG 调整（`--num-alus`）。 |
| DMA 带宽 | `DEFAULT_DMA_BYTES_PER_CYCLE=4` 为占位，需用 test_runif.py 标定。|
| cfgNum 未自动抽取 | dot 不含 config 字数；仍需 `--cfg-num` 传入或后续从 EmitPytest 产物读。 |
| trip-count 依赖 ADORA.kernel | 完全原始的 affine MLIR（未包 kernel）trip-count 回退到默认 1。规范化后 MLIR 正常。 |

### 下一步（v2 校准）

1. 在有仿真环境的机器上用 CGRA-Cocotb-Sim 对 gemm/IntVecAdd 做回归，标定 routeLat 与 DMA 带宽。
2. 自动从 EmitPytest 产物抽取 cfgNum。

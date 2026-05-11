# ADORA Compiler — Paper vs Code 差距盘点

> 生成时间: 2026-05-07
> 代码基线: `/data00/home/loujiahang/adora/adora-compiler` (upstream clean 版)
> 论文基线: `/data00/home/loujiahang/adora/adora-paper/chapters/*.tex`
> Build LLVM: `$HOME/CGRVOPT/llvm-project-onnx/build` (与 aicb-agent 共用)
>
> 本文档只做"承诺 vs 实现"盘点与证据落位，不提修复方案；修复方案在另外的 plan/design 文档里展开。

---

## 0. Install 状态

- `build_tools/build_adora.sh:7-8` 已把硬编码 `/home/share/onnx-mlir/...` 改为 `LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-$HOME/CGRVOPT/llvm-project-onnx/build}"` + 同款 fallback `LLVM_INSTALL_DIR`。
- `build_tools/build_adora.sh:3-4` 同时修了 upstream 的 `PROJECT_ROOT` 计算 bug（原脚本把 `dirname` 结果当 root，实际需要 `../`）。
- 一次成功 build 后产物：`build/bin/{cgra-opt, cgra-mapper, tensor-opt, adoracc.py}`。
- `ninja check-adora` 结果：**通过 8 / 预期失败 3 / 失败 7 / 共 18**。
- 失败集中在 CDFGGen 一类（见 §1 缺口 8），是 upstream 缺陷，非本次改动引入。

---

## 1. 论文承诺 vs 代码缺口 — 逐项

### 缺口 1 — `analyzeDependencyInGraph()` 是空 stub

**承诺来源**：paper §4.2 开头，"maximize concurrent execution opportunity" 所需的依赖图分析。

**证据**：`lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:160-166`

```cpp
void analyzeDependencyInGraph(TaskGraph* graph){
  // Pingpong优化下的依赖分析，当任务a依赖任务b，且任务b是否依赖任务a
  // 当a_in依赖b_out时，反向依赖 b_in 依赖 a_out
  // firstly,
}
```

函数体只有一行 `// firstly,` 注释，无实现。这是后续所有 reorder / fusion 依赖分析的前置，直接堵死下游三条路径。

**影响面**：缺口 2（三原则）、缺口 5（Task Fusion）、缺口 6（Pingpong 覆盖决策）全部因此走不通。

**严重度**：🔴 高。

---

### 缺口 2 — "Schedule Nodes on Graph" 三原则全部是空壳

**承诺来源**：paper §4.2.1，"reorder nodes by three principles"。

**证据**：`adora-paper/chapters/4-mainbody.tex:386-392`

```tex
1. If one load node
2. If one load node is
3. If one kernel node is
```

**论文本身也只写到这里**（三句标题后无正文）。代码侧 `grep -r 'reorderBy\|scheduleNodesOnGraph' adora-compiler/` 无命中。

**严重度**：🔴 高（paper 与 code 同时缺失）。

---

### 缺口 3 — `RemoveRedundantBlockLoads` 整体被注释

**承诺来源**：paper §4.2 的 load 合并优化（与 store/load 对去冗余互补）。

**证据**：`lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:218-256`

整块 `/* ... */` 注释，函数体约 40 行全部失效。仅配对的 `RemoveRedundantBlockStoreLoadPair`（`ScheduleAdoraTasks.cpp:177-216`）是 live 的。

**严重度**：🔴 高。

---

### 缺口 4 — `TaskScheduleAlgorithm` 只有候选枚举的壳，缺评估与 Pareto

**承诺来源**：paper §4.2 / §4.4，算法骨架：

1. 三种 pingpong 策略（WeightStationary / InputStationary / OutputStationary）
2. SPM 合法布局枚举
3. 基于 bank conflict 的 unroll / unroll-and-jam 决策
4. 对每个组合 generate+map DFG，计算 II / RU / TCV
5. 按 Pareto 取 top-5 回代到 MLIR

**证据**：

- `mapper/src/tensorop/mapGemm.cpp`、`mapper/src/tensorop/mapConv.cpp` 各自内联枚举 4 种 stationary × `double_buffer=true` × `prefetch_depth=1`（在 aicb-agent 分支里有同样代码；adora 分支完全无 `pipeline_scheduler.h`，连壳都没有）。
- 无 SPM 合法性剪枝、无 bank conflict 分析、无 unroll-and-jam 决策、无 Pareto top-5。
- `mapper/src/mapper/mapper.cpp` 是单点映射，不做多候选。

**严重度**：🟠 中（框架零散，需要把 4 要素聚合起来）。

---

### 缺口 5 — Task Fusion（§4.3）未实现

**承诺来源**：paper §4.3 开头。

**证据**：

- `adora-paper/chapters/4-mainbody.tex` §4.3 只有一句"This will cause a more complicated mapping ..."，无算法描述。
- `grep -ri fusion adora-compiler/lib adora-compiler/mapper` 无命中。

**严重度**：🟠 中（paper+code 同时缺）。

---

### 缺口 6 — "Covered Data Transfer (Pingpong)" §4.4 正文为空

**承诺来源**：paper §4.4 子节 "Covered Data Transfer(Pingpong)"。

**证据**：

- `adora-paper/chapters/4-mainbody.tex:446` 为文件末行，子节标题后无内容。
- 代码侧 `mapper/src/mapper/configuration/pingpongCfg.cpp`（532 行）只做 pingpong 配置编码（strideName / 循环边界翻译），**没有**"何时 cover / 如何选层数"的决策逻辑。

**严重度**：🟠 中（代码有低层配置，决策层空白）。

---

### 缺口 7 — Chapter 5（backend）章节级正文缺失

**承诺来源**：paper §5 backend。

**证据**：`adora-paper/chapters/5-backend.tex:1-22`

全章 22 行，仅泛化描述"backend is binded with specific hardware more tightly ... unified IR is also flexible"。没有：
- mapping 算法层级结构
- II / RU 评估说明
- Emit 管线（EmitCGRACall / EmitPytest / EmitVitisSDK）的章节交代

**代码侧反而已实现**：

| 文件 | 功能 |
|---|---|
| `mapper/src/emit/EmitCGRACall.cpp` | CGRA 调用码生成 |
| `mapper/src/emit/EmitPytest.cpp` | 测试骨架生成 |
| `mapper/src/emit/EmitVitisSDK.cpp` | Vitis SDK host 代码 |

**严重度**：⚠️ paper-only（代码在、paper 缺表述）。

---

### 缺口 8 — `check-adora` 测试套存在 7 个 CDFG 侧失败（upstream 预存在）

**证据**：`ninja check-adora` 输出

```
Failed Tests (7):
  ADORA :: cgra-mapper/FPVecAdd/FPVecAdd.mlir
  ADORA :: cgra-opt/cdfggen/gemm/gemm.mlir
  ADORA :: cgra-opt/cdfggen/getTanh/getTanh.mlir
  ADORA :: cgra-opt/cdfggen/interleave/mergeadd_opt.mlir
  ADORA :: cgra-opt/cdfggen/memref_load/memrefload.mlir
  ADORA :: cgra-opt/cdfggen/mmul_relu/mmul_relu_opt.mlir
  ADORA :: tensor-opt/gemmop_0/gemm_0_cdfg.mlir
```

崩溃点统一：

```
tensor-opt:
  lib/DFG/DFGgen.cpp:2396:
  void FixLinearAccessOfVectorNode(LLVMCDFG*, bool):
    Assertion `node->isLSaffine() && node->getTypeName() == "Input"' failed.

调用链:
  FixLinearAccessOfVectorNode
  → generateCDFGfromKernelAfterOptimization  (DFGgen.cpp:3147)
  → mlir::ADORA::generateCDFGfromKernel      (DFGgen.cpp:3290)
  → ADORATensorOpCdfgGenPass::runOnOperation (TensorCDFGGenPass.cpp:117/124)
```

断言暴露 `FixLinearAccessOfVectorNode` 的前置约束（节点必须是 `LSaffine` 且 typeName 为 `"Input"`）被违反；意味着 CDFG 生成阶段对"非仿射 load / 非 Input 节点的向量化访存修复"没有 fallback 分支。

**影响面**：任何走 `--adora-gen-tensor-op-cdfg` 管线的 GEMM / ReLU / Tanh / memref_load / interleave / FPVecAdd 用例都会崩。

**严重度**：🔴 高（回归测试大面积红）。

---

### 缺口 9 — aicb-agent 分支上的 runtime-online 钩子完全不存在

**背景**：aicb-agent fork 在 `mapper/src/mapper/pipeline_scheduler.{h,cpp}` + `online_ranker.{h,cpp}` + `agent_trace.{h,cpp}` 加了三件套，用于把 pipeline 候选导出给 Python ranker + 记录 trace。

**证据**：

```
adora/adora-compiler/:
  grep -r pipeline_scheduler mapper/   → 无命中
  grep -r online_ranker      mapper/   → 无命中
  grep -r agent_trace        mapper/   → 无命中
```

**结论**：adora 分支是上游 clean 版；若要把 aicb-agent 的在线调度实验回移到这里，需要整块端移 6 个新文件 + mapGemm/mapConv 的 schedulePipeline 挂点（当前都没有）。

**严重度**：ℹ️ 设计取舍（不是 bug，是分支设计差异）。

---

## 2. 汇总表

| # | paper 宣称 / 代码承诺 | 实现状态 | 关键证据 | 等级 |
|---|---|---|---|---|
| 1 | `analyzeDependencyInGraph` | 空 stub | `ScheduleAdoraTasks.cpp:160-166` | 🔴 |
| 2 | Schedule Nodes on Graph 三原则 | paper+code 同缺 | `4-mainbody.tex:386-392` | 🔴 |
| 3 | `RemoveRedundantBlockLoads` | 整段注释 | `ScheduleAdoraTasks.cpp:218-256` | 🔴 |
| 4 | TaskScheduleAlgorithm（候选×评估×Pareto） | 只剩枚举壳 | `mapGemm.cpp` / `mapConv.cpp` / `mapper.cpp` | 🟠 |
| 5 | Task Fusion §4.3 | 空 | paper+code 双缺 | 🟠 |
| 6 | Covered Data Transfer Pingpong §4.4 | paper 空 / code 只有配置编码 | `pingpongCfg.cpp:1-532` | 🟠 |
| 7 | Chapter 5 backend 正文 | paper 空 / code 已实现 | `mapper/src/emit/Emit*.cpp` | ⚠️ |
| 8 | CDFG 生成稳健性 | assertion 崩 (7 例) | `lib/DFG/DFGgen.cpp:2396` | 🔴 |
| 9 | runtime-online pipeline hooks | adora 分支无 | aicb-agent 独有 | ℹ️ |
| — | unified IR / pass 可选 | ✅ 已实现 | `lib/Dialect/ADORA/Transforms/*.cpp` | — |
| — | RemoveRedundantBlockStoreLoadPair | ✅ 已实现 | `ScheduleAdoraTasks.cpp:177-216` | — |
| — | Emit pipeline (CGRACall/Pytest/Vitis) | ✅ 已实现 | `mapper/src/emit/Emit*.cpp` | — |

等级图例：🔴 高 / 🟠 中 / ⚠️ paper-only / ℹ️ 设计取舍。

---

## 3. 建议优先级（供后续 plan 用，不在本文档执行）

1. **先堵 🔴 高危**：缺口 8（CDFGGen assertion，阻断测试 / 下游）→ 缺口 1（`analyzeDependencyInGraph` 空壳）→ 缺口 3（`RemoveRedundantBlockLoads` 注释）。
2. **再补 🟠 算法层**：缺口 4 的 Pareto 评估 → 缺口 6 的 pingpong 决策 → 缺口 5 的 task fusion。
3. **最后补 ⚠️ 文档**：paper §4.2.1 / §4.3 / §4.4 / §5 的正文返工。
4. **独立议题**：缺口 9 的 runtime-online 三件套移植，走单独 branch，不与上面混。

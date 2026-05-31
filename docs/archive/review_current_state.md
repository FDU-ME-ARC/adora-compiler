# ADORA Task-Schedule 全面现状综述

> 生成时间：2026-05-17  
> 分支：`jhlou/scheduletasks`  
> 最新 commit：`c54411b`  
> 本文是对 docs/ 目录下所有文档的综合整理，中文，面向你快速恢复上下文。

---

## 一、这个 Pass 在做什么（背景）

`--adora-schedule-tasks` 是 ADORA 编译器的核心调度 pass，位于 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`。

它的职责是：把 ADORA MLIR 中的 `BlockLoad`、`BlockStore`、`KernelOp` 等 op 组成一张**任务图（TaskGraph）**，分析它们之间的读写依赖（RAW / WAR / WAW / RAR），然后向这些 op 插入 `!ADORA.token` 异步依赖凭证，最终让 emit 层（`EmitPytest` / `EmitCGRACall`）能生成并发的 Python asyncio 代码，让 CGRA 上的多个 DMA 和计算核真正并发执行而非串行等待。

整条编译流水线是：

```
源代码 (.C)
  → adoracc.py（前端）
  → cgra-opt（多 pass 变换，含 schedule-tasks）
  → cgra-mapper（后端 mapping）
  → EmitPytest / EmitCGRACall（代码生成）
  → host Python + CGRA 固件
```

---

## 二、已完成的工作（按 PR 顺序）

### PR1 ~ PR3：异步 Token 基础设施

这三个阶段建立了整个异步 token 系统的底层类型与 pass，是后续所有工作的前提。

**PR1 — `!ADORA.token` 类型 + KernelOp 异步形式**
- 在 ADORA dialect 中新增 `!ADORA.token` 类型（`include/ADORA/Dialect/ADORA/IR/ADORAOps.td`）
- `KernelOp` 支持 `asyncDependencies` 列表和 `asyncToken` 返回值
- 添加 `adora.scheduled` marker attr，标记已被 schedule-tasks 处理过的 func

**PR2 — DMA async form + token threading**
- `BlockLoad`、`BlockStore` 全部支持 async form（`async [%tok1, %tok2, ...]`）
- `threadTokensOnDMAs`：实现 BlockLoad→Kernel 和 Kernel→BlockStore 的 token chain 自动插入
- `DedupAsyncDeps` canonicalization：去掉 asyncDependencies 中的重复 token

**PR3 — Event ops + 降级 passes**
- 新增 `ADORA.event.create`、`ADORA.event.destroy`、`ADORA.event.signal`、`ADORA.event.wait` 四个 op
- `adora-lower-async-tokens` pass：把 `!ADORA.token` SSA 链路降级为 event op 序列
- `adora-to-llvm-async-runtime` pass：把 event ops 降到 runtime stub 调用

---

### PR4：完整 token chain + 多核修复 + 流水线分配

这是最大的一个阶段，修复了多 kernel 场景下的核心 bug，并建立了流水线级别的并发语义。

**PR4-A — `adora-assign-streams` pass**（`lib/Dialect/ADORA/Transforms/AssignStreams.cpp`）

pass 的算法是：
1. 收集 FuncOp 内所有 async-capable ops（BlockLoad / BlockStore / KernelOp）
2. Kahn 拓扑排序（按 asyncDependencies SSA 边）
3. 贪心分配 stream ID：无前驱分配新 stream（上限 `max-streams`，默认 4），有前驱继承最小前驱 stream ID
4. 把 `stream : i32` 属性写回每个 op

新增 lit 测试：`assign_streams_linear.mlir`、`assign_streams_parallel.mlir`、`assign_streams_fanin.mlir`，全部通过。

**PR4-B — `lower-async-tokens` 读 stream 属性**

`LowerAsyncTokens.cpp` 新增 `getOpStream()` helper，让 `emitSignal` / `emitWait` 能读取上一步写入的 `stream` attr，signal 用 producer stream，wait 用 consumer stream。

**PR4-C — 修复跨 kernel RAW token 未生成的 bug**

这是最关键的 bugfix。原来对于 `kernel_0 → BlockStore → BlockLoad → kernel_1` 这种多核 RAW 依赖，`emit-token=true` **不生成 token**，导致 mapper emit 层看不到依赖关系。

根因有两处叠加：
1. `analyzeDependencyInGraph`：注释误写"RAW already wired via SSA"，但 `_depEdges` 实际从未收到 BlockStore→BlockLoad 的 RAW 边
2. `RemoveRedundantBlockStoreLoadPair`：尝试 erase 有 live use 的 BlockLoad op，产生 broken IR

修复方法：
- Fix 1：在 `analyzeDependencyInGraph` 补加 `BlockStoreNode → BlockLoadNode` RAW dep edge
- Fix 2：`RemoveRedundantBlockStoreLoadPair` 只保留图拓扑 wiring（`addConnectionBetweenTwoNode`），不再 erase MLIR op，让 `threadTokensOnDMAs` 自行处理

新增 `test/cgra-opt/kernel/schedule_3mm.mlir`，通过。

**PR4-D — EmitPytest 并发 asyncio（dep_flag 从 token 计算）**

`GenerateCGRACFGAndEXE` 的 `execute()` 命令 dep_flag 现在从 KernelOp 的 async token 计算：
- 无 async dep（root kernel）→ `dep_flag = 0`
- 有 async dep →  `dep_flag = EX_DEP_ST_LAST_TASK`

**PR4-E — Buffer Reuse（Store→Load 同块消除）**

`RemoveRedundantBlockStoreLoadPair`：当 `Store(C[ti,tj])` 后紧跟 `Load(C[ti,tj])` 访问同一 data block，Load op 被消除，下游直接复用 on-chip `LocalMemAlloc` buffer。

3mm 效果：`kernel_3mm_2` 原来需要从 DRAM 读 2 次，现在节省 2 次 DMA。

**PR4-F — dot 可视化工具**

```bash
cgra-opt your.mlir --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/tok.dot"
dot -Tpng /tmp/tok.dot -o tok.png
```

---

### PR5（含于 PR4 后期）：`adora.dep_summary` 序列化

- 在 `ScheduleAdoraTasks.cpp` 中把 DataBlock 级依赖关系序列化为 `adora.dep_summary` ArrayAttr，挂在 `funcOp` 上
- 提供 `parseDepSummary(Operation*)` 读回 `DepSummaryRecord` POD（位于 `lib/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.{h,cpp}`）
- 新增 lit test `test/cgra-opt/schedule/schedule_tasks_dep_summary.mlir` 通过

---

### PR6：Loop-Carried 依赖（最新，分四步）

**背景**：tiled GEMM 的 `tk` 循环中，第 k 次迭代的 BlockStore(C) 和第 k+1 次迭代的 BlockLoad(C) 存在跨迭代 RAW 依赖（loop-carried dep）。原始的 intra-iteration token chain 无法表达这个依赖。

**PR6.1 — Loop-carried dep 检测与诊断**（已完成）

- `findLoopCarriedStoreLoadPair`：识别 `affine.for` body 内 Store→Load 的 loop-carried RAW dep
- `wireLoopCarriedToken`：打印诊断信息（是 stub，不做变换）
- 新增 `adora.lc_dep_summary` attr 记录 loop-carried deps

**PR6.2 — affine.for iter_args token yield**（已完成）

验证并实现了 `affine.for` 支持 `!ADORA.token` 类型的 `iter_args` yield。在 `scf.for` 转换路径下：

```mlir
%init_tok = ADORA.event.create → !ADORA.token
%final_tok = scf.for %tk = 0 to 4
    iter_args(%carry = %init_tok) → (!ADORA.token) {
  %c, %war_tok = ADORA.BlockLoad async [%carry] %C ... → !ADORA.token
  ...
  %store_tok = ADORA.BlockStore async [...] %local, %C → !ADORA.token
  scf.yield %store_tok : !ADORA.token
}
ADORA.event.destroy %final_tok
```

`emit-token` 默认值从 `false` 改为 `true`（PR6.2 fix）。

**PR6.3 — `stripTokenIterArgsFromAffineFor` Pass**（已完成）

PR6.2 产生的 `scf.for iter_args(!ADORA.token)` 在 emit 层会被误识别为计算结果。PR6.3 的 Pass 5 负责在 emit 前把它们 strip 掉，还原为普通 `affine.for`，同时把依赖信息落到 `adora.dep_summary` attr 上供 emit 层消费。

- `rebuildStoreSync` 修复：加入 `dropAllUses` 防止 strip 时产生 broken IR
- 新增 strip 回归测试，全部通过
- 5 个 experiment 的 IR 终态验证：token iter_args 全部 strip，非 token reduction 正确保留

**PR6.4 — cgra-mapper `--enable-async` 接线**（已完成，但端到端未验证）

- cgra-mapper 新增 `--enable-async` CLI flag（默认 `false`）
- 开启时在 mapping 前自动插入：`schedule-tasks → assign-streams → lower-async-tokens` 三 pass 流水
- cmake 加入 `MLIRADORATransforms` link

默认关闭时字节级等同 PR6 前的行为。

---

### 当前完整 token chain 形态（正常路径）

```
BlockLoad_A ──tok0──┐
BlockLoad_B ──tok1──┤→ ADORA.kernel async[tok0,tok1,...] ──tokK──→ ADORA.BlockStore async[tokK]
BlockLoad_C ──tok2──┘
```

- `BlockLoad → Kernel`：load 完成后 kernel 才开始计算 ✅
- `Kernel → BlockStore`：kernel 完成后 store 才写出 ✅
- `BlockLoad WAR/RAW → BlockStore`：同 memref tile 的 intra-iteration fence ✅
- `loop-carried RAW`：跨迭代依赖检测 ✅，emit 语义待实现（见未解决 §4）

---

## 三、尚未完成的工作（按优先级排序）

### 阻塞级（必须先解决，才能 merge PR6）

**【未解决 1】PR6.4 端到端冒烟未通过**

现象：`cgra-mapper --enable-async=true` 在 `gemm_funccall` 样例上崩溃。  
根因：cgra-mapper 对这个 bf16 gemm_funccall 有 pre-existing bug，与 PR6.4 无关——`--enable-async=false` 下同样崩溃。

当前状态：PR6.4 的代码逻辑本身正确，但无法用现有样例端到端验证。

---

### 较高优先级（影响正确性）

**【未解决 2】`_idToOp` 多 func 场景缓存污染**

EmitPytest 中 `_idToOp`、`_depSummary`、`_lcDepSummary` 是进程级缓存，跨多个 `func.func` 时不清除会导致 task id 错乱。

修复位置：EmitPytest 的 `emitFunc` 入口处加三行 `clear()`。
改动规模：5 行以内，低风险。

**【未解决 3】`EmitCGRACall` / `EmitVitisSDK` 未对称移植**

PR6.4 的双路径逻辑（`_idToOp` / `_depSummary` 缓存 + `getDepsTaskNames` SSA→dep_summary 双路径 + WAR 过滤）只在 `EmitPytest.cpp` 实现，另外两个 emit 文件 `EmitCGRACall.cpp` 和 `EmitVitisSDK.cpp` 各有约 ~200 行对称代码待迁移。

---

### 中等优先级（影响完整性）

**【未解决 4】Loop-carried 依赖在 emit 层未翻译**

`_lcDepSummary` 在 EmitPytest 已经缓存，但**没有消费**。当前跨迭代依赖在生成的 Python 中是缺失的（串行保守行为）。

设计问题：LC dep 在 Python 里应该用什么表达？候选方案：
- `asyncio.Semaphore`（简单，但只能控制并发数量）
- `collections.deque` 作为 N-stage pipe（更精确，但复杂）

**【未解决 5】`RemoveRedundantBlockLoads`（Load-after-Load 消除）未实现**

代码中整块是注释（`ScheduleAdoraTasks.cpp:218-256`），约 40 行全部失效。论文 §4.2 承诺了 Load 合并优化，但 Store→Load 对只完成了一半（Store-Load pair 消除已做，Load-Load 未做）。

**【未解决 6】PR6.3 遗留的孤儿 `ADORA.event.create` sentinel**

PR6.2 在循环前创建的 token sentinel，在 PR6.3 strip 后没有 user，等同 no-op。  
影响：生成的 emit 代码中可能有冗余的 create/destroy 调用对，但不影响正确性。  
修复方案：在 PR6.3 Pass 5 后加一次 dead-event sweep（注意 event ops 有 side-effect，标准 DCE 不处理，需要手写）。

**【未解决 7】5 个 pre-existing lit failures**

以下测试在本 session 开始之前就已失败，与 PR6 无关：

```
cgra-opt/cdfggen/gemm/gemm.mlir
cgra-opt/cdfggen/getTanh/getTanh.mlir
cgra-opt/cdfggen/interleave/mergeadd_opt.mlir
cgra-opt/kernel/gemm.mlir
cgra-opt/schedule/schedule_gemm_tiled.mlir
```

建议：单独开 PR 修，或者加 `XFAIL` 标记（至少先让 `ninja check-adora` 干净）。

---

### 远期待做（P4.1 / P2 路线）

**【待做 P4.1】mapper 侧消费 `adora.dep_summary`**

目前 dep_summary 只在 compiler pass 侧产生，mapper 侧没有任何消费。`psg_next_phase.md` 计划了一个 "log-only" 消费阶段作为桥梁，但这个方案有几个设计疑点（见§四 Q1~Q6），需要你拍板后才能动工。

**【待做 P2】`analyzeDependencyInGraph` 是空 stub**

函数体只有一行注释 `// firstly,`，没有实现。这是后续所有 reorder / fusion / pingpong 决策的前置依赖。

---

## 四、需要你来做决定的事项

以下问题我无法自行决定，需要你明确答复后才能推进。

### 【决定 1】找一个干净的多 kernel 样例（最高优先级，阻塞 PR6 merge）

PR6.4 的 `--enable-async=true` 路径无法用 `gemm_funccall`（bf16，cgra-mapper pre-existing crash）验证。

**你需要**：找一个 cgra-mapper 能正常跑通 mapping 的样例，条件是：
- fp32（规避 bf16 bug）
- 有多个 kernel 且它们之间有 BlockLoad/BlockStore 依赖（这样才能验证 gather 语义）
- 最好是 `.C` 源文件经 `adoracc.py` 产出的 `opt.mlir`

候选目录：`test/cgra-mapper/ADORATensor/*/`，逐个筛选哪个能跑通。

验证命令：
```bash
./build/bin/cgra-mapper --enable-async=true \
  <样例>/opt.mlir --output=<out>
# 在 emit 产物里 grep：
grep "await asyncio.gather" <out>
# 若出现 = PR6.4 Path 2 dep_summary 路径在工作
```

---

### 【决定 2】5 个 pre-existing lit failures 怎么处理

选项：
- (A) 单独开 PR 修复根因（需要你有时间排查 CDFGGen 相关的 upstream 问题）
- (B) 加 `XFAIL` 标记，先让 CI 干净，之后再说
- (C) 先不管（但 CI 噪音多，后续排查新问题时干扰大）

---

### 【决定 3】Loop-carried dep 的 Python emit 语义设计

`_lcDepSummary` 已经缓存，但在 Python 里如何表达跨迭代依赖，需要你做设计决定：

- **方案 A**：先不管，生成保守串行代码（最简单，正确但性能差）
- **方案 B**：`asyncio.Semaphore(1)` 控制流水级数（实现简单，语义有歧义）
- **方案 C**：`asyncio.Queue`，每次迭代 `put` store token，下次迭代 `get` 才开始 load（语义最精确）

---

### 【决定 4】`adora.dep_summary` 的消费者归属（P4.1 设计方向）

`psg_next_phase.md §5` 里有 6 个设计疑点，核心是：

**Q1**：mapper 是不是 dep_summary 的自然消费者？  
还是说消费者其实应该在 compiler 侧（`TaskPipelineAnalysis` / `DoubleBufferInsertion`），根本不需要跨越 pass/mapper 边界、不需要 attr 序列化？

**Q2**：dep_summary attr 挂 funcOp 还是 DataBlockOp？

如果你觉得消费者在 compiler 侧，则：
- `adora.dep_summary` 这个 attribute 本身的必要性要重新评估
- P4.1 "mapper log-only" 计划整体作废，改为 compiler 侧的 pipeline feasibility pass

请你看完 `docs/psg_next_phase.md §5` 后拍板。

---

### 【决定 5】`EmitCGRACall` / `EmitVitisSDK` 对称移植时机

这 ~200 行机械改动可以：
- (A) 现在就做（但 PR6.4 端到端还没验证，等于把未验证的逻辑复制三份）
- (B) 等 PR6.4 冒烟通过（决定 1 解决）后再做

建议 B，但需要你确认。

---

## 五、论文 vs 代码 差距（`gap.md` 整理）

这部分影响论文能否说圆，需要单独规划。

| # | 差距 | 严重度 | 现状 |
|---|------|--------|------|
| 缺口 1 | `analyzeDependencyInGraph()` 是空 stub | 🔴 高 | 函数体只有一行注释，堵死下游三条路径 |
| 缺口 2 | "Schedule Nodes on Graph" 三原则 | 🔴 高 | 论文正文和代码**同时缺失** |
| 缺口 3 | `RemoveRedundantBlockLoads` 整体被注释 | 🔴 高 | 约 40 行全部是 `/* */` 注释 |
| 缺口 4 | `TaskScheduleAlgorithm`（评估 + Pareto）| 🟠 中 | 只有候选枚举的壳，无评估逻辑 |
| 缺口 5 | Task Fusion（§4.3）| 🟠 中 | paper + code 同时缺，只有一句话描述 |
| 缺口 6 | Covered Data Transfer / Pingpong 决策层 | 🟠 中 | 代码有低层配置，决策层空白 |

这 6 个差距中，缺口 1-3 直接影响 paper 声称的核心贡献（依赖分析 + 调度优化），在投稿前需要明确是实现还是降低声称范围。

---

## 六、快速恢复上下文命令

```bash
# 切换到工作分支
cd /data00/home/loujiahang/adora/adora-compiler
git checkout jhlou/scheduletasks

# 看最新 commits
git log --oneline -10

# 看所有未解决问题
cat docs/pr6_session_handover.md

# 看论文 vs 代码差距
cat docs/gap.md

# 看 P4.1 设计疑点（需要你拍板）
cat docs/psg_next_phase.md

# 跑全部 lit 测试（预期：通过 ~13，失败 5 个 pre-existing）
ninja -C build check-adora 2>&1 | tail -20
```

---

## 七、总结：下一步建议行动顺序

| 步骤 | 做什么 | 谁做 | 依赖 |
|------|--------|------|------|
| 1 | 找干净多 kernel 样例，跑 `--enable-async=true` 冒烟 | **你** | — |
| 2 | 修 `_idToOp` 多 func cache clear（5 行） | 我 | — |
| 3 | 5 个 pre-existing failures 加 XFAIL / 修复 | **你决定方向**，我执行 | — |
| 4 | 冒烟通过后，对称移植 `EmitCGRACall` / `EmitVitisSDK` | 我 | 步骤 1 |
| 5 | 确认 LC dep Python 语义后，实现 emit | 你决定方案，我执行 | — |
| 6 | 拍板 P4.1 架构方向（Q1~Q6） | **你** | — |
| 7 | 缺口 1-3 补实现或降低论文声称范围 | **你决定**，我协助 | — |

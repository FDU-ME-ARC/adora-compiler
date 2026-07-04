# ADORA 任务调度流水线 — Pass 全景 & dep_summary 说明

> 本文回答四件事:
> 1. `adoracc.py` 流水线里每个 cgra-opt pass 干嘛、在哪个源文件;
> 2. 为什么是一串独立 pass 而不是合成一个;
> 3. `--adora-schedule-tasks` 与 `--adora-llm-pipeline-schedule` 的关系与分工;
> 4. `dep_summary` 到底是什么。
>
> 所有结论带源码 `文件:行号` 证据。代码库根:`adora-compiler/`。

---

## 0. 一张图看懂全流程

```
  C 源 (#pragma scop)
      │  cgeist -O2 --raise-scf-to-affine
      ▼
  ┌─────────────── 1_frontend ───────────────┐
  │  affine.for + memref (标准 MLIR)          │
  └───────────────────────────────────────────┘
      │  ① normalize 组
      ▼
  ┌─────────────── temp/normalize ────────────┐
  │  规范化后的 affine                          │
  └───────────────────────────────────────────┘
      │  ② kernel-extract 组
      ▼
  ┌─────────────── temp/kernel-extract ───────┐
  │  ADORA.kernel 出现(循环体被抽成 kernel)   │
  └───────────────────────────────────────────┘
      │  ③ kernel-opt 组
      ▼
  ┌─────────────── 2_kernel-opt ──────────────┐
  │  kernel 已优化 + datablock 化(LOAD/STORE) │  ← LLM 调度的输入就是这一份!
  └───────────────────────────────────────────┘
      │  ④ --adora-schedule-tasks(规则版调度)
      ▼
  ┌─────────────── 3_task-schedule ───────────┐
  │  .final.mlir:依赖分析 + async token       │
  │  + adora.dep_summary + module 标 scheduled │
  └───────────────────────────────────────────┘
      │  ⑤ --adora-kernel-dfg-gen(出 DFG,可选)
      ▼
   DFG(给 mapper / 可视化)

  ───────── 另一条独立分支(真正的 "LLM 调度")─────────
  2_kernel-opt 那份 ──▶ --adora-llm-pipeline-schedule ──▶ 带 hw_dep_type 的调度 MLIR
       (注:它读 dep_summary/token,让 LLM ranker 决定每个 task 的 dep_type)
```

**关键认知:**
- `adoracc.py` 末尾的 `--adora-schedule-tasks` 是**规则版调度**(做依赖分析、连 token、写 dep_summary),**不是 LLM**。
- **LLM 调度是单独的 `--adora-llm-pipeline-schedule` pass**,通常另起一条命令跑,输入是 `2_kernel-opt` 的 MLIR。

---

## 1. adoracc.py 的 cgra-opt 命令序列(从 pipeline.log 抓)

| 组 | cgra-opt 参数 | 产物目录 |
|---|---|---|
| ① normalize | `--affine-loop-normalize --affine-simplify-structures --normalize-memrefs` | `temp/normalize` |
| ② kernel-extract | `--canonicalize -reconcile-unrealized-casts --affine-loop-fusion --adora-extract-affine-for-to-kernel --arith-expand --memref-expand -cse` | `temp/kernel-extract` |
| ③ kernel-opt | `--adora-simplify-affine-loop-levels --canonicalize -cse --adora-simplify-loadstore --adora-math-rewrite --adora-adjust-kernel-mem-footprint=...` | `2_kernel-opt` |
| ④ schedule | `--adora-schedule-tasks=dump-token-graph=...` | `3_task-schedule` |
| ⑤ dfg-gen | `--adora-kernel-dfg-gen` | (stdout / DFG) |

driver 源:`tools/adoracc/adoracc.py`(extract :241 → simplify-loop :259 → simplify-loadstore :262 → math-rewrite :263 → adjust-footprint :265 → schedule-tasks :288 → kernel-dfg-gen :332)。

---

## 2. 每个自定义 `adora-*` Pass 的职责与源文件

Pass 注册名→源文件映射定义在两个 TableGen:
`include/ADORA/Dialect/ADORA/Transforms/Passes.td` 和
`include/ADORA/Dialect/ADORA/Lowering/LowerPasses.td`(仅 `adora-math-rewrite` 来自后者)。

| Pass 注册名 | 源文件 | 一句话职责 | 输入 → 输出 |
|---|---|---|---|
| `adora-extract-affine-for-to-kernel` | `lib/.../Transforms/Kernel/AffineForToKernelPass.cpp` | 把符合条件的 `affine.for` 循环体抽成 `ADORA.kernel` op | affine.for → ADORA.kernel |
| `adora-simplify-affine-loop-levels` | `lib/.../Transforms/Loop/AffineLoopSimplify.cpp` | 简化/折叠 kernel 内多层 affine 循环层级 | 多层 for → 规范层级 |
| `adora-simplify-loadstore` | `lib/.../Transforms/SimplifyLoadStore.cpp` | LICM:把循环不变的 load/store 提到循环外 | 冗余访存 → 精简访存 |
| `adora-math-rewrite` | `lib/.../Lowering/MathRewrite.cpp` | 数学算子改写/降级(便于后续映射) | 复杂 math → 基础算子 |
| `adora-adjust-kernel-mem-footprint` | `lib/.../Transforms/Loop/AdjustMemoryFootprint.cpp` | 按 cache/单数组大小切 datablock,显式化 LOAD/STORE 内存足迹 | kernel → datablock 化 kernel |
| `adora-schedule-tasks` | `lib/.../Transforms/TaskPipeline/ScheduleAdoraTasks.cpp` | **依赖分析 + token threading + 写 dep_summary**(规则版调度) | datablock kernel → 带 token+dep_summary 的 scheduled MLIR |
| `adora-kernel-dfg-gen` | `lib/.../Transforms/DFGgenPass.cpp` | 从 kernel 生成数据流图(DFG)给 mapper/可视化 | kernel → DFG |

> 注:`adora-schedule-tasks` = `ScheduleAdoraTasks.cpp`,经
> `createScheduleADORATasksPass()` 注册(Passes.td:387 → ScheduleAdoraTasks.cpp:900-904)。

### 通用 MLIR pass(非 adora 自定义,起辅助作用)
- `--affine-loop-normalize` / `--affine-simplify-structures` / `--normalize-memrefs`:把前端 IR 规整成统一形态,后续 adora pass 才有稳定假设。
- `--canonicalize` / `-cse`:标准清理(常量折叠、公共子表达式消除),在每组之间穿插。
- `--affine-loop-fusion`:循环融合,减少 kernel 数量。
- `-reconcile-unrealized-casts` / `--arith-expand` / `--memref-expand`:类型/算子/memref 降级补丁。

---

## 3. 为什么拆成多个 pass(而不是一个大 pass)

源码注释已点明设计原则。核心理由:

1. **单一职责 + 可组合(MLIR 哲学)**
   每个 pass 只做一件事(抽 kernel / 简化循环 / LICM / 切 footprint / 调度 / 出 DFG)。
   MLIR 的 PassManager 本来就鼓励小 pass 串成 pipeline,任意重排/插拔。

2. **normalize 与 transform 分离**
   ① 组(normalize)纯规整,不改语义;② ③ 组才做真正变换。
   先 normalize 让后面的 adora pass 能假设"输入已规范",大幅简化各自实现。

3. **可单独测试**
   每个 pass 在 `test/cgra-opt/` 下有独立 lit/smoke 测试(如
   `test/cgra-opt/schedule/llm_pipeline_schedule_live_smoke.sh`),
   单 pass 粒度便于定位回归。

4. **依赖外部进程的部分必须隔离**
   `--adora-llm-pipeline-schedule` 要 fork+exec 外部 Python ranker(见 §4),
   把它和纯 IR 分析的 `--adora-schedule-tasks` 拆开,才能:
   - 离线/CI 跑纯分析(不联网、不调 LLM);
   - 需要时再单独叠 LLM 决策。

5. **复用**
   `adora-schedule-tasks` 产出的 dep_summary 是公共产物,既可喂给
   `--adora-llm-pipeline-schedule`,也可喂给 estimator,拆开才好复用。

---

## 4. `--adora-schedule-tasks` vs `--adora-llm-pipeline-schedule`

两者**都叫"调度",但分工完全不同**,是「硬分析」与「软决策」的两层:

| 维度 | `--adora-schedule-tasks` | `--adora-llm-pipeline-schedule` |
|---|---|---|
| 源文件 | `ScheduleAdoraTasks.cpp` | `LLMPipelineSchedule.cpp` |
| 角色 | **生产者**(硬分析) | **消费者**(软决策) |
| 做什么 | O(N²) RAW/WAR/WAW/RAR 依赖分析(`analyzeDependencyInGraph` :186-257);`!ADORA.token` SSA threading(`threadTokensOnDMAs` :815);写 `adora.dep_summary`(:840);标 `module.adora.scheduled`(:883) | 读 dep_summary/async-token(`getAsyncDeps` :344,:367);fork+exec 外部 Python LLM ranker(:18-19);按 tile 数(`clNumTiles` :82-85)让 LLM 选每个 task 的 `hw_dep_type`(:104,取值 `LD_DEP_NONE`/`EX_LAST_TASK`/`ST_LAST_TASK` :99-101) |
| 外部依赖 | 无(纯 IR) | 需要 Python ranker 进程(可 dry-run) |
| 默认行为 | 总是跑完整分析 | 不给 `--ranker-cmd` = dry-run,用保守串行计划(plan 0) |
| 先后 | **先跑** | **后跑**,输入是带 dep_summary 的 IR |

**一句话总结:**
- `adora-schedule-tasks` 把"谁依赖谁"算出来、连成 token、写进 dep_summary(**这是事实,不是决策**)。
- `adora-llm-pipeline-schedule` 在这些事实之上,让 LLM 决定"每个 task 用哪种硬件依赖策略 / 怎么分 tile 重叠"(**这是决策**)。

> ⚠️ 关键反直觉点:`--adora-llm-pipeline-schedule` 不给 `--ranker-cmd` 时**不报错也不调 LLM**,
> 直接用保守串行默认计划。所以"让 LLM 真正参与调度",`--ranker-cmd` 实质上必须给。

### LLM 调度 pass 选项(全部可选,都有默认值)
源:`LLMPipelineSchedule.cpp:60-90`,注释明写 "all optional - defaults give safe no-op behaviour"。

| 选项 | 默认 | 不给的后果 |
|---|---|---|
| `--adora-llm-pipeline-schedule-ranker-cmd` | `""` | dry-run:不调 LLM,用保守串行 plan 0 |
| `--adora-llm-pipeline-schedule-ranker-timeout` | `10000` ms | 超时 10s |
| `--adora-llm-pipeline-schedule-ranker-log` | `""` | 不写决策日志 |
| `--adora-llm-pipeline-schedule-num-tiles` | `1` | 当单 tile,无多 tile 重叠 |
| `--adora-llm-pipeline-schedule-pe-per-tile` | `16` | 16(本就是常用值,常可省) |

**最精简能"真用 LLM"的命令:**
```bash
cgra-opt 2_kernel-opt/<k>_opt.mlir \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend dryrule" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  -o <k>_sched.mlir
# pe-per-tile=16 可省;不给 ranker-cmd 则退化为 dry-run。
```
ranker 后端(`task_schedule_ranker.py --backend`):`dryrule`(规则模拟,离线)/`openai`/`local`。

---

## 5. `dep_summary` 完整说明

### 5.1 它是什么
`adora.dep_summary` 是 `ScheduleAdoraTasks` 挂在 **func / kernel 上的一个 ArrayAttr 属性**,
记录编译器分析出的**数据块级依赖图**:每条边 = "task A 依赖 task B,依赖类型是 X"。
它是 token 之外的**完整依赖事实**(token 在文本序列化时可能丢 WAR/WAW,但 dep_summary 不丢)。

### 5.2 定义
- 依赖种类 enum:`include/ADORA/Dialect/ADORA/Analysis/DepKind.h:25-30`
  ```cpp
  enum class DataBlockDepKind {
    RAW = 0,  // read-after-write   (store -> later load)
    WAR = 1,  // write-after-read   (load  -> later store)
    WAW = 2,  // write-after-write
    RAR = 3   // read-after-read    (保守,load coalescing)
  };
  ```
  另外 kernel 体内还会出现 `LC-RAR`(loop-carried RAR,循环携带)。

- 每条边的 schema:`include/ADORA/Dialect/ADORA/Analysis/DepSummaryView.h:27-38`
  ```cpp
  struct DepSummaryRecord {
    int64_t          blockIdx;     // 数据块编号
    int64_t          srcNodeId;    // 源 task/op 序号
    int64_t          dstNodeId;    // 目标 task/op 序号
    DataBlockDepKind kind;         // RAW/WAR/WAW/RAR
    bool             mustOverlap;  // 是否必然重叠(保守)
  };
  StringRef getDepSummaryAttrName() { return "adora.dep_summary"; }
  ```

### 5.3 生成
- pass:`ScheduleAdoraTasks.cpp`
  - `analyzeDependencyInGraph`(:186-257):算出全部 4 种依赖
  - `appendDepEdgesToAttrList`(:264-282):序列化成 ArrayAttr 行
  - 挂载:`func->setAttr("adora.dep_summary", ...)`(:840)
- 同块判定(WAR/WAW 是否碰同一 SPAD):
  `lib/Dialect/ADORA/Analysis/DependencyAnalysis.cpp:621-664`(`AccessSameDataBlock`):
  同 memref + 归一化 affine-map 相等,OR 上保守 overlap。

### 5.4 实际样子(gesummv.final.mlir 真例)
```mlir
dep_summary = [{block_idx = 0 : i64, edges = [
  {dst = 4,  kind = "RAW", overlap = true, src = 3},
  {dst = 4,  kind = "RAW", overlap = true, src = 2},
  ...
  {dst = 8,  kind = "RAR", overlap = true, src = 2},   // RAR
  {dst = 11, kind = "WAR", overlap = true, src = 6},   // WAR(反依赖!)
  {dst = 5,  kind = "WAR", overlap = true, src = 0}    // WAR
]}]
// kernel 体内另有: {kind = "LC-RAR", step = 1, exact = true} ...
```
注意:**WAR/RAR 都在 dep_summary 里**,完整无丢失。

### 5.5 谁消费它
| 消费者 | 用 dep_summary 干嘛 | 状态 |
|---|---|---|
| `--adora-llm-pipeline-schedule` | 读 dep_summary + async-token 构造给 LLM 请求(`getAsyncDeps` :344,:367),据此让 LLM 选 `hw_dep_type` | 已用 |
| **mapper**(`cgra-mapper.cpp`) | 现在:walk 每个 FuncOp 读 `adora.dep_summary`/`adora.lc_dep_summary`,统计边数发 AgentTrace(:715-737,注释标 "observational only")。将来(T3/T4):喂进 OnlineRanker/PipelineScheduler,**替掉 `mapping.cpp` 里手写的依赖图** | 已接入,过渡中 |
| cycle-estimator(事件级仿真器) | **目前只读 async-token,没读 dep_summary** → 已知缺口(token 文本序列化丢 WAR/WAW,见 §5.6),补全 = 直接 parse `adora.dep_summary` | 缺口 |

> dep_summary 从设计上就是"**for mapper-side consumption via DepSummaryView**"
> (ScheduleAdoraTasks.cpp:833),即专为下游/mapper 读取而生的公共数据通道。

---

## 5.6 为什么有了 async token,还要 dep_summary?

这是最容易困惑的点。答案:**它俩是同一份依赖的两种表达,各有短板,当前处于"token 逐步取代 dep_summary"的迁移过渡期。**

### 决定性证据(源码注释)
- `ScheduleAdoraTasks.cpp:833-837`:dep_summary 挂载处 —
  > "attach `adora.dep_summary` for mapper-side consumption... gated by `emit-summary`
  > (default true) so that **once the ecosystem fully migrates to SSA tokens we can
  > retire this attribute**"
- `ScheduleAdoraTasks.cpp:879-882`:
  > "`adora.dep_summary` remains the **authoritative data channel in PR1**;
  > **PR2 will make async tokens carry the ordering and retire this attribute**"

即:token 是想取代 dep_summary 的新方案,但迁移未完成,现两者并存,**dep_summary 仍是权威源**。

### 两者对比

| | **async token(`!ADORA.token`)** | **dep_summary(属性)** |
|---|---|---|
| 形态 | IR 里的 SSA 值(def-use 链) | FuncOp 上的 ArrayAttr 数据 |
| 表达 | "谁的 token 喂给谁"(只有有/无) | 显式边表 `{src,dst,kind,overlap}` |
| **保留 kind** | ❌ 丢 RAW/WAR/WAW/RAR 类型 | ✅ 每条边带 kind |
| **文本序列化** | ❌ WAR/WAW 的 sink 是 store、无下游 → `async[...]` 打印时被跳过 → 文本丢边 | ✅ 属性完整落盘,不丢 |
| 读取 | 要遍历 SSA def-use 重建图 | 直接 `parseDepSummary()` 一个属性 |
| 谁天然爱用 | dialect 内 pass(IR 变换靠 SSA 保序) | mapper / estimator / 外部工具(读数据比重建 SSA 图省事) |

### 三个并存理由

1. **token 丢信息,dep_summary 补全**
   token 只说"A 依赖 B",不说是哪种。而调度/映射要区分:RAW 是真数据依赖(必须串行)、
   WAR/WAW 是反依赖(可用双缓冲绕开)。这个 **kind 只有 dep_summary 有**;
   且 token 文本序列化还会整条丢掉 WAR/WAW,dep_summary 不丢。

2. **消费方式不同**
   - dialect 内 pass:做 IR 变换,天然靠 SSA token 强制保序(token 是活在 IR 里的约束)。
   - mapper/estimator/外部工具:只想读一张依赖表做决策,重建 SSA 图麻烦,
     parse 属性最省事。dep_summary 就是为此而设。

3. **交叉校验(CI 安全网)**
   `verifyTokensMatchSummary()`(:581)专门核对 token 边 == dep_summary 边,
   不一致即报错。dep_summary 还充当 token 的对照真值,本身就需要两者并存。

### 一句话
> **token** = IR 里活的、强制保序的依赖约束(但丢 kind、文本会丢 WAR/WAW);
> **dep_summary** = 落在属性上的完整依赖表(带 kind、不丢、易读)。
> 现处于 token 逐步取代 dep_summary 的迁移中(PR1→PR2),但因 token 补不齐 kind、
> 且外部工具读属性更方便,dep_summary 仍是权威源,`emit-summary` 默认开。

> 推论:cycle-estimator 只读 token 会漏 WAR/WAW,**正确做法是像 mapper 一样读
> dep_summary**——它才是不丢边、带 kind 的权威源。

---

## 6. 速查:我该跑哪条命令

```bash
source env.sh    # 必须先 source,cgra-opt/cgeist 才在 PATH

# (A) C -> 规则调度后的 final MLIR(adoracc 一条龙,含 schedule-tasks)
adoracc.py <k>.c --work-dir /tmp/<k> -o /tmp/<k>/<k>_opt.mlir
#   产物: /tmp/<k>/adora-cc-ir/{1_frontend,2_kernel-opt,3_task-schedule}/

# (B) 真·LLM 任务调度(输入 = 2_kernel-opt 的 _opt.mlir)
cgra-opt /tmp/<k>/adora-cc-ir/2_kernel-opt/<k>_opt.mlir \
  --adora-llm-pipeline-schedule \
  --adora-llm-pipeline-schedule-ranker-cmd="python3 $LLM_RANKER --backend dryrule" \
  --adora-llm-pipeline-schedule-num-tiles=2 \
  -o /tmp/<k>/<k>_sched.mlir

# (C) 交给事件级仿真器出图
cd "$ADORA_COMPILER/tools/cycle-estimator"
python3 run.py --event-sim --mlir /tmp/<k>/<k>_sched.mlir --spec "$VITRA_SPEC" \
  --viz /tmp/<k>_gantt.png --viz-sram /tmp/<k>_sram.png
```

---

## 附:源文件速查表

| 主题 | 路径 |
|---|---|
| 流水线 driver | `tools/adoracc/adoracc.py` |
| Pass 注册(主) | `include/ADORA/Dialect/ADORA/Transforms/Passes.td` |
| Pass 注册(降级) | `include/ADORA/Dialect/ADORA/Lowering/LowerPasses.td` |
| 抽 kernel | `lib/.../Transforms/Kernel/AffineForToKernelPass.cpp` |
| 简化循环层级 | `lib/.../Transforms/Loop/AffineLoopSimplify.cpp` |
| LICM 访存简化 | `lib/.../Transforms/SimplifyLoadStore.cpp` |
| 数学改写 | `lib/.../Lowering/MathRewrite.cpp` |
| 调 footprint | `lib/.../Transforms/Loop/AdjustMemoryFootprint.cpp` |
| **规则调度+dep_summary** | `lib/.../Transforms/TaskPipeline/ScheduleAdoraTasks.cpp` |
| **LLM 调度** | `lib/.../Transforms/TaskPipeline/LLMPipelineSchedule.cpp` |
| DFG 生成 | `lib/.../Transforms/DFGgenPass.cpp` |
| 依赖种类 enum | `include/ADORA/Dialect/ADORA/Analysis/DepKind.h` |
| dep_summary schema | `include/ADORA/Dialect/ADORA/Analysis/DepSummaryView.h` |
| 同块判定 | `lib/Dialect/ADORA/Analysis/DependencyAnalysis.cpp` |

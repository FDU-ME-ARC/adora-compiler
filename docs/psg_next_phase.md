# PSG 下一阶段计划（P4.1 mapper 消费 + P2 仿射扩展路线图）

> 本文档为 `jhlou/scheduletasks` 分支之上的**下一阶段详细计划草案**，在动工前先落到 doc 供 review。
>
> 上游状态：`99c0fb6` (P1.0 + P4.0 落地，serializer + DepSummaryView parser + lit test 通过)。
>
> **本文档不是最终拍板，我有几个疑点不确定和你最初的设想是否吻合，见 §5。先 review 疑点，再决定是否开工。**

---

## 1. 目标回顾

P4.0（已完成）：pass 侧把 DataBlock 级依赖序列化成 `adora.dep_summary` ArrayAttr，挂在 funcOp 上；并提供 `parseDepSummary(Operation*)` 读回 `DepSummaryRecord` POD。

P4.1（本阶段拟做）：在 **mapper 侧**第一次真正消费这个 attribute —— *log-only*，不改变任何调度/分配决策。作为 P2/P3（affine 升级 / pipelining）真正依赖驱动 mapping 之前的烟测桥梁。

---

## 2. 现状盘点（调研结论）

| 项 | 位置 | 状态 |
|---|---|---|
| DataBlock dep 分析 | `lib/Dialect/ADORA/Transforms/TaskGraph/DataBlockDepAnalysis.{h,cpp}` | ✅ P1.0 完成 |
| 序列化到 attr | `ScheduleAdoraTasks.cpp:431` 附近 | ✅ P4.0 完成 |
| `parseDepSummary` | `lib/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.{h,cpp}` | ✅ Iter#4 完成，**位于 compiler 库，非 mapper 库** |
| mapper 入口 | `mapper/src/tensorop/mapGemm.cpp:140` `visitOp(GemmOp)`<br>`mapper/src/tensorop/mapConv.cpp` `visitOp(ConvOp)` | per-op 粒度 |
| mapper 类层级 | `Mapper` / `MapperSA` / `Mapping` | **不存在 `MappingContext`**（之前 plan 里是我臆造的） |
| mapper 读取模型 | `dfg_ir.h` + `adg_ir.h`（DFG/ADG） | 不直接读 MLIR funcOp attribute |
| dep_summary 的 consumer | 无 | `ScheduleAdoraTasks.cpp:431` 只留了一行注释 |

**关键 gap**：P4.0 的 producer 与 parser 都在 compiler 侧；而原设计 §6.2 说 parser 应该在 mapper 侧（`mapper/include/mapper/taskgraph_dep_view.h`）。Iter#4 把它放错了位置（放到了 compiler 的 `Transforms/TaskGraph/` 下）。**这是 P4.1 需要先修正的第一件事**，或者反过来承认设计文档 §6.2 过时、把它改掉。

---

## 3. P4.1 拟定方案（mapper 侧 log-only 消费）

### 3.1 parser 归属修正（二选一）

**方案 A（遵守原设计 §6.2）**：把 `DepSummaryView.{h,cpp}` 从 `lib/Dialect/ADORA/Transforms/TaskGraph/` **搬家**到 `mapper/include/mapper/taskgraph_dep_view.{h,cpp}`。mapper 才是真正的消费者。compiler pass 侧不需要 parser，只需要 serializer。

**方案 B（修订设计 §6.2，承认 parser 属于 compiler）**：保留现在的位置，但让 mapper CMake 依赖 `ADORATransforms`（或者抽一个更小的 `ADORAInterfaces`-style 头-only lib）。缺点：mapper 和 pass 库 ABI 耦合。

👉 **倾向方案 A**（parser 纯 POD，迁到 mapper 零成本；解耦最干净）。但这一步就是**我对最初架构不清楚的第一个点**，见 §5。

### 3.2 mapper 侧接入点

per-op 入口 `visitOp(GemmOp)` / `visitOp(ConvOp)` **不是**好的挂载点，因为 dep_summary 挂在 funcOp 上，同一 func 内每个 op 都会重复 parse。

拟在 `TensorDataflowGen` 更外层加一个 `perFuncOpInit(func::FuncOp)` 钩子（首次进入 func 时调一次）：

```cpp
void TensorDataflowGen::perFuncOpInit(func::FuncOp fop) {
  cachedDepSummary_ = parseDepSummary(fop);        // 一次解析
  if (dumpDepView_) dumpDepSummaryTable(fop, cachedDepSummary_);
}
```

`cachedDepSummary_` 为成员字段；`visitOp` 内目前**不消费**，仅把它作为观测信号。

### 3.3 CLI 开关

`cgra-opt` 端早已通过 attribute 自动携带；需要的是 mapper 入口（`cgra-mapper` / `mapperMain` 之类）加一个 `--dump-dep-view=false`（默认）。开启时把每个 block 的 RAR/WAR/WAW 条数、以及每条边的 `(from_task, to_task, block_id)` 三元组打到 stderr。

### 3.4 lit 测试

新增 `test/cgra-mapper/dep_view_dump.mlir`：
1. 跑 `cgra-opt --adora-schedule-tasks` 产生 `adora.dep_summary`
2. 管道到 `cgra-mapper --dump-dep-view`
3. `FileCheck` 验证 stderr 打出 `[dep-view] block=0 RAW=1 WAR=0 WAW=0` 之类规整一行

### 3.5 文件清单（方案 A）

新增：
- `mapper/include/mapper/taskgraph_dep_view.h`（由 `DepSummaryView.h` 迁移而来）
- `mapper/src/mapper/taskgraph_dep_view.cpp`（由 `DepSummaryView.cpp` 迁移而来）
- `test/cgra-mapper/dep_view_dump.mlir`

修改：
- `mapper/src/tensorop/tensorop.cpp`（加 `perFuncOpInit` 钩子 + 成员字段）
- `mapper/include/tensorop/TensorOp.h`（成员声明）
- `mapper/CMakeLists.txt`（加入新 cpp，同时**不再**依赖 compiler 的 DepSummaryView）
- `lib/Dialect/ADORA/Transforms/TaskGraph/CMakeLists.txt` 移除 `DepSummaryView.cpp`
- `include/ADORA/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.h` 删除
- `docs/psg_design.md §6.2`：更新 parser 真实路径
- `tools/cgra-mapper/*.cpp`：加 `--dump-dep-view` CLI

预计改动规模：~200 行（含 lit），3 个 commit：
1. `refactor: move DepSummaryView parser from compiler lib to mapper`
2. `feat: log-only dep_summary consumption in TensorDataflowGen`
3. `test: cgra-mapper lit for --dump-dep-view`

---

## 4. P2 路线图（远期，不在本阶段实作）

P2.1 affine dep：把 `DataBlockDepAnalysis` 内的 "常量矩形 overlap" 路径升级为 `FlatAffineValueConstraints` 上的 `checkMemrefAccessDependence` 参考实现。

P2.2 pipelining feasibility：`analyzeDependencyInGraph` 输出打通 `TaskPipelineAnalysis`，由它决定是否 II>1 软流水。

P3 mapper 真消费：P4.1 的 log-only 升级为真实决策（例如阻断 RAW 跨 pingpong buffer 的复用），此时 `DepSummaryRecord` 才产生工程价值。

---

## 5. 我的疑点（**开工前需要你拍板**）

这一段是重点——我实作前越想越觉得有几个地方和你最初的图景可能对不上，先写出来：

### Q1. mapper 真的是 dep_summary 的自然消费者吗？

`adora.dep_summary` 描述的是**两个 task 节点在 block 级 memref 区间上的读写冲突**。这种信息本质是**调度/流水**级语义（TaskGraph 层的 pipelining、DoubleBuffer 复用、跨 pingpong 容量决策），而 mapper 的工作语义是 **DFG → CGRA PE 阵列的 SA 空间布局**——两者关心的粒度和对象完全不同。

> 担心：我们可能在把一个**调度层**的分析结果硬塞到**布局层**的消费者身上。log-only 期"看不出问题"，真到 P3 要落消费逻辑时会发现：mapper 根本没有操作 task 级时序的手柄。

**候选的自然消费者**：
- (a) `TaskPipelineAnalysis` / 新的 `PipelineFeasibilityPass` —— 仍在 compiler 侧，**不需要经 mapper 中转**。
- (b) `DoubleBufferInsertion` / `RemoveRedundantBlockLoads` —— 也在 compiler 侧。
- (c) mapper 侧 **IO scheduler**（`mapper/src/mapper/io_scheduler.cpp`）——**可能**有用，因为它决定 DRAM↔片上 buffer 的流水——但得先看它是否已经有等价的 alias 分析。

**Q1 决策请求**：P4 这个"attribute 桥梁"本身是否必要？如果消费者其实在 compiler 侧，我们就根本不需要跨越 pass/mapper 边界、根本不需要 attr 序列化——只需要把 `DataBlockDepAnalysis` 的结果以 IR-level pass pipeline 的方式传递即可。**`adora.dep_summary` 这个 attribute 是不是一个"为了序列化而序列化"的产物？**

### Q2. Attribute 挂载粒度

P4.0 把 `adora.dep_summary` 挂在 **funcOp** 上，用 `{block_idx, edges}` 分组。但 TaskGraph 是 per-`DataBlockOp` 构建的，一个 func 里可能有多个 TaskGraph。

> 担心：funcOp 级汇总需要 consumer 反查 `block_idx → DataBlockOp`，增加 coupling；如果挂在 **DataBlockOp 本身**上（每个 block 一个自洽 attr），consumer 看到的就是 "this block's deps"，不需要任何上下文还原。

**Q2 决策请求**：是否要把挂载点从 funcOp 改成 DataBlockOp？这会让 `parseDepSummary(Operation*)` 的接口变得更自然（op 就是 block），代价是 P4.0 的 lit 测试要改。

### Q3. Parser 归属（§3.1 A/B）

见 §3.1。**我倾向 A（搬到 mapper）**，但前提是 Q1 判定"mapper 确实是消费者"。如果 Q1 判定消费者在 compiler 侧，则 parser 应该**留在 compiler**，甚至 §6.2 整段要重写。

### Q4. log-only commit 的价值

P4.1 按现在的写法会产生 **零行为变更的 ~200 行代码**。`log-only` 在 compiler 工程里通常是过渡态，MR review 时容易被问"这代码现在保护的是什么？"。

> 担心：把"P4.1 log-only"与"P3 真消费"拆成两个 MR，第一个 MR 本身讲不出商业价值。

**Q4 决策请求**：是否合并 P4.1 + P3（最小可消费，比如 "RAW 边触发 pingpong 冲突 warning"），一起进一个 MR？还是坚持 log-only 先入？

### Q5. 设计文档 §6 过时

`docs/psg_design.md §6.2` 写的是 `mapper/include/mapper/taskgraph_dep_view.h`，但 Iter#4 实际落到 `include/ADORA/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.h`。**我没在 Iter#4 同步更新设计文档**——这是我的一处疏忽。需要在 P4.1 首个 commit 里补齐。

### Q6. `cgra-mapper` 入口到底是哪个二进制

我还没定位出 `cgra-mapper` 这个命令在代码里的对应 `tools/` 目录（如果存在的话）。`--dump-dep-view` 加在哪里还需要先翻 `tools/`。如果这个二进制不存在、mapper 是作为一个 library 被别的驱动调用的，那 CLI 方案要换成 pass option。

---

## 6. 动手前的决策清单

在我写一行新代码之前，请你回答：

1. **Q1**：mapper 是不是真的消费者？还是应该把消费者定位回 compiler 侧？
2. **Q2**：attribute 挂 funcOp 还是 DataBlockOp？
3. **Q3+Q5**：parser 归属修正 → 搬家，还是承认设计文档过时？
4. **Q4**：log-only 单独 MR，还是合并到一个最小真消费 MR？
5. **Q6**：mapper 的 CLI 入口点是什么？

只要 Q1 / Q4 任何一个答案否定当前路线，P4.1 的整体方案都要改。所以**我强烈建议在这里先停一下**。

---

## 7. 执行预算（假定上面全部按当前方案过关）

- 编码：~1.5 工作日
- lit + 回归：~0.5 工作日
- docs 更新：~0.5 工作日
- 合计：**~2.5 工作日**，产出 3 个 commit，**不 push 到 develop**（和之前约定一致）。

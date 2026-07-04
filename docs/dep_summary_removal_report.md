# `adora.dep_summary` 删除可行性调研报告

调研分支：`origin/jhlou/scheduletasks`（工作副本 `tmp-fix-blockstore`）
调研方法：全代码库静态审查（`lib/`、`tools/`、`mapper/`、`include/`、`test/`、`docs/`、`experiment/`），逐个核实生产者与消费者，附 `file:line` 证据。
一句话结论：**`adora.dep_summary` / `adora.lc_dep_summary` 是"有写无读"的死数据，可直接删除，mapping / emit 全程不使用它，删除不需要补任何东西。**

---

## 1. 背景：两条并行的依赖通道

调度 pass `--adora-schedule-tasks` 分析出任务间依赖后，把同一份依赖信息写成两份：

| 通道 | 形态 | 引入 | 现状 |
|------|------|------|------|
| `adora.dep_summary` 属性 | funcOp 上的 ArrayAttr，per-edge `{src,dst,kind,overlap}` | PR1（早） | **无人读回** |
| SSA `!ADORA.token` | async op 的 `async [%tok]` operand / result | PR2（晚） | **唯一真实依赖通道** |

两者同源，都来自 `graph->depEdges()`（`ScheduleAdoraTasks.cpp:270` 写属性，`:511` 写 token）。token 上线后，dep_summary 沦为冗余的第二份拷贝，仅剩调试可读性与 CI 校验用途。

---

## 2. 生产侧（谁写）

- 唯一写入点：`lib/Dialect/ADORA/Transforms/TaskPipeline/ScheduleAdoraTasks.cpp`
  - `appendDepEdgesToAttrList()`（`:264`）序列化每条边为 `{src,dst,kind,overlap}`；`src`/`dst` 是 **TaskGraph 内部 node id**（`graph->getNodeId()`），IR 里并不存在，这是它只能挂在整个 function 上做全局表的根本原因。
  - `func->setAttr("adora.dep_summary", ...)`（`:837`）、`func->setAttr("adora.lc_dep_summary", ...)`（`:871`）。
- 受 pass option `emit-summary` 门控（`include/.../Passes.td:396-421`），默认 `true`。

---

## 3. 消费侧（谁读）—— 核心结论

**逐点核实，mapping / emit 全链路零消费：**

| 检查点 | 是否读 dep_summary | 证据 |
|--------|:---:|------|
| mapper 核心 `mapper/src/mapper/*.cpp`（mapping / mapGemm / mapConv / io_scheduler …） | 否 | grep 无命中 |
| MLIR→C++ 桥 `tools/cgra-mapper/cgra-mapper.cpp` | 否 | 仅 `:186` 一行注释提及 |
| emit 层 `mapper/src/emit/*.cpp`（EmitPytest / EmitCGRACall / EmitVitisSDK） | 否 | 依赖走 SSA token，见下 |
| 调度/降级 pass（ScheduleAdoraTasks / AssignStreams / LowerAsyncTokens） | 否 | 内部读内存 `graph->depEdges()`，不读属性 |
| LLMPipelineSchedule / schedule_order_decision / partition_budget | 不存在 | 本分支无这些文件（在其它分支） |
| cycle-estimator | 否 | grep 无命中 |
| `parseDepSummary` / `DepSummaryView`（唯一读属性的 API） | 死代码 | 定义于 `DepSummaryView.cpp:51/77/80`，全仓无调用者 |

**mapping / emit 实际依赖 = SSA token**，经 `ADORA::getAsyncDeps(op)`（定义 `include/.../Utility/Utility.h:147`）读取：
- `EmitCGRACall.cpp:72 / 1328`
- `EmitPytest.cpp:92 / 115`（`EmitPytest` 还在 emit 时用 `computeStreamIds()` 自行按 token 边染色算 stream，`EmitPytest.cpp:81-84 / 1764`）

**唯一名字含 "Summary" 的校验**：`verifyTokensMatchSummary()`（`ScheduleAdoraTasks.cpp:587`，受默认关闭的 `cross-check-summary-vs-token` 门控）比对的是**内存 TaskGraph vs token**，**不读属性**，删属性不受影响。

---

## 4. WAW / WAR / RAW / RAR 这些 kind 重要吗？在 mapper 中用到吗？

这是本报告的重点问题，独立调研结论如下：

### 结论：kind 在 mapper 与 emit 中**完全没被用到**，四种 kind 对最终产物**等价**。

| 阶段 | 是否使用 kind | 证据 |
|------|:---:|------|
| kind 枚举定义 | — | `DataBlockDepKind{RAW,WAR,WAW,RAR}` 定义于 `include/.../Analysis/DepKind.h:25-30`（含 `toString`/`parseDepKind`） |
| 建图分析阶段 | 生产但**不据其分支决策** | `ScheduleAdoraTasks.cpp:218-251` 四路分类算出 kind，但 `:196-200` 的建边 lambda 对所有 kind **一视同仁**（统一 `addDepEdge`），无按 kind 分支 |
| SSA token | **不携带 kind** | `threadTokensOnDMAs`（`:503-579`）只取 `src/dst` 传纯 `Value` token；`rebuildAsyncLoad/Store/Kernel`（`:330/367/399`）签名无 kind；async op 内置属性集（`:299-308`）无 kind |
| mapper 核心 | 否 | `mapper/src/mapper` 下对 `DataBlockDepKind` / `.kind` grep 零命中 |
| emit 层 | 否 | `EmitCGRACall.cpp:69-91` `computeDepFlag` 按 **slot 距离**选同步宏（与 kind 无关）；`:1326-1331` `ex_dep` 仅判空；`EmitPytest.cpp:92-127` `getAsyncDeps` 生成 gather，也与 kind 无关 |
| 属性 dep_summary | 序列化后无人读回 | 写于 `:274`，`DepSummaryView.cpp:36-46` 能解析出 `out.kind`，但 `parseDepSummary` 无调用者 → 死数据 |

### 唯一一处 kind 真正影响产物

`LoopCarriedDep.cpp:149`：`if (!includeRAR && e.kind == LCKind::RAR) continue;`
两个调用方都传 `includeRAR=false`（`ThreadLoopCarriedTokensImpl.cpp:54`、`ScheduleAdoraTasks.cpp:790`），即 loop-carried token 化时**排除 RAR 边**。

- 这是一个 **"RAR vs 其余" 的二元开关**（RAR 读后读，通常无需强制排序，故不生成 token）。
- 注意它用的是 **LC 分析里的 `LCKind`**，而非 dep_summary 属性；即使删掉 dep_summary，这个 RAR 过滤逻辑照常工作。
- RAW / WAR / WAW 三者之间**依旧不区分**。

### 含义

- **对当前 mapping / emit**：把四种 kind 全当成"纯 happens-before"，产物不变——kind 的区分目前只存在于 dep_summary 属性里，无人消费。
- **对未来 LLM 调度**：之前设计的 `LLMPipelineSchedule`（per-task/per-edge dep_type）曾计划读 dep_summary 的 per-edge kind 来决定 `LD_DEP_NONE / EX_LAST / ST_LAST`。**但该 pass 不在本分支**。若将来引入，正确做法不是保留这张 node-id 全局表，而是把 kind 就近挂到 async op 属性上（`dep_kinds = [...]` 与 `async[%tok]` 顺序对齐），得到干净的 per-edge 载体。

---

## 5. DepSummaryView 组件

| 文件 | 行数 | 处置 |
|------|------|------|
| `include/.../Analysis/DepSummaryView.h` | 43 | 删 |
| `lib/.../Analysis/DepSummaryView.cpp` | 90 | 删 |
| shim 头 `include/.../TaskPipeline/TaskGraph/DepSummaryView.h` | 3（标注 "Delete after downstream"） | 删 |
| `lib/.../Analysis/CMakeLists.txt:4` | — | 移除该行 |

- 除自身与 shim 外无任何 `.cpp` include 它，**删除不影响编译**。
- **不可删**：`DepKind.h` 的 `parseDepKind` / `DataBlockDepKind` 仍被 TaskGraph 使用。

---

## 6. 测试影响

`test/cgra-opt/schedule/` 下 6 个 lit 文件引用 `dep_summary`：

| 文件 | 处置 |
|------|------|
| `schedule_tasks_dep_summary.mlir` | 专测该属性，**整删** |
| `schedule_complex_viterbi.mlir` | 删相关 CHECK 行（保留 token 检查） |
| `schedule_complex_sobel.mlir` | 同上 |
| `schedule_complex_ffn.mlir` | 同上 |
| `schedule_tasks_default_emit_token.mlir` | 同上 |
| `schedule_cgra_tasks_tokens.mlir` | 同上 |

`lc_dep_summary` 无任何 lit CHECK。

---

## 7. 最终结论与删除清单

### 结论

1. **mapping 过程用到 dep_summary 吗？** 否。mapper 核心、桥、emit 层全部零消费，实际依赖走 SSA token。
2. **WAW/WAR/RAW/RAR 重要吗？在 mapper 用到吗？** 不重要，mapper/emit 完全不用；四种 kind 对产物等价。唯一影响产物的是 LC 分析里 `includeRAR=false` 的 RAR 二元过滤（且不依赖 dep_summary 属性）。
3. **能否直接删除？** **能，纯减法，无需补任何东西。** 当前分支无任何消费者。

### 删除改动清单

| 类别 | 动作 | 位置 |
|------|------|------|
| 写入代码 | 删 `appendDepEdgesToAttrList` 及两处 `setAttr` + LC 序列化 | `ScheduleAdoraTasks.cpp:264, 837, 871` |
| pass 选项 | 删 `emit-summary`、`cross-check-summary-vs-token`（及配套 `verifyTokensMatchSummary`） | `Passes.td:396-421`；`ScheduleAdoraTasks.cpp:587` |
| 死代码组件 | 删 3 文件 + CMake 一行 | `Analysis/DepSummaryView.{h,cpp}`、shim 头、`Analysis/CMakeLists.txt:4` |
| 保留 | `DepKind.h` 的 `parseDepKind` / `DataBlockDepKind` 不动 | — |
| 测试 | 1 整删 + 5 删 CHECK 行 | 见 §6 |

### 建议顺序

先删 dep_summary（现在即可，纯减法）；未来若 `LLMPipelineSchedule` 需要 per-edge kind，再给 async op 增加 `dep_kinds` 属性，而非保留 node-id 全局表。

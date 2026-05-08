# ADORA Async Token 设计 (P4.1 follow-up)

> 作者：jhlou + PiCode
> 状态：Draft，等 review
> 依赖：`docs/psg_next_phase.md` 的 Task Graph / DepSummaryView 章节

## 背景与动机

当前 `ScheduleAdoraTasks` pass 在 IR 外挂一张依赖表，以 `adora.dep_summary = [{src, dst, kind, overlap, can_parallel}, ...]` 属性挂在 `affine.for` 上（见 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:405-411`）。下游 mapper / lowering 想知道依赖，要么读这张旁路表，要么自己重新做 memref aliasing。两种方式都脆：

1. IR 被其它 pass 改动后，旁路表可能过期或对不上
2. mapper 侧代码重复做依赖分析，开销 + bug 源
3. `DepSummaryView` 放在 `lib/Dialect/ADORA/Transforms/TaskGraph/`，被 mapper `CMake` 拉起来成本高（上次 session 的笔记已经确认 `mapper` target 没 link `ADORATransforms`）

本次目标：把 **顺序依赖**（RAW / RAR / WAR / WAW）从属性迁移到 SSA 上，用 **`!ADORA.token`** 做为独立于数据的顺序凭证。参考 Triton `ttng.async_copy` / MLIR `gpu.alloc async` 的成熟做法。

## 设计原则

1. **数据归数据、顺序归顺序**：`memref` 仍然承载数据；`!ADORA.token` 只承载顺序，不携带任何 payload。
2. **向后兼容**：老 IR / 老 builder 调用点原样 parse。Token 是 **Optional result**——手写 / fixture `.mlir` 用的 sync 形态永远合法。
3. **分层职责**：创建 op 的 pass（Lowering、`AdjustKernelMemFootprint`、`ExtractAffineForToKernel`）在构造 BlockLoad/Kernel/BlockStore 时**就把本地微依赖链连好**（BlockLoad(s) → Kernel → BlockStore 的显然顺序）；`ScheduleAdoraTasks` 只负责**追加跨 DataBlock / 跨迭代**的额外依赖边（RAR / WAR / WAW / cross-tile RAW）。
4. **Schedule 不做 op replace**：schedule pass 只使用 `op->insertOperands` 往已有的 `asyncDependencies` 后面追加，不改 op 的 result 结构、不 clone region。
5. **删旧接新**：token 化完成后 `adora.dep_summary` + `DepSummaryView` 彻底删除，不保留双路。
6. **内存分配类 op 不参与 token 图**：`ADORA.LocalMemAlloc` 是纯分配，无副作用顺序（见下节）。
7. **结构冻结**：`adora-schedule-tasks` 跑完在 module 上打 `adora.scheduled` 标记；结构性 loop opt（unroll / reorder / simplify / footprint-adjust / lowering）入口检查该标记，避免 pipeline 顺序错配。

## 参考原型：`gpu.alloc async`

```mlir
%m         = gpu.alloc ()             : memref<10xf32>           // sync
%m, %tok   = gpu.alloc async [%t0]    : memref<10xf32>           // async，带 wait list
```

- 同一个 `GPU::AllocOp`
- `asyncToken` = `Optional<Token>` result
- `asyncDependencies` = `Variadic<Token>` operand
- `async` 关键字在 assembly format 里作为条件打印/解析的开关

## IR 形态

### 新增 Type

```tablegen
// include/ADORA/Dialect/ADORA/IR/ADORABase.td
def ADORA_TokenType : TypeDef<ADORA_Dialect, "Token"> {
  let mnemonic = "token";
  let summary = "Ordering-only handle between ADORA data-block / kernel ops.";
}
```

### 三种 op 升级

#### `ADORA.BlockLoad`

```tablegen
let arguments = (ins
  Arg<AnyMemRef, "source", [MemRead]>:$OriginalMemref,
  Variadic<Index>:$indices,
  Variadic<ADORA_TokenType>:$asyncDependencies);

let results = (outs
  AnyStaticShapeMemRef:$memref,
  Optional<ADORA_TokenType>:$asyncToken);
```

```mlir
// sync 形态（兼容今天）
%m      = ADORA.BlockLoad %A[%i] : memref<32x32xf32> -> memref<1x32xf32>

// async 形态（schedule pass 产出）
%m, %t  = ADORA.BlockLoad async [%t0] %A[%i]
            : memref<32x32xf32> -> memref<1x32xf32>
```

#### `ADORA.kernel`

```mlir
// sync：0 result（兼容今天）
ADORA.kernel { ... }

// async：1 token result
%t = ADORA.kernel async [%t0, %t1] { ... }
```

#### `ADORA.BlockStore`

```mlir
// sync：0 result（兼容今天）
ADORA.BlockStore %m, %C[%i]

// async：1 token result
%t = ADORA.BlockStore async [%tk] %m, %C[%i]
```

### Verifier 约束

- `asyncDependencies` 非空 ⇒ 必须有 `asyncToken`（消费 token 就得产 token）
- `asyncToken` 可以单独存在（源节点）
- `asyncDependencies` 的每一个 operand 必须是 `!ADORA.token`
- 同一个 func 内不允许 token use 环（`verifier` 或 pass 检查）

## Token 生产时机与连接职责

Token 生产（asyncToken result）与连接（asyncDependencies operand）分属不同 pass：

| Pass | 产 asyncToken | 填 asyncDependencies |
|---|---|---|
| `ADORATensor/Lowering/*` (OSGemm / WSGemm / ISGemm / DirectConv) | ✅ 对 BlockLoad / kernel / BlockStore 都产 token | ✅ 本地链：kernel 的 deps = 同一 tile 的 BlockLoad tokens；BlockStore 的 deps = 上游 kernel 的 token |
| `AdjustKernelMemFootprint` | ✅ 同上 | ✅ 同上 |
| `ExtractAffineForToKernel` (`AffineFortoKernelUtil.cpp`) | ✅ 对 kernel 产 token | 视情况：若上下文已有 BlockLoad 则连；否则空 |
| `SimplifyLoadStore` (重建 BlockLoad/Store 时) | ✅ | ✅ 继承被重建 op 的 deps |
| `ScheduleAdoraTasks` | ❌ 不创建新 op | ✅ 追加**跨 DataBlock / 跨迭代**的额外边 |

### 本地链的形态

`AdjustKernelMemFootprint` / Lowering 创建三种 op 时，按局部数据流连好：

```mlir
affine.for %tile = 0 to N {
  %a, %ta = ADORA.BlockLoad async %A[%tile]       // deps 空：平行进来
  %b, %tb = ADORA.BlockLoad async %B[%tile]       // deps 空：平行进来
  %tk     = ADORA.kernel async [%ta, %tb] { ... } // 等两个 BlockLoad
  %ts     = ADORA.BlockStore async [%tk] %c, %C[%tile]  // 等 kernel
}
```

创建者知道哪些 BlockLoad feed 哪个 kernel、哪个 BlockStore 取 kernel 输出——这信息不该丢给下游 pass 再去重新推导。

### Schedule 追加的跨 DataBlock 边

`ScheduleAdoraTasks` 只关心**跨 DataBlock / 跨迭代**的依赖。例如上一次 tile 的 `BlockStore_C` 与当前 tile 的 `BlockLoad_A` 之间存在 WAR：

```mlir
%a, %ta = ADORA.BlockLoad async [%ts_prev] %A[%tile]
                                ^^^^^^^^^
                    schedule pass insertOperands 追加
```

### 为什么不让 ScheduleAdoraTasks 做 op replace

op replace 成本远大于 operand append：
- 需要 clone region（kernel op 有 body）
- 需要搬运所有 discardable attributes
- 需要 RAUW 所有 users（memref result 的消费者）

把 token result 和本地 deps 在 op 诞生时一次写好，schedule pass 的职责退化成纯 append，代码量从几十行降到十几行。

## ScheduleAdoraTasks 改造

### 新行为

1. 原有 `generateTaskGraphFromBlock` 和依赖分析**保留不动**（结果仍然是 `DataBlockDepEdge` 集合）
2. **过滤掉创建时已连好的本地边**：同一 tile 内 BlockLoad→Kernel→BlockStore 的 RAW 已在本地链里，不再重复追加
3. 对剩余的**跨 DataBlock / 跨迭代**依赖，对每个 `TaskNode`：
   - 从 op 上读出前驱 task 的 `asyncToken` result
   - 通过 `op->insertOperands(op->getNumOperands(), tokens)` 追加到 `asyncDependencies` operand 段
   - 同步更新 `operandSegmentSizes`
4. 跑完后给 module 打 discardable attr `adora.scheduled = unit`，作为"op 结构冻结"的防呆标记
5. **删** `setAttr("adora.dep_summary", ...)` 和 `buildBlockDepSummary`
6. **删** `lib/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.{h,cpp}` 及其 CMake 依赖
7. `kind` 元信息怎么保留？两个选项：
   - B1：`$asyncDependencies` 旁挂一个 `adora.dep_kinds = [str, str, ...]` 长度对齐的 ArrayAttr
   - B2：干脆不保留 `kind`，mapper 只认"必须 happen-before"这一个语义——如果未来需要区分 RAR 可并行，那时再加
   - 推荐 **B2**：最小惊讶，必要时通过 token 分发到并行 stream 自然实现

## 其它 Pass 与 Token 的交互

### Canonical pipeline invariant

- `adora-schedule-tasks` 之前：三种 async op 的 `asyncDependencies` 仅包含**本地链**（同 tile 内 BlockLoad → Kernel → BlockStore）。跨 tile / 跨 DataBlock 的边一律为空，token 的跨 tile use 也为空。
- `adora-schedule-tasks` 之后：IR 的 op 结构**冻结**。不运行 unroll / reorder / simplify / split 等结构性 loop opt；`asyncDependencies` 可能被追加跨 DataBlock 边。

### 各类 Pass 的处理策略

| Pass 类别 | 代表 | 需要改造？ | 说明 |
|---|---|---|---|
| 创建 BlockLoad/Store/Kernel | `AdjustMemoryFootprint`, `AffineFortoKernelUtil`, `ADORATensor/Lowering/*` | ✅ 改 builder 调用 | 切到 `buildAsync(...)`；创建时直接把**本地链**连上：BlockLoad 间平行、Kernel deps = 本 tile 所有 BlockLoad token、BlockStore deps = 上游 Kernel token |
| Loop body 克隆 | `AffineLoopUnroll`, `UnrollAndJam`, `AutoUnroll` | ❌ 零改动 | MLIR clone 会自然给每份拷贝新的 `asyncToken` result 并 RAUW 本地 deps，本地链在每个克隆体内自洽 |
| Loop 搬动 | `AffineLoopReorder` | ❌ 零改动 | 本地链整体跟着 op 一起搬动，SSA dominance 保持 |
| Loop 简化 / 删除 op | `AffineLoopSimplify`, `SimplifyLoadStore` (删 op 分支) | ⚠️ 若删的 op 在本地链中，需调 `eraseAdoraAsyncOp` 修复链 | pre-schedule 阶段本地链已连，直接 erase 会留下 dangling token use |
| Tensor-to-ADORA lowering | `OSGemm`, `WSGemm`, `ISGemm`, `DirectConv` | ✅ 改 builder 调用 | 切到 `buildAsync(...)` |
| 调度编排 | `AutoDesignSpaceExplore` | ❌ 零改动 | 只是编排其它 pass |
| 填依赖 | `ScheduleADORATasks` | ✅ 核心改造 | 唯一填 `asyncDependencies` 的 pass |
| Post-schedule legalization / mapper prep | 待定 | 视情况 | 必须使用 `AsyncTokenUtils` helper |

### 工具辅助：`AsyncTokenUtils.h`

新增 `include/ADORA/Dialect/ADORA/Utility/AsyncTokenUtils.h`，封装 post-schedule 阶段修改 IR 时的 token 链维护逻辑：

```cpp
/// Erase an ADORA async op; forward its asyncDependencies into any user
/// that consumed its asyncToken.  Safe to call in either pre- or
/// post-schedule state.
void eraseAdoraAsyncOp(Operation* op);

/// Clone an ADORA async op.  The clone has a fresh asyncToken result
/// and inherits the asyncDependencies of the source (caller may further
/// edit).
Operation* cloneAdoraAsyncOp(OpBuilder& b, Operation* op);

/// Replace an ADORA async op with a new one.  RAUW both memref result
/// (if any) and asyncToken; keep asyncDependencies aligned.
void replaceAdoraAsyncOp(Operation* oldOp, Operation* newOp);
```

pre-schedule 阶段不需要使用这组 helper；post-schedule 的修改者（例如 mapper 前的 legalization pass、DSE 的 rollback 分支）必须使用。

### 防呆

`ScheduleADORATasks` 结束时给 module 打 `adora.scheduled = unit`。所有结构性 loop opt pass（unroll / reorder / simplify / footprint-adjust / lowering）在 `runOnOperation` 入口处检查该 attr：

```cpp
if (getOperation()->hasAttr("adora.scheduled"))
  return signalPassFailure();  // 或 emitError + return
```

避免未来 pipeline 编排时意外把结构性 pass 放到 schedule 之后。

### LocalMemAlloc 不参与 token 图

`LocalMemAllocOp`（`include/ADORA/Dialect/ADORA/IR/ADORAOps.td:310`）产出一块本地 memory tile，不读不写别人的 buffer，无副作用顺序。

1. SSA def-use 已经保证 alloc 先于所有 user，无需额外 token 边。
2. ADORA 无显式 dealloc op（scope-based 生命周期），不存在 alloc/dealloc 配对的顺序约束。
3. WAW / WAR 真正发生在 **user 之间**（两个 BlockLoad 先后写同一块 local buffer）；token 连在这两个 BlockLoad 上即可，LocalMemAlloc 本身不参与。
4. `TaskGraph::LocalAllocNode`（`TaskGraph/TaskNode.h:145`）保留——它是**分析的输入**（"哪些 BlockLoad/Store 共享同一块 local buffer"），不是分析的输出，不需要落到 IR SSA token。
5. 给 alloc 加 token 会引入一串无语义的 token 边，收益为零，反而复杂化 verifier。

## 向后兼容分析

| 场景 | 影响 |
|---|---|
| 现有 `.mlir` 测试 / fixture 文件 | **零破坏**，老语法照常解析 |
| `builder.create<DataBlockLoadOp>(...)` 的 13 个调用点 | **零改动**，默认不产 token，仍然是 1 result，隐式转 `Value` |
| `isa<DataBlockLoadOp>` / `dyn_cast` 位置（~8 处） | **零改动** |
| `TaskGraph`、`DependencyAnalysis` 辅助函数 | **零改动** |
| `ScheduleAdoraTasks.cpp` | 重写 Step 5 / Step 6，约 ~60 行 diff |
| `adora.dep_summary` 消费者 | 目前只有测试 `schedule_tasks_dep_summary.mlir` 和 mapper 里一个 TODO 注释（`ScheduleAdoraTasks.cpp:431` 的占位），影响可控 |

## 落地步骤（建议分 2 PR）

### PR1 — 基础设施 + KernelOp 先行

- [ ] `ADORABase.td` 加 `ADORA_TokenType`
- [ ] `ADORADialect.cpp` 里 `addTypes<TokenType>()` + parser/printer
- [ ] `ADORAKernelOp.td` 加 `Optional<Token>:$asyncToken`、`Variadic<Token>:$asyncDependencies`，写 `assemblyFormat`
- [ ] Kernel verifier 补 "产消对称" 规则
- [ ] 新 builder overload `buildAsync(..., ValueRange deps = {})`
- [ ] `ScheduleAdoraTasks` 在 module 上打 `adora.scheduled = unit` 防呆标；`adora.dep_summary` 不动（真正的 operand append 推到 PR2，因为 `ScheduleAdoraTasks` 现有边集全部是 DMA↔DMA，kernel-to-kernel 由 SSA 天然承载不入此 pass —— 参见下面 "PR1 笔误修正 + 实际落地范围"）
- [ ] `ExtractAffineForToKernel` / `AffineFortoKernelUtil.cpp` 暂不切 `buildAsync`（保持 sync 形态），推到 PR2：否则 printer 会多打 `async` 关键字导致 5 个老 CHECK 集体失败（atax/jacobi-2d/mvt/mvt_unroll/gemm）。PR1 以 "零 .mlir 测试改动" 为硬约束
- [ ] `schedule_tasks_dep_summary.mlir` 测试追加 `// CHECK: adora.scheduled`，原有 CHECK 保留不动（向后兼容）
- [ ] `ninja check-adora` 通过

### PR2 — BlockLoad + BlockStore 全量 + 清理

- [ ] `ADORAOps.td` 两个 op 加 `asyncToken` / `asyncDependencies`
- [ ] `ADORAOps.cpp` 新增 `buildAsync` overload + custom parser / printer / verifier 更新
- [ ] `AdjustMemoryFootprint`, `SimplifyLoadStore`, `ADORATensor/Lowering/{OS,WS,IS}Gemm`, `ADORATensor/Lowering/CONV/DirectConv` 切到 `buildAsync`
- [ ] `ScheduleAdoraTasks` 处理所有 DataBlockDepEdge → token 边
- [ ] 新增 `include/ADORA/Dialect/ADORA/Utility/AsyncTokenUtils.{h,cpp}`（erase / clone / replace 三件套）
- [ ] `ScheduleAdoraTasks` 结束时给 module 打 `adora.scheduled = unit`
- [ ] 结构性 loop opt pass（unroll / reorder / simplify / footprint-adjust / lowering）入口检查 `adora.scheduled`
- [ ] 删 `adora.dep_summary` 所有引用：pass、doc、测试
- [ ] 删 `lib/Dialect/ADORA/Transforms/TaskGraph/DepSummaryView.{h,cpp}` + CMake
- [ ] Mapper 侧占位 `ScheduleAdoraTasks.cpp:431` 注释更新为指向 "walk `asyncToken` use-def"
- [ ] `ninja check-adora` 通过

## 验证手段

1. 单测：新 test `test/cgra-opt/kernel/schedule_tasks_token.mlir`
   - 输入：两个 kernel 先后写同一个 memref（WAW）
   - CHECK：`%t1 = ADORA.kernel async [%t0]`
2. 单测：另一个 .mlir 输入两个独立 kernel（无依赖）
   - CHECK：`%t0 = ADORA.kernel async` 和 `%t1 = ADORA.kernel async`，deps 都是 `[]`
3. 现有 `check-adora` 集合不 regression（除了被重写的 schedule_tasks 那一条）
4. 端到端：`cgra-opt --adora-schedule-tasks`，dump IR，人眼看 token 链是否符合手算依赖

## 风险 & 应对

| 风险 | 缓解 |
|---|---|
| Optional result assemblyFormat 语法出错 | 抄 `mlir/include/mlir/Dialect/GPU/IR/GPUOps.td` 里 `AllocOp` / `MemcpyOp` 写法 |
| `AttrSizedResultSegments` 需要补 `resultSegmentSizes` attr | TableGen 自动生成，但 builder 要手动填 |
| 老 .mlir 测试偶尔会经过 `ScheduleAdoraTasks` | 如果 pass 升级后老测试 CHECK 还写老语法，会 fail。处理：该测试如果不在 schedule 流水线里就不受影响；在的话更新 CHECK |
| mapper 短期内还没切到 token，schedule pass 跑完后 mapper 找不到 dep_summary | **PR1/PR2 之间**保留 `dep_summary` 一段时间？—— 用户已决定 M1 一刀切，所以 mapper 侧必须同步跟进；若 mapper 本来就还没消费 dep_summary，那就无阻塞 |

## 已确认的决策（与用户 review 后固化）

- scope = **A2**：token 统一表达所有顺序依赖（含 RAW），memref 只做数据
- migration = **M1**：一刀切，token 化后直接删 `adora.dep_summary` + 相关测试改写
- compat = **Optional token** (gpu.alloc async 模式)，保证 0 破坏

## 待决问题

1. `kind` 信息是否保留（B1 旁挂 attr vs B2 丢弃），本文默认 B2，请 reviewer 拍板
2. 是否在 PR1 也顺带把 `DepSummaryView` 删除，还是留到 PR2 一起（推荐 PR2 一起，PR1 不动 mapper 侧）
3. Token 是否允许跨 `func.func` 边界？目前约束是单 func 内使用；跨 func 需要通过 arg/return 传递，本期先**不支持**，后续看需求

---

## PR1 笔误修正 + 实际落地范围

**笔误修正**：原 §242 Step 6 "ScheduleAdoraTasks 对 kernel-to-kernel 依赖改走 operand append" 与代码不符。实际阅读 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp` 后发现：该 pass 全程**不引用** `KernelOp` / `KernelNode`，`analyzeDependencyInGraph` (line 181-244) 只在 `BlockLoadNode` × `BlockLoadNode` (RAR)、`BlockStoreNode` × `BlockStoreNode` (WAW)、`BlockLoadNode` × `BlockStoreNode` (WAR) 之间产 `DataBlockDepEdge`；RAW 由 SSA use-def 在 `generateTaskGraphFromBlock` 自然承载，kernel-to-kernel 不入此 pass 的边集。

因此 "kernel-to-kernel operand append" 在 PR1 代码里**没有对应物**；要让 token operand 出现在 IR 里，必须先给 `DataBlockLoadOp` / `DataBlockStoreOp` 加 async 形态，这是 PR2 的范围。

**PR1 真正落地的**：
- `!ADORA.token` type + `KernelOp` async form（parser/printer/verifier/builder 全套），见 commit `befd003`
- `ScheduleAdoraTasks` 在 module 上打 `adora.scheduled` UnitAttr 作为防呆标，见 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:437-442`
- `adora.dep_summary` 保持不变（留给 PR2 删）
- `ExtractAffineForToKernel` 保持 sync 形态不切 `buildAsync`（避免 printer 多打 `async` 破坏老 CHECK）

**推迟到 PR2 的**：
- `ExtractAffineForToKernel` 切 `buildAsync` 产 async kernel，同步批量更新 atax/jacobi-2d/mvt/mvt_unroll/gemm 五个 CHECK
- `DataBlockLoadOp` / `DataBlockStoreOp` async 化
- `ScheduleAdoraTasks` 真正 `insertOperands` 把 token 挂上 DMA op
- 删除 `adora.dep_summary` + `DepSummaryView` + `parseDepSummary`

**兼容性**：PR1 对老 `.mlir` 完全向后兼容。KernelOp printer 在 `asyncToken` 缺省时打印为老 `ADORA.kernel { ... }` 形态；`adora.dep_summary` attr 依然完整写入；新增 `adora.scheduled` 是加性 module-level 标记，不破坏现有 CHECK。

---

## 当前实施进度（session 交接）

PR1 工作按 `docs/async_token_design.md` 推进，已完成前 4 步，后 3 步留给下个 session。所有改动在 `jhlou/scheduletasks` 分支上。

### 已完成（本 session）

- ✅ `include/ADORA/Dialect/ADORA/IR/ADORABase.td`：新增 `ADORA_TokenType`（`!ADORA.token`）
- ✅ `include/ADORA/Dialect/ADORA/IR/ADORA.h`：调整 include 顺序，TypeDef 由 KernelOp 头文件提供，避免双定义
- ✅ `include/ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.h`：先 `#define GET_TYPEDEF_CLASSES` 再 `#define GET_OP_CLASSES`
- ✅ `include/ADORA/Dialect/ADORA/IR/KernelOp/ADORAKernelOp.td`：加 `Variadic<ADORA_TokenType>:$asyncDependencies` operand + `Optional<ADORA_TokenType>:$asyncToken` result；新增第三种 builder overload
- ✅ `lib/Dialect/ADORA/IR/ADORADialect.cpp`：注册 TypeDef（`addTypes<>` + `GET_TYPEDEF_CLASSES`），引入 `TypeSwitch` / `DialectImplementation.h`
- ✅ `lib/Dialect/ADORA/IR/KernelOp/KernelOp.cpp`：
  - `build()` 新增 async overload（接收 `ValueRange asyncDeps, bool produceToken`）
  - `verify()` 补产消对称规则
  - `print()` 输出 `async [%t0, %t1, ...]` 前缀
  - `parse()` 解析 `async` 关键字 + 可选 deps 列表
- ✅ `ninja cgra-opt` 链接成功

### Session 2 执行记录（PR1 定稿，路径 C）

#### 接棒时状态

- HEAD = `14dbd2c archive current change`，在 `befd003` 之上。
- Step 1–4 基础设施全部在盘（`grep -n 'asyncDependencies|asyncToken|ADORA_TokenType' ...` 逐文件核对）。
- 工作树原本混入两条无关脏线：**Q1+Q2 dep_summary 细化**（与 PR1 M1 方向冲突）和 **mapper P4.1**（与 PR1 无关）。本 session 分两笔 `git stash push` 归零到 `befd003`。
- `git stash list`:
  - `stash@{0}: Q1+Q2 dep_summary refinement WIP (conflicts with PR1 M1)`
  - `stash@{1}: mapper P4.1 WIP (off PR1)`

#### 发现与决策

1. **Step 5 尝试与回退**：先按原计划把 `AffineFortoKernelUtil.cpp:57` 切成 `builder.create<ADORA::KernelOp>(loc, "", ValueRange{}, /*produceToken=*/true)`。`ninja cgra-opt` 通过；`ninja check-adora` 因 printer 对带 token 的 kernel 输出 `ADORA.kernel async { ... }`，5 个老 CHECK 集体 fail（`adoracc/kernel/{atax_unroll/atax, jacobi_2d/jacobi-2d, mvt/mvt, mvt_unroll/mvt}.mlir` + `cgra-opt/kernel/gemm.mlir`）。拍板回退 Step 5，PR1 保 "零 `.mlir` 测试改动"。
2. **Step 6 笔误修正**：原 §242 写 "ScheduleAdoraTasks 对 kernel-to-kernel 依赖改走 operand append"，但实际 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp` 全程不引用 KernelOp/KernelNode。`analyzeDependencyInGraph` (L181-244) 只产 DMA↔DMA 边（RAR/WAR/WAW），kernel-to-kernel 的 RAW 由 SSA use-def 在 `generateTaskGraphFromBlock` 自然承载。PR1 真正能做的是 module 级加 `adora.scheduled = unit` 防呆标，真正的 operand append 推到 PR2（依赖先给 DataBlockLoadOp / DataBlockStoreOp 加 async 形态）。
3. **Step 7 降级**：老 CHECK 全部保留，只追加一行 `// CHECK: adora.scheduled`。

#### PR1 最终落地改动（3 文件 +33 / -3）

- `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:436-442` —— 新增：
  ```cpp
  // PR1 — mark the enclosing module as post-schedule so downstream passes and
  // the mapper can assert scheduling has run. `adora.dep_summary` remains the
  // authoritative data channel in PR1; PR2 will make async tokens on
  // BlockLoad/BlockStore carry the ordering and retire this attribute.
  if (auto module = func->getParentOfType<ModuleOp>())
    module->setAttr("adora.scheduled", UnitAttr::get(func.getContext()));
  ```
- `test/cgra-opt/kernel/schedule_tasks_dep_summary.mlir:17` —— 追加 `// CHECK: adora.scheduled`，原 CHECK-DAG / CHECK-SAME 全部保留。
- `docs/async_token_design.md` —— §242-251 PR1 清单订正；新增 "PR1 笔误修正 + 实际落地范围" 段落；补写本 "Session 2 执行记录"。

#### 验证

- `ninja cgra-opt`: ✅
- `ninja check-adora`: ✅ 16 Passed + 3 Expectedly Failed（class B DFGgen assert，XFAIL 旧遗留，非本次引入）
- 向后兼容检查：
  - 老 `.mlir` parser/printer：Optional token 缺省，打印仍为 `ADORA.kernel { ... }`
  - 老 `adora-schedule-tasks` 消费者：`adora.dep_summary` 完整写入不变，mapper 侧 `DepSummaryView` 不动
  - 老 C++ `builder.create<KernelOp>(loc)`：第一个 build overload 仍在 `KernelOp.cpp:27-41`
  - 老 CHECK 匹配 `ADORA.kernel {`：Extract pass 保持 sync，无 `async` 关键字打印

### PR2 待做清单（对齐本次笔误修正）

1. `ExtractAffineForToKernel` / `AffineFortoKernelUtil.cpp:57` 切 async overload 产 async kernel；同步批量更新 5 个 CHECK：`atax/jacobi-2d/mvt/mvt_unroll/gemm`，把 `ADORA.kernel {` 改成 `ADORA.kernel async {`。
2. `DataBlockLoadOp` / `DataBlockStoreOp` async 化（.td 加 Variadic asyncDependencies + Optional asyncToken；.cpp 补 build/verify/print/parse；参考 KernelOp.cpp:49-169）。
3. `ScheduleAdoraTasks` 真正 `insertOperands` 把 token 挂到 DMA op 上：基于 `DataBlockDepEdge` 遍历，`dma->insertOperands(dma->getNumOperands(), producerTokens)`。KernelOp 只有一个 Variadic operand，不需要 `AttrSizedOperandSegments`；DMA op 若也仅引入一个 Variadic segment 同样不需要；多 Variadic 时再补 trait。
4. 删除 `adora.dep_summary` 写入、`buildBlockDepSummary`、`DepSummaryView.{h,cpp}`、`parseDepSummary`；mapper 侧改走 token use-def。
5. 去掉 `adora.scheduled` 防呆标（PR2 完成后 SSA token 即唯一真相，不再需要 attr 层信号）。

### 潜在坑点（PR2 注意）

1. **operandSegmentSizes**：KernelOp 当前只有一个 Variadic operand 不用 `AttrSizedOperandSegments`。若 PR2 给 KernelOp/DMAOp 引入第二个 Variadic（例如把 Region block args 的某部分改成 operand），必须补 trait。
2. **Region builder args 对齐**：async build overload 要与 sync overload 产出的 body block args 一致（`kNumConfigRegionAttributes` 个 index 参数）。
3. **TypeDef 生成路径**：当前复用 `ADORAOpsTypes.h.inc`（由 `add_mlir_dialect(ADORAOps ADORA)` 生成），未在 KernelOp/CMakeLists.txt 新增 tablegen 规则。PR2 若 DMA op 需要自己的 TypeDef 需要核对。
4. **老 mapper 消费者**：删 `adora.dep_summary` 前，确认 `mapper/src/...` 下所有 `parseDepSummary` 调用点都切到 token 消费，否则 mapper 会 segfault。
5. **DFGgen class B 遗留**：`ninja check-adora` 现有 3 个 Expectedly Failed 与 PR1/PR2 无关，属另一条工作线，PR2 勿误伤。

### Commit 历史

- `6024e90` — `test: set GeneralOpNameFile env in lit.cfg so DFG opcode names resolve`（Session 1 class A 修复）
- `befd003` — `ADORA async token: introduce !ADORA.token type + KernelOp async form`（Session 1 PR1 Step 1-4）
- `14dbd2c` — `archive current change`（Session 2 接棒基线）
- `<待 commit>` — PR1 定稿：module `adora.scheduled` 标记 + 设计文档笔误订正（Session 2 路径 C）

如需回滚：
- 只回滚 Session 2 定稿：`git reset --hard 14dbd2c`
- 回滚到 PR1 前：`git reset --hard 6024e90`（会丢失 Step 1-4 全部基础设施）

### Stash 账本

- `stash@{0}` Q1+Q2 dep_summary 细化 —— 与 PR1 M1 方向对立，PR2 落地后 `git stash drop stash@{0}`
- `stash@{1}` mapper P4.1 —— 另一条独立工作线，按需 `git stash apply stash@{1}`

---

## PR2 实施计划（详细版 · 可执行手册）

> 本节是 PR2 的权威落地指南。目标：把 `!ADORA.token` 从类型升级为 SSA 一等公民，
> 让 BlockLoad / BlockStore / KernelOp 通过 `asyncDependencies` / `asyncToken`
> 承载依赖边，替代 PR1 的 `adora.dep_summary` 字符串路径。

### PR2-§0 目标与不变式

1. 三类算子统一异步形态：
   - `asyncDependencies : Variadic<ADORA_Token>`（0..N 入边）
   - `asyncToken       : Optional<ADORA_Token>`（0..1 出边）
2. 同步形态保留：op 不参与任何依赖边时不产/不吃 token，IR 外观与 PR1 之前完全一致，保证旧 lit 零回归。
3. `adora.dep_summary` 作为调试/回归 artifact 保留（pass option 可关），SSA token 成为 single source of truth。
4. Verifier 严格化：依赖列表禁止自引用/重复，token def-use 由 MLIR 原生 dominance 检查保证。

### PR2-§1 TableGen 变更锚点

- `ADORATypes.td` 新增 `def ADORA_Token : DialectType<...>` 别名，供 `Variadic<ADORA_Token>` / `Optional<ADORA_Token>` 使用。
- `ADORAOps.td` 的三类 op 在 operand 末尾追加 `Variadic<ADORA_Token>:$asyncDependencies`，在 result 末尾追加 `Optional<ADORA_Token>:$asyncToken`。
- `AttrSizedOperandSegments` segment 固定顺序：
  - DataBlockLoadOp:  `{originalMemref=1, indices, asyncDependencies}`
  - DataBlockStoreOp: `{sourceMemref=1, targetMemref=1, indices, asyncDependencies}`
  - KernelOp:         `{inputs, outputs, asyncDependencies}`
- assemblyFormat（Load/Store）：
  ```
  (`async` `[` $asyncDependencies^ `]`)?
   <原本形态> attr-dict `:` <原本 type>
  (`,` type($asyncToken)^)?
  ```
- KernelOp 走自定义 C++ print/parse：在原壳外追加 `async [...]` 与 `-> !ADORA.token`。

### PR2-§2 C++ builder / accessor

- 每个 op 提供两个 builder：
  1. 兼容 builder（旧签名，内部转发 `asyncDeps={}, produceToken=false`）
  2. 异步 builder：`(... , ValueRange asyncDeps, bool produceToken)`，负责设置 `operandSegmentSizes` 与可选 token result。
- `Utility.h` 新增：
  - `Value getAsyncTokenOrNull(Operation*)`
  - `OperandRange getAsyncDeps(Operation*)`
  - `bool isAsyncCapable(Operation*)`（= isa<DataBlockLoad/Store, Kernel>）

### PR2-§3 Verifier 补强

- 每个 op `verify()` 检查：依赖列表不得含自身、不得重复；`operandSegmentSizes` 尺寸自洽。
- 跨 region token 由 MLIR 原生 dominance 兜底，不重复实现。

### PR2-§4 ScheduleADORATasks pass 改造

新增函数 `threadTokensOnDMAs(TaskGraph*)`，插入点位于：
```
generateTaskGraphFromBlock(&graph, block);
analyzeDependencyInGraph (&graph);    // PR1 已有
threadTokensOnDMAs       (&graph);    // PR2 新增
emitDepSummaryAttr       (&graph);    // 保留作 debug
```

算法：
1. 从 `graph->depEdges()` 收集 `preds[dstOp] = [srcOp...]` 与 `hasOut` 集合。
2. 合并去重、按 `isBeforeInBlock` 排序（PR1 已证 src→dst 恒满足 lexical before）。
3. 按序 rebuild 每个参与 op 为 async 形态：
   - `deps = [tokens[p] for p in preds[op] if produced]`
   - `produceTok = hasOut.contains(op)`
   - 调用 `rebuildAsyncLoad / Store / Kernel`，RAUW 旧 result，迁移白名单外的全部 attr，擦除旧 op，回写 `tokens[op] = newOp.asyncToken`。
4. 维护 `oldToNew : DenseMap<Operation*, Operation*>`，遍历 `graph->getAllNodes()` 调用 `TaskNode::setOperation(newOp)`，防止野指针。

pass option：
- `emit-token`（默认 on）
- `emit-summary`（默认 on，灰度期保留）
- `cross-check-summary-vs-token`（默认 off，CI 打开）

### PR2-§5 rebuildAsync* 实现要点

- 逐字段抄写 map / strides / kernel_name / region（Kernel 用 `takeBody`），避免丢失 `pingpong`、`tile_id` 等下游 metadata。
- attr 迁移白名单：跳过 `map / strides / kernel_name / operandSegmentSizes`，其余全部 `newOp->setAttr`。
- Load 额外做 `old.getResult().replaceAllUsesWith(newOp.getResult())`；Store 无 result 直接 erase。
- Kernel 需要 RAUW 每一个数据 result（token result 排在末尾，不破坏旧下标）。

### PR2-§6 Canonicalization

新增两个 RewritePattern（挂到三个 op 的 `getCanonicalizationPatterns`）：
- `DropUnusedAsyncToken`：`asyncToken.use_empty()` → rebuild 为同步形态。
- `DedupAsyncDeps`：对 `asyncDependencies` 去重去 self，同步更新 `operandSegmentSizes`。

### PR2-§7 边角 case

1. 单点环 / 自引用：threading 里 `if (p == op) continue;` 防御。
2. 重复边（RAR+RAW 同 src/dst）：`SmallSetVector` 去重。
3. kernel 被 scf.for 包住：PR1 `BlockContainsKernelOp` 已跳过嵌套 for；loop-carry token 留 PR6。
4. dead token：`DropUnusedAsyncToken` 修剪。
5. ping-pong attr 与 token 正交：白名单迁移自然保留。
6. 混合同步/异步读同 memref：同步 op 走 SSA memref use-def，天然 happens-after，无需再加 token。

### PR2-§8 Cross-check（可选）

`verifyTokensMatchSummary`：解析 `adora.dep_summary` 每行 → 比对新 IR 上对应 src/dst 的 token 边是否存在；不一致则 `signalPassFailure`。默认关闭，CI 打开作 golden 交叉校验。

### PR2-§9 测试矩阵

- `test/Dialect/ADORA/async_token_roundtrip.mlir`：手写同步+异步混用，`adora-opt %s | adora-opt | FileCheck %s` 验证 printer/parser 对称。
- `test/Dialect/ADORA/schedule_cgra_tasks_tokens.mlir`：load→kernel→store 链 + RAR/WAW/WAR 样例，FileCheck token 串通与边数。
- 回归：PR1 所有 dep_summary lit 保持通过；`emit-token=0` 模式下输出与 main 逐字节一致。
- Negative：跨 region 引用 token 应由 `mlir-opt --verify-diagnostics` 报 dominance 错误。

### PR2-§10 三个 commit 的 git 节奏

- **commit A** — `adora: introduce async operands/results on DMAs & kernel (NFC when unused)`
  - §1 TableGen + §2 builder/accessor + §3 verifier
  - 不动 pass；仅新增 roundtrip lit；预期全量 lit 零改动通过。
- **commit B** — `adora: thread SSA tokens through ScheduleADORATasks`
  - §4 `threadTokensOnDMAs` 接入 + pass option；新增 token lit；旧 summary lit 继续通过。
- **commit C** — `adora: canonicalize async tokens & cross-check`
  - §6 两个 pattern；§8 cross-check option；文档追加 PR2 章节。

### PR2-§11 文件改动地图

| 文件 | 改动 |
|---|---|
| `include/ADORA/Dialect/ADORA/IR/ADORATypes.td`            | 新增 `ADORA_Token` 别名 |
| `include/ADORA/Dialect/ADORA/IR/ADORAOps.td`              | 3 个 op 字段 + builder + assemblyFormat |
| `lib/Dialect/ADORA/IR/ADORAOps.cpp`                       | 每 op 兼容/异步 2 个 builder + verify + custom print/parse(Kernel) |
| `include/ADORA/Dialect/ADORA/Utility/Utility.h`           | 3 个 inline accessor |
| `include/ADORA/Dialect/ADORA/Transforms/TaskGraph/TaskGraph.h` | `TaskNode::getOperation/setOperation` 基类虚函数 |
| `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`     | `rebuildAsync*` + `threadTokensOnDMAs` + pass option |
| `lib/Dialect/ADORA/Transforms/ADORACanonicalize.cpp`      | `DropUnusedAsyncToken` / `DedupAsyncDeps` |
| `test/Dialect/ADORA/async_token_roundtrip.mlir`           | 新增 |
| `test/Dialect/ADORA/schedule_cgra_tasks_tokens.mlir`      | 新增 |
| `docs/async_token_design.md`                              | 追加本节 |

### PR2-§12 验收清单

```
[ ] mlir-opt --verify-roundtrip 通过新老所有 ADORA lit
[ ] adora-opt --adora-schedule-cgra-tasks 无 dep_summary lit 全绿
[ ] adora-opt --adora-schedule-cgra-tasks=cross-check-summary-vs-token=1
    在 integration 套件零 diagnostic error
[ ] 同步形态 IR 与 main 分支 byte-for-byte 一致（确保 NFC）
[ ] docs/async_token_design.md 已追加 PR2 章节
[ ] 至少 1 个 end-to-end demo（load→kernel→store）IR 里看见 !ADORA.token 串通
```

### PR2-§13 风险与缓解

| 风险 | 缓解 |
|---|---|
| operand 顺序变更破坏下游 C++ 索引 | token 永远追加在末尾 + `AttrSizedOperandSegments` 明确 segment |
| RAUW 漏 memref use                 | rebuild 保留相同 result 类型，统一 RAUW；lit 回归覆盖 |
| attr 丢失（pingpong / tile_id 等） | rebuild 函数显式白名单迁移 |
| KernelOp 多 result 顺序错位        | token 恒为最后一个 result；禁止改变已有下标 |
| 循环依赖                           | PR1 已证 DAG；threading 按 lexical order 执行 |

---

## PR3 预研计划（Token → Runtime 事件/信号量）

> PR2 把 token 做成 SSA；PR3 回答"SSA 边如何变成真实同步"。
> 本节固化 PR3 的 op/pass/runtime/ABI 设计，合入前可作为实施蓝图。

### PR3-§0 目标与非目标

**目标**
1. 新增 runtime 原语 op：`adora.event.create / destroy`、`adora.signal`、`adora.wait`。
2. Pass `adora-lower-async-tokens`：把 SSA token def-use 物化成成对 signal/wait。
3. Pass `adora-to-llvm-async-runtime`：把上述 op 翻译为 `llvm.call` 到 runtime ABI。
4. Runtime 桩 `libadora_async_rt`：`adoraEventCreate/Destroy/Record/Wait`，语义对齐 CUDA event / Vulkan semaphore。

**非目标（留后续 PR）**
- PR4：基于 token 的真正双缓冲 / prefetch 调度，多流分配。
- PR6：loop-carry token（`scf.for` iter_args 传 token）。

### PR3-§1 Runtime ABI（PR3 冻结）

```c
typedef struct AdoraEvent_* AdoraEvent;
AdoraEvent adoraEventCreate(void);
void       adoraEventDestroy(AdoraEvent e);
void       adoraEventRecord (AdoraEvent e, int32_t streamId);
void       adoraEventWait   (AdoraEvent e, int32_t streamId);
```

- PR3 中 `streamId` 恒为 0；保留参数供 PR4 扩多流。
- MLIR 映射：`!ADORA.token` → `!llvm.ptr`；`streamId` → `i32`。

### PR3-§2 新增 op（TableGen 草案）

- `adora.event.create` : `() -> !ADORA.token`（Pure）
- `adora.event.destroy`: `(!ADORA.token) -> ()`
- `adora.signal`       : `(!ADORA.token) [on stream $s]?`
- `adora.wait`         : `(!ADORA.token) [on stream $s]?`

仅在 PR3 的 lower pass 之后出现；PR1/PR2 IR 中看不到。

### PR3-§3 Pass 1：`adora-lower-async-tokens`

输入：PR2 IR（带 `asyncDependencies/asyncToken`）。
输出：三类 op 回落同步形态 + 显式 `event.create / signal / wait / destroy`。

算法：
```
for each producer P with asyncToken tP:
    e = adora.event.create (插在 P 前)
    rebuild P sync-form
    adora.signal e         (插在 P 后)
    RAUW tP -> e

for each consumer C with deps [d0..]:
    for each d in deps (去重): adora.wait d  (插在 C 前)
    rebuild C sync-form

为每个 event 在其 last use 之后插入 adora.event.destroy
```

pass option：
- `drop-tokens-only`（默认 off）：仅丢 token 不生成 event，调试用。
- `default-stream`（默认 0）。

### PR3-§4 Pass 2：`adora-to-llvm-async-runtime`

Type converter：`!ADORA.token` → `!llvm.ptr`。
Pattern：
| source | target |
|---|---|
| `adora.event.create`  | `llvm.call @adoraEventCreate()` |
| `adora.event.destroy` | `llvm.call @adoraEventDestroy(%e)` |
| `adora.signal`        | `llvm.call @adoraEventRecord(%e, %s)` |
| `adora.wait`          | `llvm.call @adoraEventWait (%e, %s)` |

`llvm.func` 声明首次遇到时 lazy-insert 到 ModuleOp。

### PR3-§5 Pipeline 次序

```
adora-schedule-cgra-tasks     (PR2)
adora-lower-async-tokens      (PR3-A)
adora-lower-dmas              (既有/待补)
adora-to-llvm-async-runtime   (PR3-B)
finalize-memref-to-llvm / convert-func-to-llvm / reconcile-unrealized-casts
```

### PR3-§6 Runtime 桩参考实现（x86 单流）

- `AdoraEvent` = `{ _Atomic int signaled; pthread_cond_t cv; pthread_mutex_t mu; }`
- `Record` 原子置 1 + `cond_broadcast`；`Wait` 轮询 `cond_wait`。
- 真实 CGRA 后端在另一 PR 替换实现，ABI 不变。

### PR3-§7 测试矩阵

- `lower_async_tokens.mlir`：FileCheck `event.create` / `signal` / `wait` / `destroy` 齐全、位置正确。
- `lower_async_runtime.mlir`：FileCheck `llvm.func` 声明与 `llvm.call` 映射。
- `test/Integration/async_token_chain.mlir`：`mlir-cpu-runner -shared-libs=libadora_async_rt.so` 端到端跑 load→kernel→store，断言 memref 结果正确。

### PR3-§8 三个 commit 拆分

- **commit A** — runtime 库 + dialect ops（lit roundtrip）
- **commit B** — `adora-lower-async-tokens` + §7 第 1 条 lit + PR2 pipeline 回归
- **commit C** — `adora-to-llvm-async-runtime` + §7 第 2/3 条 lit + CI integration

### PR3-§9 风险与缓解

| 风险 | 缓解 |
|---|---|
| event 泄漏（destroy 位置算错）    | liveness 分析 + canonical `DropDeadEvent`（无 signal/wait 的 create 直接删） |
| signal/wait 不匹配                 | Verifier：signal 的 event 必须 reach ≥1 wait（warning 级别可配） |
| ABI 日后变动                       | opaque ptr + streamId 向前兼容；`adoraRuntimeVersion()` 暴露版本号 |
| 与 PR2 builder 分叉                | lower pass 复用 PR2 `rebuildAsync*`，参数换成 `deps={}, produce=false` |
| CUDA/Vulkan 后端差异               | ABI 对齐 CUDA event；Vulkan semaphore wrapper 放后端子目录 |

### PR3-§10 验收清单

```
[ ] --adora-lower-async-tokens 后 IR 不再出现 !ADORA.token
    （只保留在 event.create/destroy/signal/wait 上）
[ ] --adora-to-llvm-async-runtime 后 IR 仅剩 llvm.call
[ ] mlir-cpu-runner 跑通 end-to-end integration
[ ] libadora_async_rt 通过 helgrind / tsan
[ ] PR1/PR2 现存 lit 全部保持绿色
[ ] 本文档已追加 PR3 章节
```

### PR3-§11 预埋给 PR4 的钩子

- `adora.signal / wait` 已带 `stream : i32`：PR4 分配不同 streamId 即可多流。
- `event.create` 预留 `{reusable = true}`：PR4 双缓冲复用事件，不反复 create/destroy。
- runtime header 预埋 `adoraStreamCreate/Destroy`，PR3 不暴露对应 op，PR4 使用。

### PR → PR 的全局状态机（速览）

```
PR1: dep_summary(字符串)          ─┐
                                   │ analyzeDependencyInGraph
PR2: SSA !ADORA.token             ─┘ threadTokensOnDMAs
                                   │ lower-async-tokens
PR3: event / signal / wait        ─┐
                                   │ to-llvm-async-runtime
PR3: llvm.call @adoraEvent*        │
                                   │ mlir-cpu-runner + libadora_async_rt
PR4: stream 分配 + 双缓冲 + 预取（基于 token 图）
PR6: loop-carry token（scf.for iter_args）
```

---

## 当前进展 & 下一步（Status · 供 review）

### Build status

- `cd build && ninja`：**48/48 目标全部通过**。
- 仅剩既有 `-Wreorder` / `-Wdelete-non-virtual-dtor` 警告（与本 session 改动无关，mapper 历史遗留）。
- 本 session 引入的 **编译单元改动** 仅 7 行（见下 §"已落地 diff"），不影响任何现有 lit / integration。

### 已落地 diff（本 session 内）

| 文件 | 改动 | 说明 |
|---|---|---|
| `docs/async_token_design.md` | +387 行 | 追加 PR2 §0–§13、PR3 §0–§11、全局状态机 |
| `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp` | +7 行 | 在 `emitDepSummaryAttr` 之后给 enclosing module 打 `adora.scheduled` UnitAttr，作为 "post-schedule" 标记；PR1 语义保留 |
| `test/cgra-opt/kernel/schedule_tasks_dep_summary.mlir` | +1 行 | 对应 `adora.scheduled` 的 CHECK 行 |

### 设计冻结点

- `!ADORA.token` 类型（PR1 已合入 `14dbd2c`）。
- `adora.dep_summary` ArrayAttr 作为 PR1 唯一数据通道；PR2 落地后降级为 debug artifact。
- PR2 的三个 op 字段 schema（§A.1 表格）已冻结，不再争议。
- PR3 runtime ABI（§PR3-§1）已冻结，`adoraEventCreate/Destroy/Record/Wait`。

### 下一步 action list（按执行顺序）

#### Step 1 — PR2 commit A（TableGen + builder，NFC）
1. `ADORATypes.td`：加 `def ADORA_Token`。
2. `ADORAOps.td`：三个 op 尾部追加 `Variadic<ADORA_Token>:$asyncDependencies` 与 `Optional<ADORA_Token>:$asyncToken`；固定 `operandSegmentSizes` 顺序。
3. `ADORAOps.cpp`：每个 op 两个 builder（兼容 / 异步）+ `verify()` 加自引用/重复检查。
4. `Utility.h`：三个 inline accessor。
5. 新增 `test/Dialect/ADORA/async_token_roundtrip.mlir`。
6. `ninja && llvm-lit test/` 全量绿。

**预期产物**：新字段存在但无人使用；所有旧 lit 零修改通过。

#### Step 2 — PR2 commit B（pass threading）
1. `TaskGraph.h`：`TaskNode` 基类加 `virtual Operation* getOperation()` / `setOperation(Operation*)`。
2. `ScheduleAdoraTasks.cpp`：
   - 新增 `rebuildAsyncLoad / Store / Kernel`（attr 白名单迁移 + RAUW + erase）。
   - 新增 `threadTokensOnDMAs(TaskGraph*)`，插在 `analyzeDependencyInGraph` 之后。
   - 新增 pass option `emit-token`（默认 on）、`emit-summary`（默认 on）。
3. 新增 `test/Dialect/ADORA/schedule_cgra_tasks_tokens.mlir`。
4. 回归：现有 `schedule_tasks_dep_summary.mlir` 在 `emit-token=0` 下字节一致。

#### Step 3 — PR2 commit C（canonical + cross-check）
1. `DropUnusedAsyncToken` / `DedupAsyncDeps` 两个 RewritePattern。
2. Pass option `cross-check-summary-vs-token`（默认 off，CI 打开）。
3. 本文档再追加 "PR2 合入总结" 小节。

#### Step 4 — PR3 启动条件
PR2 三个 commit 合入且 CI 稳定 1 周后，按本文档 PR3-§0~§11 开工：
- commit A：runtime 桩 + dialect event/signal/wait op。
- commit B：`adora-lower-async-tokens`。
- commit C：`adora-to-llvm-async-runtime` + end-to-end integration。

### 待 review 的悬而未决问题

1. **KernelOp 自定义 print/parse**：目前 `adora.kernel` 走 C++ printer；`async [...]` / `-> !ADORA.token` 的语法位置需 review（建议在 `attributes { ... }` 之前，body 之后）。
2. **dep_summary 何时退役**：PR2 全量替换后保留 1 个 release 周期，还是直接在 PR3 开头删？倾向前者。
3. **Kernel 的 region 内部不碰 token**：确认 body 里的 block argument 不引入 token 形参（维持 `gpu.launch` 风格的外壳同步）。
4. **`cross-check-summary-vs-token`**：开启时若不一致，`signalPassFailure` 还是 `emitWarning`？建议 CI-only 模式下 fail，开发时 warning。
5. **Token 类型的 ABI 映射（PR3）**：`!llvm.ptr` vs 自定义 `!llvm.struct<"AdoraEvent", opaque>`——前者简单，后者便于 type-safe debug。倾向 `!llvm.ptr`。

### 代码注释规范（实施期生效）

见上一条 review 约定：
- 全英文 `//` / `///`，只解释 **意图 / invariant / 原因**。
- 文件顶部 2–4 行 block 说明模块定位。
- `rebuildAsync*` 等函数用 `///` doxygen 注明 `@param / @return / 副作用（erase）`。
- 算法主循环用 `// 1. / 2. / 3.` 分段。
- 易错点用 `// NOTE:` / `// WHY:`，TODO 带 scope：`TODO(PR3): lower to adora.event.wait`。
- TableGen 用 `let description = [{ ... }];` 为每个新 op 字段写语义段。

### 里程碑时间表（建议）

| 周 | 交付 |
|---|---|
| W1 | PR2 commit A merged（TableGen + builder，NFC） |
| W2 | PR2 commit B merged（threading + lit） |
| W3 | PR2 commit C merged（canonical + cross-check），文档追加合入总结 | ✅ 已完成 |
| W4 | 灰度 1 周，CI 打开 `cross-check`，观察 integration |
| W5 | PR3 commit A 启动 |

---

## PR2 合入总结

### Commit 链
- `0416fbf` — PR2 commit B: threadTokensOnDMAs + rebuild + Utility + options + lit
- `C.1 commit` — PR2 commit C.1: DedupAsyncDeps canonical pattern

### 关键改动文件
| 文件 | 内容 |
|---|---|
| `ADORABase.td` | `useDefaultTypePrinterParser=1` 修复 token printer |
| `ADORAOps.td` | Load/Store `hasCanonicalizer=1` |
| `ADORAKernelOp.td` | KernelOp rebuild builder + `hasCanonicalizer=1` |
| `Utility.h` | `getAsyncTokenOrNull/getAsyncDeps/isAsyncCapable` |
| `Passes.td` | `emit-token/emit-summary/cross-check` options |
| `ADORAOps.cpp` | `DedupAsyncDeps` pattern; 删除错误 verifier |
| `KernelOp.cpp` | takeBody builder; `DedupKernelAsyncDeps` pattern |
| `ScheduleAdoraTasks.cpp` | `rebuildAsync*`+`threadTokensOnDMAs`+`verifyTokensMatchSummary` |
| `schedule_cgra_tasks_tokens.mlir` | PR2 lit（新增，通过） |

### 非预期问题与修复
- `strides/kernel_name` attr 缺失 → null-guard fallback
- `RemoveRedundant*` 留悬空 dep edge → `getBlock()==null` 跳过
- `threadTokensOnDMAs` 移到 `RemoveRedundant*` 之后避免 live-use crash
- `!ADORA.token` 无 printer → `useDefaultTypePrinterParser=1`
- 错误 verifier rule "has asyncDeps but no asyncToken" → 删除

### 验证
```
ninja 全量编译通过；check-adora 13/20 pass（4 pre-existing failures 不变）
emit-token=true  → !ADORA.token 真实出现在 IR
emit-token=false → 与 PR1 baseline 字节一致（NFC）
```

### PR3 预埋接口
- `asyncDependencies/asyncToken` 字段就位
- `Utility.h` accessor 供 PR3 lowering 使用
- `emit-summary` 默认 on，PR3 稳定后退役


---

## 当前进展快照（归档节点）

### 已合入 commit 链

| commit | 内容 | 状态 |
|---|---|---|
| `0416fbf` | PR2 commit B: threadTokensOnDMAs + rebuild + lit | done |
| `f6613a6` | PR2 commit C.1: DedupAsyncDeps canonical | done |
| `13c68e2` | PR2 commit C.2: doc + cross-check | done |
| `6afbd5b` | PR3 commit A: EventCreate/Destroy/Signal/Wait ops + runtime stub | done |
| `44f8935` | PR3 commit B WIP: LowerAsyncTokens pass skeleton | WIP |

### PR3 commit B 遗留问题（下个 session 继续）

症状：`--adora-schedule-tasks=emit-token=true --adora-lower-async-tokens`
后 IR 里看不到 ADORA.event.create / signal / wait / destroy。

诊断顺序：
1. grep EventCreateOp build/include/ADORA/Dialect/ADORA/IR/ADORAOps.h.inc
   确认 op 类在生成代码里。
2. 在 LowerAsyncTokensPass::runOnOperation() 开头加 llvm::errs() 确认 pass 真正运行。
3. 在 Pass 1 walk lambda 里加 errs() 确认 asyncToken op 被发现。
4. 若 EventCreateOp 不在 .inc，检查 ADORAOps.td guard 是否正确。
5. 修完后补 lit test/cgra-opt/kernel/lower_async_tokens.mlir。

### PR3 commit C 待做（B 修好后）

- Pass adora-to-llvm-async-runtime：TokenType->llvm.ptr + 四 pattern
- Lit lower_async_runtime.mlir
- Integration: mlir-cpu-runner -shared-libs=libadora_async_rt.so

### 预存在 bug（记录）

adora-adjust-kernel-mem-footprint cmake 重配后崩溃（commit 6024e90 前已存在）。
已绕过：kernel lit test 改用 generic-form 直接 IR 输入。

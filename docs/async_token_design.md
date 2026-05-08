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
- [ ] `ExtractAffineForToKernel` / `AffineFortoKernelUtil.cpp` 切到 `buildAsync`（pre-schedule 阶段自然产空 deps）
- [ ] `ScheduleAdoraTasks` 对 kernel-to-kernel 依赖改走 operand append 路径（不再 replace op）
- [ ] `schedule_tasks_dep_summary.mlir` 测试改为 `schedule_tasks_token.mlir`，CHECK 形如 `%t1 = ADORA.kernel async [%t0]`
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

### 下一步（下个 session 接棒）

#### Step 5 — AffineFortoKernelUtil / ExtractAffineForToKernel 切 buildAsync

**目标**：让 `ExtractAffineForToKernel` pass 创建的 kernel 直接带 asyncToken result，本地链暂时空（该 pass 此时没有 BlockLoad 配合）。

**关键文件**：
- `lib/Dialect/ADORA/Transforms/Kernel/AffineFortoKernelUtil.cpp`：找到 `builder.create<KernelOp>(loc, kernelName)` 的调用点，改成新 async builder overload：`builder.create<KernelOp>(loc, kernelName, /*deps=*/ValueRange{}, /*produceToken=*/true)`
- 检查 `ExtractAffineForToKernel.cpp`（如果存在）内有无直接构造 KernelOp 的地方，一并切换

**验证**：`cgra-opt --adora-extract-affine-for-to-kernel <simple_input.mlir>` 观察输出，kernel 应为 `%t = ADORA.kernel async { ... }` 形态。

#### Step 6 — ScheduleAdoraTasks 改 operand append

**目标**：pass 不再写 `adora.dep_summary` attr，也不 op replace；改为：
1. 原有 task graph 依赖分析保留
2. 遍历每个 TaskNode 对应的 kernel op，根据 `DataBlockDepEdge` 找前驱 kernel 的 `getAsyncToken()`
3. 过滤掉 "本地链已存在的 RAW"（PR1 阶段 kernel 之间只会有跨迭代的依赖，不存在本地链冲突，先简单全 append）
4. 调用 `op->insertOperands(op->getNumOperands(), tokens)`
5. **关键**：因为 `asyncDependencies` 是 `Variadic`，MLIR 会自动根据 operandSegmentSizes attr 重新分段——确认 KernelOp 的 `arguments` 只有 `$asyncDependencies` 一个 variadic，就不需要 `AttrSizedOperandSegments` trait；若有多个 variadic 后续再补
6. 删 `buildBlockDepSummary` 调用和 `setAttr("adora.dep_summary", ...)`
7. 结束时：`getOperation()->getParentOfType<ModuleOp>()->setAttr("adora.scheduled", UnitAttr::get(context))`

**关键文件**：
- `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:405-411`（现在 setAttr 的位置）
- `lib/Dialect/ADORA/Transforms/TaskGraph/TaskNode.h`（看 TaskNode 里怎么拿 op 指针——可能叫 `getOp()` 或 `op()`）

**验证**：用 `schedule_tasks_dep_summary.mlir` 的输入手跑，输出应该看到 `%t1 = ADORA.kernel async [%t0] { ... }`。

#### Step 7 — 测试改写 + check-adora

**目标**：把 `test/cgra-opt/kernel/schedule_tasks_dep_summary.mlir` 重命名为 `schedule_tasks_token.mlir`（或保留名字），CHECK 内容从 `adora.dep_summary = [...]` 改为 token SSA 边。

**关键文件**：
- `test/cgra-opt/kernel/schedule_tasks_dep_summary.mlir:20-27`（原 CHECK 块）
- 新 CHECK 形如：
  ```
  // CHECK: %[[T0:.+]] = ADORA.kernel async { ... }
  // CHECK: %[[T1:.+]] = ADORA.kernel async [%[[T0]]] { ... }
  ```

**验证**：
- `ninja check-adora` 通过（或只有已知遗留 failure：类 B 的两个 DFGgen assert abort、`DepSummaryView` 目前只在 PR2 删）
- Class A 5 个已修 + PR1 不应引入新 regression

### 潜在坑点（下个 session 注意）

1. **operandSegmentSizes**：KernelOp 目前只有一个 Variadic operand（`$asyncDependencies`），MLIR 默认不会加 `AttrSizedOperandSegments` trait；若出错检查 `ADORAKernelOp.td` 是否需要补 `[AttrSizedOperandSegments]` trait。
2. **Region builder**：新 async builder 中手动 `addRegion` + `addArgument`——必须保证 body block args 与 sync builder 一致（目前用 `kNumConfigRegionAttributes` 个 index 参数）。
3. **`include/ADORA/Dialect/ADORA/IR/KernelOp/CMakeLists.txt`** 是否需要额外 tablegen 规则生成 TypeDef——目前复用 `ADORAOpsTypes.h.inc`（由主 `add_mlir_dialect(ADORAOps ADORA)` 生成）应够用。
4. **TaskNode → op 指针映射**：`ScheduleAdoraTasks.cpp` 中 `TaskNode` 和实际 `KernelOp` / `BlockLoadOp` 的关联需要确认；PR1 只处理 kernel 相关节点，BlockLoad/Store 节点在 PR2 再接。

### Commit 历史

- `6024e90` - `test: set GeneralOpNameFile env in lit.cfg so DFG opcode names resolve`（类 A 修复）
- `<PR1 进度 commit>` - 本 session 落下去的 Token 基础设施 + KernelOp 改造

如需回滚 PR1 进度：`git reset --hard 6024e90`（但会丢失 KernelOp 的 async 改造）。

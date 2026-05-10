# `--adora-schedule-tasks` Pass 使用说明

## 概述

`adora-schedule-tasks` 是 ADORA 编译器的任务调度核心 pass，负责：

1. 在 `KernelOp` 之前构建 **Task Graph**（BlockLoad / Kernel / BlockStore / LocalMemAlloc 节点）
2. 分析 **数据块级别依赖**（RAW / WAR / WAW / RAR）
3. **消除冗余** BlockLoad（buffer reuse）
4. 把依赖转成 SSA **`!ADORA.token` async chain**
5. 发射 **`adora.dep_summary`** dict attr 供 mapper 消费
6. 检测 **loop-carried dep**（PR6 尚未完全实现，目前只报诊断）

这个 pass 是整条 async pipeline 的入口，后面跟着 `adora-buffer-reuse`（已冗余）、`adora-assign-streams`、`adora-lower-async-tokens`、`adora-to-llvm-async-runtime`。

---

## 基本用法

```bash
cgra-opt input.mlir --adora-schedule-tasks
cgra-opt input.mlir --adora-schedule-tasks="emit-token=true"
cgra-opt input.mlir --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/g.dot"
```

---

## Pass Options 完整列表

| Option | Type | Default | 作用 |
|--------|------|---------|------|
| `emit-token` | bool | `false` | 将依赖转成 SSA `!ADORA.token` async chain。**推荐开启** |
| `emit-summary` | bool | `true` | 在 func 上发射 `adora.dep_summary` DictionaryAttr（PR1 数据通道） |
| `cross-check-summary-vs-token` | bool | `false` | CI-only：断言 token edges 与 dep_summary 一致，不一致则 pass 失败 |
| `dump-token-graph` | string | `""` | 非空时把 async token chain 写入 Graphviz DOT 文件。要求 `emit-token=true` |

---

## 输出效果

### 输入

```mlir
%a = ADORA.BlockLoad %arg0 [0, 0]
    : memref<32x32xf32> -> memref<32x32xf32>
    {Id = "0", KernelName = "k0"}
%b = ADORA.BlockLoad %arg1 [0, 0]
    : memref<32x32xf32> -> memref<32x32xf32>
    {Id = "1", KernelName = "k0"}
%local = ADORA.LocalMemAlloc memref<32x32xf32>
    {Id = "2", KernelName = "k0"}
ADORA.kernel {
  ADORA.terminator
} {KernelName = "k0"}
ADORA.BlockStore %local, %arg2 [0, 0]
    : memref<32x32xf32> -> memref<32x32xf32>
    {Id = "2", KernelName = "k0"}
```

### `emit-token=false`（default）

只加 `adora.dep_summary` attr，IR 结构不变。

### `emit-token=true`

完整 async token chain：

```mlir
%a, %tok0 = ADORA.BlockLoad %arg0 [0, 0]  -> !ADORA.token
%b, %tok1 = ADORA.BlockLoad %arg1 [0, 0]  -> !ADORA.token
%local    = ADORA.LocalMemAlloc
%tokK     = ADORA.kernel async [%tok0, %tok1] { ... }
ADORA.BlockStore async [%tokK] %local, %arg2 [0, 0]
```

---

## 完整 Pipeline

```bash
cgra-opt input.mlir \
  --adora-schedule-tasks="emit-token=true" \
  --adora-assign-streams \
  --adora-lower-async-tokens \
  --adora-to-llvm-async-runtime \
  -o output.mlir
```

**各 pass 职责**：

| Pass | 作用 | 状态 |
|------|------|------|
| `adora-schedule-tasks` | 构建 task graph + 依赖分析 + token 链接 + buffer reuse | ✅ 必需 |
| `adora-buffer-reuse` | 消除冗余 BlockStore/BlockLoad 对 | ⚠️ **冗余**（见下方说明） |
| `adora-assign-streams` | 给每个 async op 分配硬件 stream ID（拓扑排序） | ✅ 必需 |
| `adora-lower-async-tokens` | `!ADORA.token` → `ADORA.event.create/signal/wait/destroy` | ✅ 必需 |
| `adora-to-llvm-async-runtime` | `ADORA.event.*` → `llvm.call @adoraEvent*` runtime ABI | ✅ 必需 |

### ⚠️ `adora-buffer-reuse` 冗余说明

从 **commit `88d0e75`（任务一）** 开始，`RemoveRedundantBlockStoreLoadPair`
已经在 `adora-schedule-tasks` **token threading 之前**就完成了冗余 Load 的
`replaceAllUsesWith` + erase。

因此：
- **运行 `adora-buffer-reuse` 前后 IR 完全相同**（实测 3mm 例子 BlockLoad 数量一致）
- **它现在是 no-op**，保留只是为了兼容可能存在的旧 pipeline / legacy mapper
- **未来删除计划**：等 mapper 端全部切换到消费 `async [token]` 后，这个 pass 可以直接移除

**为什么 schedule-tasks 阶段做 buffer reuse 更好**：

1. 在 token threading 之前消除冗余 Load，避免为无用 Load 生成 token
2. 不需要额外的 SSA pattern matching（直接基于 TaskGraph 的 in/out edges）
3. 同一个 pass 里保证 dep edges 也被正确更新（kernel→kernel 边取代 Store→Load）

---

## 核心特性

### 1. 完整 async token chain

`BlockLoad → Kernel → BlockStore` 三段都参与 token 传递：

```
BlockLoad_A ──tok0──┐
BlockLoad_B ──tok1──┤→ ADORA.kernel async[tok0,tok1] ──tokK──→ ADORA.BlockStore async[tokK]
BlockLoad_C ──tok2──┘  （同 memref tile 的 WAR 也汇入）
```

### 2. Buffer Reuse（on-chip 数据复用）

当 `Store(tile)` 后跟着 `Load(tile)` 访问同一 data block 时，`Load` 被消除，下游直接用 on-chip `LocalMemAlloc` buffer。

**3mm 效果**：`kernel_3mm_2` 原本要从 DRAM 读 2 次（`Load %arg0`, `Load %arg3`）的 Load 被消除，节省 2 次 DMA。

### 3. Token 可视化

```bash
cgra-opt input.mlir \
  --adora-schedule-tasks="emit-token=true dump-token-graph=tokens.dot"

dot -Tpng tokens.dot -o tokens.png
```

节点着色：
- **蓝** = `BlockLoad`
- **黄** = `Kernel`
- **红** = `BlockStore`

边标签 = "token"（绿色箭头），节点标签包含 `Id`、`KernelName`、`stream`。

### 4. Loop-carried Dep 检测（PR6 stub）

当 `affine.for` body 里存在 `Store(C[tile]) → Load(C[tile])` loop-carried RAW（跨 iteration），pass 会打印诊断：

```
[PR6-TODO] loop-carried token detected in affine.for — 
affine.for → scf.for iter_args conversion not yet implemented.
  Store: ...
  Load:  ...
```

完整的 `scf.for iter_args(!ADORA.token)` 变换在 PR6 中完成。

---

## Pipeline 内部步骤（实现细节）

`ScheduleADORATasksInFunction` 对每个包含 kernel 的 block 依次执行：

| 步骤 | 函数 | 作用 |
|------|------|------|
| A | `generateTaskGraphFromBlock` | 建 TaskGraph 节点 + default dep 边（KernelName 匹配） |
| B | `analyzeDependencyInGraph` | O(N²) 扫 RAW / WAR / WAW dep，填入 `depEdges` |
| C | `RemoveRedundantBlockStoreLoadPair` | 消除 Store→Load 冗余对，`replaceAllUsesWith` + erase |
| C' | `RemoveRedundantBlockLoads` | Load-after-Load 消除（目前 stub） |
| D | `threadTokensOnDMAs` | 根据 `depEdges` 重建 op 为 async 形式，加 `!ADORA.token` |
| D2 | `findLoopCarriedStoreLoadPair` + `wireLoopCarriedToken` | 检测 loop-carried dep，PR6 stub |
| E | `verifyTokensMatchSummary` | `cross-check=true` 时验证 token 与 summary 一致 |

---

## 调试技巧

### 看 task graph 拓扑

```bash
cgra-opt input.mlir \
  --adora-schedule-tasks="emit-token=true dump-token-graph=tokens.dot"
cat tokens.dot
```

### 只看 dep_summary 不改 IR

```bash
cgra-opt input.mlir --adora-schedule-tasks \
  | grep adora.dep_summary
```

### CI 交叉验证

```bash
cgra-opt input.mlir \
  --adora-schedule-tasks="emit-token=true cross-check-summary-vs-token=true"
```

### 跑 experiment 里的 4 个标准例子

```bash
cd experiment/taskschedule
bash review.sh            # 全跑 schedule-tasks pass
bash 03_3mm/run.sh        # 单跑 3mm，显示 buffer reuse 效果
bash e2e_pipeline.sh      # 完整 4-pass pipeline 验证（schedule→streams→lower→runtime）
```

### End-to-End Pipeline 验证（e2e_pipeline.sh）

验证 `!ADORA.token` 能完整穿透 4 个 pass，最终下沉为 `llvm.call @adoraEvent*`：

```
▶ 01_linear_chain:    PASS  (create=3 record=3 wait=3 destroy=3)
▶ 02_fanin:           PASS  (create=4 record=4 wait=4 destroy=4)
▶ 03_3mm:             PASS  (create=8 record=8 wait=8 destroy=8)
▶ 04_gemm_tiled:      PASS  (create=5 record=5 wait=6 destroy=5)
```

验收条件：
- Pipeline 无 crash
- 最终 IR 里**没有**残留的 `!ADORA.token`（全部下沉成 runtime ABI）
- `create`/`record`/`wait`/`destroy` 数量合理匹配

---

## 已知限制

1. **Loop-carried token**（PR6）：`affine.for` body 内的跨 iteration Store→Load dep 只检测不变换
2. **Load-after-Load 消除**：`RemoveRedundantBlockLoads` 是 stub，暂未实现
3. **`AccessSameDataBlock`**：动态 shape / 复杂 tiling 下保守返回 true，可能引入假阳性 dep edge
4. **非常量边界 `affine.for`**：loop-carried 检测和变换都不支持
5. **`test/cgra-mapper/FPVecAdd` lit 测试崩溃**：在 `adora-adjust-kernel-mem-footprint` pass
   发生 SIGSEGV，**不是本 pass 的问题**（在 schedule-tasks 之前运行）。需要独立 PR 修复。

---

## 参考

- 设计文档：`docs/async_token_design.md`
- PR 历史：`docs/pr4_review.md`
- 测试：`test/cgra-opt/schedule/`
- 示例：`experiment/taskschedule/`
- 源码：
  - `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`（主 pass）
  - `lib/Dialect/ADORA/Transforms/TaskGraph/`（图数据结构）
  - `include/ADORA/Dialect/ADORA/Transforms/Passes.td`（pass 注册 + options）

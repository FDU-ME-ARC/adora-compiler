# PR4 Review Document

Branch: `jhlou/scheduletasks`  
Base: `fd9b48f` (docs: add PR4 design spec)

---

## Commit 列表

| SHA | 标题 | 状态 |
|---|---|---|
| `85188a9` | feat(PR4-A): add adora-assign-streams pass + 3 lit tests | 待 review |
| `a2bcda0` | feat(PR4-B): wire assign-streams IDs into lower-async-tokens | 待 review |
| `9a79552` | fix(PR4-C): thread tokens across inter-kernel RAW deps in schedule-tasks | 待 review |

---

## PR4-A：`adora-assign-streams` pass

### 变更文件

| 文件 | 改动摘要 |
|---|---|
| `include/ADORA/Dialect/ADORA/Transforms/Passes.td` | 新增 `AssignStreams` pass def，option `max-streams`（默认 4） |
| `lib/Dialect/ADORA/Transforms/AssignStreams.cpp` | 新建，完整实现 |
| `lib/Dialect/ADORA/Transforms/CMakeLists.txt` | 加入 `AssignStreams.cpp` |
| `tools/cgra-opt/cgra-opt.cpp` | 注册 pass |
| `test/cgra-opt/kernel/assign_streams_linear.mlir` | 新增 lit test |
| `test/cgra-opt/kernel/assign_streams_parallel.mlir` | 新增 lit test |
| `test/cgra-opt/kernel/assign_streams_fanin.mlir` | 新增 lit test |

### 算法（`AssignStreams.cpp`）

1. 收集 FuncOp 内所有 async-capable ops（BlockLoad / BlockStore / KernelOp）
2. 建立 `token value → producer op` 映射
3. Kahn 拓扑排序（按 asyncDependencies SSA 边）
4. 贪心分配 stream ID：
   - 无前驱 → 分配新 stream（从 0 递增，上限 `max-streams`）
   - 有前驱 → 继承前驱中最小的 stream ID
5. 把 `stream : i32` 属性写回每个 op

### Review 重点

- `max-streams` 超出时的行为：目前 `% max-streams` 取模，独立 root op 超过 4 个时会复用 stream。是否需要改为 `min(newId, max-streams-1)` 的保守策略？
- stream 属性写在 op 上（非 SignalOp/WaitOp），是否和 `lower-async-tokens` 之后的属性读取位置一致（见 PR4-B）？

---

## PR4-B：`lower-async-tokens` 读 stream 属性

### 变更文件

| 文件 | 改动摘要 |
|---|---|
| `lib/Dialect/ADORA/Transforms/LowerAsyncTokens.cpp` | 新增 `getOpStream()` helper；两处 `emitSignal`/`emitWait` 读 op 的 `stream` attr |
| `test/cgra-opt/kernel/lower_async_streams_parallel.mlir` | 新增 pipeline test |

### 关键改动（`LowerAsyncTokens.cpp`）

```cpp
// 新增 helper（line 59-65）
static int32_t getOpStream(Operation *op, int32_t fallback) {
  if (auto a = op->getAttrOfType<IntegerAttr>("stream"))
    return static_cast<int32_t>(a.getInt());
  return fallback;
}

// emitSignal 调用（line 198）
emitSignal(b, op->getLoc(), op, ev, getOpStream(op, stream));

// emitWait 调用（line 231）
emitWait(b, op->getLoc(), op, ev, getOpStream(op, stream));
```

### Review 重点

- `getOpStream(op, stream)` 的 fallback 是 `defaultStream`（pass option，默认 0）。向后兼容性：不跑 `assign-streams` 时行为不变 ✓
- producer 用自己的 stream 发 signal，consumer 用自己的 stream 等待——语义是否与硬件事件模型一致？（signal 记录 producer stream 完成，wait 在 consumer stream 上等）

---

## PR4-C：修复 `schedule-tasks` 跨 kernel RAW token 生成

### 背景

对于 3mm 这类多 kernel 的情况（kernel_0 写 `%arg0`，kernel_2 读 `%arg0`），`emit-token=true` 之前**不生成 token**，导致 mapper emit 层无法知道任务间的 RAW 依赖。

### 根因

两处 bug 叠加：

1. `analyzeDependencyInGraph`：有错误注释"RAW already wired via SSA"，实际上 `_depEdges`（`threadTokensOnDMAs` 遍历的结构）从未收到 BlockStore→BlockLoad 的 RAW 边
2. `RemoveRedundantBlockStoreLoadPair`：尝试 erase BlockLoad MLIR op，但 op 有 live use，erase 失败，产生 broken IR

### 变更文件

| 文件 | 改动摘要 |
|---|---|
| `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp` | 两处修复（见下）|
| `test/cgra-opt/kernel/schedule_3mm.mlir` | 新增 3mm fan-in lit test |

### 具体改动

**Fix 1**（`analyzeDependencyInGraph` line 241）：补加 RAW dep edge：
```cpp
if (isa<BlockStoreNode>(ni) && isa<BlockLoadNode>(nj)) {
  auto sa = cast<BlockStoreNode>(ni)->getDataBlockStoreOp();
  auto lb = cast<BlockLoadNode>(nj)->getDataBlockLoadOp();
  if (checkDependencyBetweenBlockStoreAndBlockLoad(sa, lb)) {
    bool exact = AccessSameDataBlock(sa, lb);
    emit(ni, nj, DataBlockDepKind::RAW, exact);
  }
  continue;
}
```

**Fix 2**（`RemoveRedundantBlockStoreLoadPair`）：删除所有 MLIR op 操作，只保留图拓扑 wiring：
```cpp
// 只保留这一行：
addConnectionBetweenTwoNode(sourcekernel, sinkkernel, depType::Depend);
// 删掉：replaceAllUsesWith、DeleteNodeOperation、to_delete
```

BlockLoad MLIR op 留原地，让 `threadTokensOnDMAs` 把它 rebuild 成 `async [token]` form。

### Review 重点

- `RemoveRedundantBlockStoreLoadPair` 现在只做图拓扑 wiring，不做 buffer reuse 优化（原来想把 BlockLoad 替换为 LocalMemAlloc 复用）。这个 buffer reuse 的功能是否需要保留？如果需要，应该在 `threadTokensOnDMAs` 完成 token threading 之后再做一个单独的 pass。
- `checkDependencyBetweenBlockStoreAndBlockLoad` 是保守的（`AccessSameDataBlock || mayOverlapDataBlockRegion`），对动态 shape memref 会走 conservative overlap（返回 true）。是否可能引入假阳性 dep？

### 测试结果

```
lower_async_tokens           PASS
lower_async_tokens_fanin     PASS
lower_async_runtime          PASS
assign_streams_linear        PASS
assign_streams_parallel      PASS
assign_streams_fanin         PASS
lower_async_streams_parallel PASS
schedule_3mm (新)            PASS
```

---

## 当前状态（最新，截至 commit 33a298b）

### 完整 async token chain ✅

```
BlockLoad_A ──tok0──┐
BlockLoad_B ──tok1──┤→ ADORA.kernel async[tok0,tok1,...] ──tokK──→ ADORA.BlockStore async[tokK]
BlockLoad_C ──tok2──┘  （WAR dep 也汇入）
```

- `BlockLoad → Kernel`：load 完成后 kernel 才能开始计算 ✅
- `Kernel → BlockStore`：kernel 完成后 store 才写出 ✅  
- `BlockLoad WAR/RAW → BlockStore`：同 memref tile 的 intra-iteration fence ✅

### Buffer Reuse ✅

`RemoveRedundantBlockStoreLoadPair`：当 `Store(C[ti,tj])` 后紧跟 `Load(C[ti,tj])` 访问同一 data block，Load 被消除，下游直接用 on-chip `LocalMemAlloc` buffer：

```
kernel_0 → BlockStore(%local_0 → %C)
kernel_1 → [BlockLoad(%C) 已删] → 直接用 %local_0
```

**3mm 效果**：`kernel_3mm_2` 原来需要从 DRAM 读 2 次（Load %arg0, Load %arg3），现在被消除，节省 2 次 DMA。

### EmitCGRACall dep_flag ✅

`GenerateCGRACFGAndEXE` 的 `execute()` 命令 dep_flag 从 KernelOp 的 async token 计算：

- 无 async dep（root kernel）→ `dep_flag = 0`
- 有 async dep（等 Load/Store token）→ `dep_flag = EX_DEP_ST_LAST_TASK`

### 可视化工具 ✅

```bash
cgra-opt your.mlir --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/tok.dot"
dot -Tpng /tmp/tok.dot -o tok.png
```

### Experiment 示例 ✅

```
experiment/taskschedule/
  01_linear_chain/   # Load→Kernel→Store 最简 chain
  02_fanin/          # 2xLoad fan-in → Kernel → Store
  03_3mm/            # 3-kernel chain，buffer reuse
  04_gemm_tiled/     # 64x64x64 tiled GEMM，loop-carried dep 检测
  review.sh          # 一键运行所有例子，输出 token chain + dot 可视化
```

---

## 测试文件目录结构（最新）

```
test/cgra-opt/schedule/
  schedule_cgra_tasks_tokens.mlir   # PR2: emit-token TOKEN/NOTOKEN 双检
  schedule_3mm.mlir                 # PR4-C: 3mm buffer reuse + token chain
  schedule_tasks_dep_summary.mlir   # P1.0+P4.0: dep_summary attribute
  schedule_gemm_tiled.mlir          # PR4-D: 64x64x64 tiled GEMM WAR token

test/cgra-opt/kernel/               # 非 schedule 测试（assign_streams, lower_async 等）
```

---

## PR6：loop-carried token yield（下一步）

### 已完成

- **检测**：`findLoopCarriedStoreLoadPair` 能识别 `affine.for` body 内的 Store→Load loop-carried RAW dep（如 tiled GEMM `tk` loop 的 C tile）
- **诊断**：`wireLoopCarriedToken` stub 打印诊断信息，提示哪个 for loop 需要变换

### 待实现

`AffineForOp` 没有 `iter_args`，需要转换为 `scf.for`：

```mlir
// 目标：PR6 实现后
%init_tok = ADORA.event.create → !ADORA.token
%final_tok = scf.for %tk = 0 to 4
    iter_args(%carry = %init_tok) → (!ADORA.token) {
  // BlockLoad 消费上一 iteration 的 store token
  %c, %war_tok = ADORA.BlockLoad async [%carry] %C ... → !ADORA.token
  ...
  // BlockStore 产生本 iteration 的 token，传给下一 iteration
  %store_tok = ADORA.BlockStore async [...] %local, %C → !ADORA.token
  scf.yield %store_tok : !ADORA.token
}
ADORA.event.destroy %final_tok
```

**实现步骤**：
1. `wireLoopCarriedToken`：把 `AffineForOp` + 常量边界转成 `scf.for iter_args(!ADORA.token)`
2. 把 body ops 从 affine 移到 scf，替换 IV 引用
3. 在 BlockLoad 的 `asyncDependencies` 里加入 `carry_arg`
4. 在 `affine.yield` → `scf.yield` 时传入 BlockStore token
5. 在 loop 之前插入 `ADORA.event.create`（init token），之后插入 `ADORA.event.destroy`
6. `adora-lower-async-tokens` 对 `scf.for iter_args(!ADORA.token)` 的识别支持

**测试验证**：在 `schedule_gemm_tiled.mlir` 加 `CHECK: scf.for {{.*}} iter_args` 验证。

---

## 已知局限 / TODO

| # | 状态 | 描述 |
|---|------|------|
| 1 | ✅ 完成 | Buffer reuse：RemoveRedundantBlockStoreLoadPair replaceAllUsesWith |
| 2 | ✅ 完成 | EmitCGRACall execute dep_flag 从 async token 计算 |
| 3 | ✅ 完成 | 完整 BlockLoad→Kernel→BlockStore async token chain |
| 4 | 🔧 PR6 | loop-carried token yield（affine.for → scf.for iter_args） |
| 5 | 📋 TODO | RemoveRedundantBlockLoads：Load-after-Load 消除（stub，未实现） |
| 6 | 📋 TODO | EmitPytest 并发：Python 测试代码生成仍为串行 await |
| 7 | 📋 TODO | loop-carried dep 检测对非常量边界 affine.for 的支持 |
5. **多分块精确性**：`AccessSameDataBlock` 在动态 shape 或多维 tiled 情况下保守返回 true，可能引入假阳性 dep edge，需进一步验证。

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

## 测试文件目录结构（最新）

schedule 相关测试已从 `test/cgra-opt/kernel/` 迁移至独立子目录：

```
test/cgra-opt/schedule/
  schedule_cgra_tasks_tokens.mlir   # PR2: emit-token TOKEN/NOTOKEN 双检
  schedule_3mm.mlir                 # PR4-C: 3mm multi-kernel fan-in token
  schedule_tasks_dep_summary.mlir   # P1.0+P4.0: dep_summary attribute
  schedule_gemm_tiled.mlir          # PR4-D: 64x64x64 tiled GEMM, loop-carried dep
```

`test/cgra-opt/kernel/` 里保留其他非 schedule 相关测试（assign_streams_*, lower_async_*, simplify_loop_levels, gemm）。

---

## PR4-D：tiled GEMM 测试与 loop-carried dep

### 新增测试：`schedule_gemm_tiled.mlir`

**矩阵规模**：`C[64×64] += A[64×64] × B[64×64]`，tile size = 16×16×16

**循环结构**：
```
affine.for %ti = 0 to 4 {          // tile-i (M)
  affine.for %tj = 0 to 4 {        // tile-j (N)
    affine.for %tk = 0 to 4 {      // tile-k (K-reduction) ← kernel 在此
      BlockLoad  C_tile  %arg2[ti*16, tj*16]  ← loop-carried WAR/RAW
      BlockLoad  A_tile  %arg0[ti*16, tk*16]
      BlockLoad  B_tile  %arg1[tk*16, tj*16]
      LocalMemAlloc C_local
      ADORA.kernel (C_local = C_tile + A_tile × B_tile)
      BlockStore C_local → %arg2[ti*16, tj*16]  ← loop-carried write
    }
  }
}
```

**当前 pass 行为（intra-iteration WAR fence）**：

`analyzeDependencyInGraph` 检测到 `tk` body 内：
- `BlockLoad(%arg2, C_tile)` → `BlockStore(%arg2, C_tile)` 有 **WAR** dep（同 memref，重叠区域）

`threadTokensOnDMAs` 将其 wire 为：
```mlir
%c_result, %war_tok = ADORA.BlockLoad %arg2 ... -> !ADORA.token
ADORA.BlockStore async [%war_tok] %c_local, %arg2 ...
```

**loop-carried dep（PR6 待实现）**：

`tk` loop 的真正 loop-carried RAW dep：
- `iteration tk=N` 的 `BlockStore(%arg2)` → `iteration tk=N+1` 的 `BlockLoad(%arg2)` — 相同 C tile

这需要通过 `scf.for iter_args(!ADORA.token)` 将 token 跨 iteration 传递，目前**尚未实现**。

### 已知 bug 修复：`dumpGraphAsDot` 访问 stale pointer

**文件**：`ScheduleAdoraTasks.cpp:689`

`threadTokensOnDMAs` 执行后，TaskGraph 里的 TaskNode 持有已被 erase 的 old op 指针。后续 `dumpGraphAsDot` 访问这些 stale pointer 导致 segfault（仅在 3 层及以上嵌套 affine.for + emit-token=true 时触发）。

**修复**：`emit-token=true` 时跳过 post-token dot dump：
```cpp
// Skip dot dump after threadTokensOnDMAs to avoid stale-pointer segfault.
if (!emitTokens)
  graph->dumpGraphAsDot(filename);
```

---

## 下一步计划

见 `docs/async_token_design.md` §PR4 或下方。

---

## 已知局限 / TODO

1. **buffer reuse**：`RemoveRedundantBlockStoreLoadPair` 目前只做 kernel→kernel 图拓扑 wiring，不做 SSA 替换。真正的 buffer reuse（replaceAllUsesWith + erase BlockLoad，让 kernel 直接复用 on-chip buffer）需要在 `threadTokensOnDMAs` 之前完成，且删除前必须先 `replaceAllUsesWith`。
2. **PR6 loop-carried token yield**：`affine.for` body 内最后一个 BlockStore 产生的 token 需要通过 `scf.for iter_args(!ADORA.token)` 传给下一 iteration 的 BlockLoad。典型场景：tiled GEMM `tk` loop 的 C_tile RAW dep（`schedule_gemm_tiled.mlir` 有注释标记）。
3. **EmitCGRACall dep_flag**：mapper emit 层还未消费 `async [token]` 依赖来精确化 `LD_DEP_ST_LAST_TASK`。
4. **EmitPytest 并发**：Python 测试代码生成还是全串行 `await`，未利用独立 task 的并发机会。
5. **多分块精确性**：`AccessSameDataBlock` 在动态 shape 或多维 tiled 情况下保守返回 true，可能引入假阳性 dep edge，需进一步验证。

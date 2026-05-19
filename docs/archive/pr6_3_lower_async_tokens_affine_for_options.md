# PR6.3 — LowerAsyncTokens 扩展到 affine.for iter_args：三种方案对比

**Status**: design draft, awaiting user review
**Branch**: `jhlou/pr6-loop-carried-analysis` (HEAD `bfa155d`)
**Depends on**: PR6.2 (已合并：affine.for iter_args 形态生效)
**Blocks**: PR6.4 (mapper/emit 异步原语映射)

---

## 0. 背景

PR6.2 跑通后，IR 形态长这样（05_loop_carried 实测）：

```mlir
func.func @loop_carried_min(%arg0: memref<16xf32>) {
  %0 = ADORA.event.create -> !ADORA.token
  %1 = ADORA.event.create -> !ADORA.token
  %2 = ADORA.event.create -> !ADORA.token
  %3:3 = affine.for %arg1 = 0 to 4
           iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %2)
           -> (!ADORA.token, !ADORA.token, !ADORA.token) {
    %result, %asyncToken = ADORA.BlockLoad async [%arg3] %arg0 [0] ...
                                        -> !ADORA.token
    %4 = ADORA.LocalMemAlloc memref<16xf32>
    %5 = ADORA.kernel async [%asyncToken] { ... } : !ADORA.token
    %6 = ADORA.BlockStore async [%5, %arg2, %arg4] %4, %arg0 [0]
                                        -> !ADORA.token
    affine.yield %asyncToken, %6, %6
        : !ADORA.token, !ADORA.token, !ADORA.token
  }
  return
}
```

接下来 `--adora-lower-async-tokens` 跑过后，body 内的 `BlockLoad/BlockStore/Kernel` 会被 PR3 已实装的 rebuild*Sync 退回 sync 形态：
- 去掉 `async [...]` 操作数
- 去掉 `!ADORA.token` 结果
- 配对插入 `ADORA.event.create / signal / wait / event.destroy`

**问题来了**：PR3 当前的 LowerAsyncTokens **完全不识别** `affine.for` 上的 `iter_args(!ADORA.token)` 与 `affine.yield %tok`。如果什么都不做，lower 后会得到一个**自洽但臃肿**的 IR：

```mlir
%3:3 = affine.for %arg1 = 0 to 4
         iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %2)
         -> (!ADORA.token, !ADORA.token, !ADORA.token) {
  // body 已被 rebuild 为 sync，原来 use %arg3 的 BlockLoad 已不再 use 它
  // %arg2 / %arg3 / %arg4 在 body 内变成"dead"
  ADORA.BlockLoad %arg0 [0] ...      // sync
  %5 = ADORA.LocalMemAlloc ...
  ADORA.kernel { ... }                // sync
  ADORA.BlockStore %5, %arg0 [0] ... // sync
  // yield 的 %asyncToken / %6 / %6 已不存在（被 rebuild 删了）
  // 但 affine.yield 仍要求 3 个 !ADORA.token operands → IR 不合法
}
```

这会导致 **verifier 直接 fail**：yield operand 数与类型必须匹配 iter_args，而 producer 的 token result 已被 rebuild 删掉。

所以 PR6.3 必须做点什么。问题是**做什么 / 做到什么程度**——下面三个方案。

---

## 1. 方案 A：完全清空 iter_args（推荐 ⭐）

### 思路

LowerAsyncTokens 跑完后，**重建 affine.for**，让它变回**普通**的（无 iter_args / 无 results）affine.for。理由：
- event 在 host 代码里是**命名变量**（如 `event_0`），跨迭代复用通过同一个变量自动达成，根本不需要 SSA 形式传递
- body 内 op 已 sync，sync 形态本身**没有任何 token 可以 yield**
- iter_args 在 lower 后已无意义，留着只会让 emit 端误以为还要做什么

### 改写示意

**Before（PR6.2 输出）:**
```mlir
%3:3 = affine.for %arg1 = 0 to 4
         iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %2)
         -> (!ADORA.token, !ADORA.token, !ADORA.token) {
  %r, %tA = ADORA.BlockLoad async [%arg3] %arg0[0] ... -> !ADORA.token
  ...
  %6 = ADORA.BlockStore async [%5, %arg2, %arg4] ... -> !ADORA.token
  affine.yield %tA, %6, %6 : !ADORA.token, !ADORA.token, !ADORA.token
}
```

**After（方案 A）:**
```mlir
// 三个 event 仍然在 loop 外 create
%e_load  = ADORA.event.create -> !ADORA.token
%e_store = ADORA.event.create -> !ADORA.token
%e_war   = ADORA.event.create -> !ADORA.token

affine.for %arg1 = 0 to 4 {           // ← 无 iter_args，无 results
  // body 内 op 都是 sync 形态
  ADORA.wait %e_store on stream 1 : !ADORA.token   // 等上一轮 store
  ADORA.BlockLoad %arg0 [0] ...                    // sync load
  ADORA.signal %e_load on stream 1 : !ADORA.token

  %lm = ADORA.LocalMemAlloc memref<16xf32>
  ADORA.wait %e_load on stream 2 : !ADORA.token
  ADORA.kernel { ... }                              // sync
  ADORA.signal %e_kernel on stream 2 : !ADORA.token

  ADORA.wait %e_kernel on stream 3 : !ADORA.token
  ADORA.BlockStore %lm, %arg0 [0] ...               // sync store
  ADORA.signal %e_store on stream 3 : !ADORA.token
}

ADORA.event.destroy %e_load
ADORA.event.destroy %e_store
ADORA.event.destroy %e_war
```

跨迭代依赖通过**同名 event** + **第 0 次迭代前已 create**的 event 达成自然语义：第 0 次迭代 `wait %e_store` 等的是 create 出来的初始（已 signaled）状态，第 k>0 次等的是上一轮 `signal %e_store`。

### 实现

LowerAsyncTokens 末尾加 Pass 5：

```cpp
// Pass 5: strip !ADORA.token iter_args from every affine.for.
func.walk([&](affine::AffineForOp fo) {
  if (!hasTokenIterArg(fo)) return;
  rebuildAffineForWithoutTokenIterArgs(fo);
});
```

`rebuildAffineForWithoutTokenIterArgs` 是 PR6.2 改写的**镜像操作**：构造新 affine.for（inits 空），splice body，新 yield 也是空，erase 旧 for。约 80 行。

### 优点

| ✅ | 说明 |
|---|---|
| emit 端零负担 | 三个 emit 都不需要懂 iter_args，affine.for 退回为普通循环 op |
| IR 干净 | 没有 dead iter_args / 没有"看起来重要其实没用"的 SSA 边 |
| 与 PR3 现有 4 件套语义对齐 | event 本来就是命名 host 变量，SSA 传递只是 compile-time 装饰 |
| 后续 PR6.4 工作量最小 | emit 只需新增 4 个 op case，不动 affine.for 处理 |

### 缺点

| ⚠️ | 说明 |
|---|---|
| 失去 SSA 可视化 | affine.for 上看不到"这个循环带跨迭代依赖"的标记 → 但 LowerAsyncTokens 是 pipeline 终点附近，调试需求弱 |
| 一次性改写无法回退 | 一旦 strip 就没了；但 PR3 已有 `dropTokensOnly` 旁路，需要保留 token 形态时可用 |

---

## 2. 方案 B：保留 iter_args 但替换为 runtime handle 类型

### 思路

不删 iter_args，而是把元素类型从 `!ADORA.token` 换成 runtime event handle 类型（例如 `i64`，对应 host 端 `adora_event_t`）。

### 改写示意

**After（方案 B）:**
```mlir
%e_load_h  = ADORA.event.create_handle -> i64    // 假定新增 op
%e_store_h = ADORA.event.create_handle -> i64
%e_war_h   = ADORA.event.create_handle -> i64

%3:3 = affine.for %arg1 = 0 to 4
         iter_args(%arg2 = %e_war_h, %arg3 = %e_load_h, %arg4 = %e_store_h)
         -> (i64, i64, i64) {
  ADORA.wait_handle %arg3 on stream 1 : i64
  ADORA.BlockLoad %arg0[0] ...
  %t1 = ADORA.signal_handle on stream 1 -> i64
  // ...
  affine.yield %t1, %t2, %t3 : i64, i64, i64
}
```

需要给 ADORA dialect 新增一对 op：
- `ADORA.event.create_handle -> i64`
- `ADORA.signal_handle on stream N -> i64`
- `ADORA.wait_handle %h on stream M : i64`

### 优点

| ✅ | 说明 |
|---|---|
| 保留 SSA 可视性 | IR 仍能从 yield/iter_args 看出跨迭代依赖结构 |
| emit 端 dataflow 友好 | 如果将来要在 host 代码生成时做 SSA 优化（dead event elimination 等），SSA 形式更便利 |

### 缺点

| ⚠️ | 说明 |
|---|---|
| 三个 emit 必须懂 iter_args | 每个 emitter 都要把 affine.for 的 inits/yield/result mapping 映射到 host 变量，比方案 A 多出 ~150 行/emitter |
| 与现有 4 件套语义重复 | event 本就是命名 host 变量；再走一遍 SSA i64 等于做了两次同样的事 |
| 必须新增 3 个 ADORA op | 设计、verifier、印表项目 +90 行（且会让 lower-async-tokens 出口产生**两种**等价 op 集合：原生 event ops 用于直线代码，handle ops 用于 iter_args 体内——更乱） |
| iter_args 上的 i64 与普通整数 i64 混淆 | 失去类型安全；emit 端需要靠 attr/op 类型区分 |

---

## 3. 方案 C：不在 lower-async-tokens 处理，emit 自行识别

### 思路

LowerAsyncTokens 完全跳过 affine.for，保留 `iter_args(!ADORA.token)` 形态进入 mapper/emit。让 emit 自己决定：是当作"循环 lift event 命名"，还是别的什么。

### 改写示意

LowerAsyncTokens 输出（与方案 B 类似但**保留 `!ADORA.token` 类型**）：
```mlir
%3:3 = affine.for %arg1 = 0 to 4
         iter_args(%arg2 = %0, %arg3 = %1, %arg4 = %2)
         -> (!ADORA.token, !ADORA.token, !ADORA.token) {
  // 但是 body 内 op 已经 sync 了！没有 op 产 token 给 yield
  ADORA.BlockLoad ...     // sync
  ADORA.signal %e_load on stream 1
  ...
  affine.yield ??? : !ADORA.token×3
}
```

**这一步就出问题**：body 内已无 token producer，yield 没东西可用。要么 emit 端识别 yield 不合法直接报错（不可行），要么 LowerAsyncTokens 还得给 yield 编造 token——绕一圈又回到 A/B。

### 优点

无。

### 缺点

| ⚠️ | 说明 |
|---|---|
| 把复杂度推给 3 个 emit | 每个都要自己实现"识别死 token iter_args + 忽略"逻辑 |
| IR 不合法 | yield 拿不到 token，verify 直接 fail；要么禁用 verifier（危险），要么在 LowerAsyncTokens 里造假 token（绕回 A 但代码更丑） |

**结论：C 方案不可行。** 列在这里是为完整性。

---

## 4. 推荐

**方案 A**（完全清空）。理由汇总：

1. event 在 host 代码层面**本就是命名变量**，跨迭代复用天然成立，SSA 传递是空架子
2. 三个 emit 改动最小（不需要懂 iter_args）
3. 不引入新 op，不破坏现有 dialect 形态
4. 与 PR3 已实装的 4 件套（create/signal/wait/destroy）语义完美对齐
5. 实现量最小（~80 行）

唯一让步：失去 affine.for 上"我有跨迭代依赖"的视觉标记。但
- PR6.1 的 `adora.lc_dep_summary` 仍以 attr 形式记在 FuncOp 上，作为静态文档
- LowerAsyncTokens 已经接近 pipeline 末端，调试期更需要看的是 event 名而非 SSA token

## 5. 不立即定下来的开口

- **default-init event 是否要显式 signal 一次**：第 0 次迭代的 `wait %e_store` 需要 event "已 signaled"。PR3 的 `event.create` 默认就是 signaled-at-birth，确认这一点即可（无需改 dialect）。**Open**：让我下一轮 verify 一下。
- **如果未来某个用例真需要 SSA 跨迭代追踪**（例如做 cross-iter alias analysis），可以再加一个 `dropTokensOnly=false, stripIterArgs=false` 模式作为脱离 emit 的可视化产物。**Open**：现在不做。

## 6. 决策需要回答的问题

1. **采用方案 A 吗？** （Y/N）
2. 若 Y，PR6.3 主体工作约 80-150 行新增 + ~30 行 test，单 PR 提交是否 OK？
3. 之后 PR6.4 是否按预设 4 子 PR（`EventEmitter` helper + 3 个 emitter 各 1 PR）拆分？

---

## 附录 — 三方案对比速查表

| 维度 | A. 完全清空 | B. handle 类型 | C. 留给 emit |
|---|---|---|---|
| LowerAsyncTokens 改动 | +80 行 | +120 行（含新 op）| 0 |
| ADORA dialect 新 op | 0 | 3 | 0 |
| 三个 emit 改动 | +4 op case ×3 | +4 op case + iter_args 处理 ×3 | +iter_args 处理 ×3 + 死 token 兼容 |
| IR 合法性 | ✅ | ✅ | ❌ verify fail |
| 语义对齐 PR3 4 件套 | ✅ | 部分 | ❌ |
| 总改动估行 | ~250 | ~520 | 不可行 |
| 推荐 | ⭐ | — | — |

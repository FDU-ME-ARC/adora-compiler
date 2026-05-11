# PR6.2 — Thread Loop-Carried Tokens on `affine.for`

**Status**: spec draft (post v4 plan revision)
**Branch base**: `jhlou/pr6-loop-carried-analysis` (commit `f47aff1`)
**Depends on**: PR6.1 (`adora.lc_dep_summary`), PR3 (`!ADORA.token`,
`ADORA.event.create/destroy`), PR2 (intra-iteration token threading on DMAs).

---

## 0. 关键事实锚点（防止再踩坑）

> `affine.for` **可以** yield 任意自定义类型（含 `!ADORA.token`）。
> 详见 `docs/affine_for_yield_token_verification.md` 的 4 段实测证据。
> TableGen 源：`mlir/include/mlir/Dialect/Affine/IR/AffineOps.td:117-236`，
> `Variadic<AnyType>:$inits`。
>
> **本 PR 不 promote affine.for → scf.for**。
> `lib/Dialect/ADORA/Lowering/ADORAToSCF.cpp:56-131` 的 `affineForOuterToSCF`
> 不再服务于本特性；本 PR **不动**它，由独立 cleanup PR 评估去留。

---

## 1. 目标

读取 PR6.1 已写到 `func->setAttr("adora.lc_dep_summary", ...)`（见
`lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:789-793`）的跨迭代依赖
信息，对每个有非空 LC edges 的 `affine.for` **原地重建**为带
`iter_args(!ADORA.token × N)` / `affine.yield %tok × N` 的版本，并把 token
正确地接入 body 内 DMA / Kernel op 的 `async [...]` 依赖列表，达成跨迭代
异步同步语义。

### 输入示例（PR6.1 处理后的 IR 片段）

```mlir
func.func @gemm(%A: memref<...>, %B: memref<...>, %C: memref<...>)
    attributes {adora.lc_dep_summary = [{
      loop_idx = 0, loop_op = "affine.for",
      edges = [{kind = "RAW", step = 1, exact = true}]}]} {
  affine.for %tk = 0 to 4 {
    %a, %ta = ADORA.BlockLoad async %A[...] : ... -> !ADORA.token
    %b, %tb = ADORA.BlockLoad async %B[...] : ... -> !ADORA.token
    %c, %tc = ADORA.BlockLoad async %C[...] : ... -> !ADORA.token
    %kt    = ADORA.kernel  async [%ta, %tb, %tc] { ... } : !ADORA.token
    %st    = ADORA.BlockStore async [%kt] %c_local, %C[...] : !ADORA.token
  }
  return
}
```

### 期望输出

```mlir
func.func @gemm(...) {
  %null0 = ADORA.event.create -> !ADORA.token            // sentinel
  %tN = affine.for %tk = 0 to 4
                  iter_args(%lc = %null0) -> !ADORA.token {
    %a, %ta = ADORA.BlockLoad async %A[...] : ... -> !ADORA.token
    %b, %tb = ADORA.BlockLoad async %B[...] : ... -> !ADORA.token
    %c, %tc = ADORA.BlockLoad async %C[...] : ... -> !ADORA.token
    %kt    = ADORA.kernel  async [%ta, %tb, %tc, %lc] { ... } : !ADORA.token
                                              // ^^^^ 新增：消费上一轮 store
    %st    = ADORA.BlockStore async [%kt] %c_local, %C[...] : !ADORA.token
    affine.yield %st : !ADORA.token            // 把本轮 store token 传下轮
  }
  ADORA.event.destroy %tN : !ADORA.token       // 可选：循环结束的全局同步点
  return
}
```

---

## 2. 算法

### 2.1 Chain 聚合（属于 Analysis 库扩展）

PR6.1 已给出每条边的 `(src, dst, kind, exact, step, enclosingLoop)`。本 PR
需要把它们聚合成"独立 token chain"：一条 chain = iter k 的某个 producer op
→ iter k+1 的某个 consumer op，且**同一 chain 内复用同一个 iter_arg**。

最朴素的策略（首版采用）：

- 每条 `LoopCarriedDepEdge` 自成一条 chain（不做合并）；
- 同一对 `(src, dst)` 上的多条 edges（如 WAW+RAR 同时存在）合并为 1 条。

→ chain 数 N = unique(src,dst) edges 个数。

后续可以做更聪明的合并（例如同一对 op 多 kind 仅需一个 token），但**首版
不优化**，保持算法直白。

### 2.2 IR 改写

```pseudo
for each (loopOp, chains) in func.attr("adora.lc_dep_summary"):
  assert isa<affine::AffineForOp>(loopOp)
  N = chains.size()
  
  # ---- 1. 在 loop 之前生成 N 个 null sentinel token
  OpBuilder b(loopOp)
  SmallVector<Value> inits
  for i in 0..N:
    inits.push_back(b.create<ADORA::EventCreateOp>(loc).getResult())
  
  # ---- 2. 重建一个新的 affine.for，签名带 iter_args / results
  oldFor = cast<AffineForOp>(loopOp)
  newFor = b.create<AffineForOp>(
      loc,
      oldFor.getLowerBoundOperands(), oldFor.getLowerBoundMap(),
      oldFor.getUpperBoundOperands(), oldFor.getUpperBoundMap(),
      oldFor.getStep(),
      /*iterArgs=*/inits)
  
  # ---- 3. splice body
  Block* oldBody = oldFor.getBody()
  Block* newBody = newFor.getBody()
  # 新 body 已经由 builder 创建好，包含 induction var + N 个 iter block args
  # 把 old body 的 ops（除 affine.yield 之外）搬到 newBody 末尾之前
  newBody->getOperations().splice(
      std::prev(newBody->end()),                # 在新的 affine.yield 之前插
      oldBody->getOperations(),
      oldBody->begin(),
      std::prev(oldBody->end()))                # 不搬旧的 affine.yield
  
  # ---- 4. RAUW: induction var
  oldBody->getArgument(0).replaceAllUsesWith(newBody->getArgument(0))
  
  # ---- 5. 接 token
  SmallVector<Value> yieldOperands
  for i, chain in enumerate(chains):
    Value lcArg = newBody->getArgument(1 + i)   // 跳过 induction var
    # consumer (chain.dst) 把 lcArg 加入它的 async deps
    appendAsyncDep(chain.dst, lcArg)
    # producer (chain.src) 的 token result 作为 yield operand i
    yieldOperands.push_back(getTokenResultOf(chain.src))
  
  # ---- 6. 替换 newBody 的 yield
  AffineYieldOp newYield = cast<AffineYieldOp>(newBody->getTerminator())
  OpBuilder yb(newYield)
  yb.create<AffineYieldOp>(loc, yieldOperands)
  newYield.erase()
  
  # ---- 7. RAUW 旧 for 的（无）results，然后 erase
  oldFor.erase()
```

注意点：

| 关键点 | 处理 |
|---|---|
| `affine.for` 构造函数 | 用接受 `iterArgs` 的 overload；详 `Affine/IR/AffineOps.h` |
| splice 时跳 terminator | 用 `std::prev(oldBody->end())` 避免搬入旧 `affine.yield` |
| induction var index | `newBody->getArgument(0)` 永远是 IV，`getArgument(1..N)` 为 iter_args |
| consumer / producer 跨 block 性 | LC 边由 PR6.1 算出，**两端 op 一定在同一个 loop body 里**，无须跨 block 处理 |
| 多条 chain 顺序 | 严格按 `chains` 数组下标 i → iter_arg 槽 i → yield operand i，保持一致 |

### 2.3 `appendAsyncDep(op, tok)` 实现细节

不同 op 的 `asyncDependencies` operand 位置不同：

| Op | 处理 |
|---|---|
| `ADORA::KernelOp` | `Variadic<ADORA_TokenType>:$asyncDependencies`（首参） — 调用 `op.getAsyncDependenciesMutable().append(tok)` |
| `ADORA::DataBlockLoadOp` | 同上（待确认 td） |
| `ADORA::DataBlockStoreOp` | 同上 |
| 其它 | 报错并跳过（保守） |

实现里写一个 `static LogicalResult appendAsyncDep(Operation* op, Value tok)`
集中处理；按 op 类型 dyn_cast 后调对应的 mutable accessor。

### 2.4 `getTokenResultOf(op)` 实现细节

约定：ADORA dialect 中产 token 的 op 把 `!ADORA.token` 放在 `getResults()`
的**最后一个**位置（BlockLoad 返回 `(memref, token)`，BlockStore / Kernel
只返回 `token`）。helper 用 type check 找到第一个 `ADORA::TokenType`
result，找不到则返回 null + 日志。

---

## 3. 文件落点

### 新建
- `lib/Dialect/ADORA/Transforms/ThreadLoopCarriedTokens.cpp`（~250 行）

### 修改

| 文件 | 改动 |
|---|---|
| `include/ADORA/Dialect/ADORA/Analysis/LoopCarriedDep.h` | 新增 `struct LCChain { Operation* producer; Operation* consumer; }` + `SmallVector<LCChain> groupEdgesIntoChains(const LoopCarriedDepResult&)` 声明 |
| `lib/Dialect/ADORA/Analysis/LoopCarriedDep.cpp` | 实现 `groupEdgesIntoChains` |
| `lib/Dialect/ADORA/Analysis/CMakeLists.txt` | 不需要改（同库） |
| `include/ADORA/Dialect/ADORA/Transforms/Passes.td` | 新增 `ThreadLoopCarriedTokens` def（pass name: `adora-thread-loop-carried-tokens`，scope: `func::FuncOp`） |
| `include/ADORA/Dialect/ADORA/Transforms/Passes.h` | 新增 `createThreadLoopCarriedTokensPass()` 声明 |
| `lib/Dialect/ADORA/Transforms/CMakeLists.txt` | 注册新 cpp |

### 不动
- `lib/Dialect/ADORA/Lowering/ADORAToSCF.cpp`（`affineForOuterToSCF` 保留）
- `ScheduleAdoraTasks.cpp`（PR6.1 产物消费方）
- 现有 TaskGraph

---

## 4. Pass 注册（TableGen 模板）

```tablegen
def ThreadLoopCarriedTokens
    : Pass<"adora-thread-loop-carried-tokens", "func::FuncOp"> {
  let summary = "Thread !ADORA.token through affine.for iter_args based on "
                "adora.lc_dep_summary, materializing loop-carried async deps.";
  let description = [{
    For every affine.for referenced by the `adora.lc_dep_summary` FuncOp
    attribute (produced by --adora-schedule-tasks via the LoopCarriedDep
    analysis), this pass rebuilds the loop in place with
    `iter_args(!ADORA.token × N)` and inserts the corresponding
    `affine.yield`, where N is the number of independent loop-carried
    chains. The newly introduced iter_args are appended to the
    `asyncDependencies` of the consumer op at iteration k+1, and the
    producer op's token result is yielded back as the iter_arg for the
    next iteration.

    The pass does NOT promote affine.for to scf.for (see
    docs/affine_for_yield_token_verification.md for empirical proof that
    affine.for fully supports custom token iter_args).
  }];
  let constructor =
      "mlir::ADORA::createThreadLoopCarriedTokensPass()";
  let dependentDialects = ["::mlir::ADORA::ADORADialect",
                           "::mlir::affine::AffineDialect"];
}
```

Pipeline 插入位置（待确认）：紧跟在 `--adora-schedule-tasks` 之后、
`--adora-lower-async-tokens` 之前。

---

## 5. 测试计划

### 5.1 04_gemm_tiled（已存在）

`experiment/taskschedule/04_gemm_tiled/run.sh` 末尾追加：

```bash
cgra-opt out.mlir \
  --adora-thread-loop-carried-tokens \
  -o threaded.mlir
FileCheck threaded.mlir --check-prefix=THREAD <<< '
THREAD-LABEL: func.func @gemm
THREAD: ADORA.event.create -> !ADORA.token
THREAD: affine.for {{.*}} iter_args(%{{.*}} = %{{.*}}) -> (!ADORA.token)
THREAD: ADORA.kernel async [{{.*}}, %{{.*}}] {{.*}} : !ADORA.token
THREAD: ADORA.BlockStore async [%{{.*}}] {{.*}} : !ADORA.token
THREAD: affine.yield %{{.*}} : !ADORA.token
'
```

### 5.2 05_loop_carried（PR6.1 已建）

复用同一 CHECK 模板，验证更简单的 LC case。

### 5.3 06_no_lc（新建，可选）

构造一个只 load 不 store 的循环，预期 `adora.lc_dep_summary` 为空 / 不存在；
本 pass 应是 no-op：

```
CHECK-NOT: iter_args
CHECK-NOT: ADORA.event.create
```

### 5.4 负面对照

构造一个 producer-token / consumer-async-deps 类型不匹配的 input，预期 pass
报 diagnostic 但**不**崩。

---

## 6. 风险 & 限制

| 风险 | 说明 | 应对 |
|---|---|---|
| 一对 op 多 kind 边产生多 token | 首版每 unique (src,dst) 一条 chain，可能仍然过保守 | 后续优化，保留 TODO |
| body 内有非 ADORA op 持有 token | 不应该发生（token 只来自 ADORA ops），断言保护 | `assert isa<ADORA::TokenType>(producer.token.getType())` |
| `affine.for` 已经有 iter_args（非 token） | 当前 codebase 不存在这种情况；首版断言不允许 | TODO：未来支持时把 inits 拼到末尾 |
| sentinel token 的语义 | `ADORA.event.create` 产的 token 必须 "signaled-at-birth"，否则第 0 次迭代会等死 | PR3 已定义此语义；如未定义需补 attr，详 PR3 |
| `affineForOuterToSCF` 仍可能在 pipeline 内被调用 | 仅作用于"包住 kernel 的最外层"，与本 pass 处理的内层 loop 互不相干 | 跑通后再独立 cleanup PR |

---

## 7. 验收清单

- [ ] `cgra-opt --adora-thread-loop-carried-tokens` 注册可见
- [ ] 04_gemm_tiled / 05_loop_carried run.sh + FileCheck 全绿
- [ ] 06_no_lc（可选）no-op 验证通过
- [ ] `ninja cgra-opt` 全量编译干净（no warning regression）
- [ ] commit message 引用 `f47aff1` (v4 plan) + `docs/affine_for_yield_token_verification.md`

---

## 8. PR 拆分（保持 ≤ 350 行 diff/PR）

| sub-PR | 内容 | 估行 |
|---|---|---|
| **6.2a** | Analysis 扩展：`groupEdgesIntoChains` | ~80 |
| **6.2b** | Transform pass + 注册 + 编译 | ~250 |
| **6.2c** | tests（04 / 05 / 可选 06） | ~120 |

可视情况合并为单 PR 提交。

---

## 9. 复用 / 下次新开 context

```bash
# 先恢复上下文
cat docs/fix_adjust_memory_footprint_prompt.md   # 上一任务（已完成）
cat docs/pr6_loop_carried.md                     # PR6 主设计 v4
cat docs/affine_for_yield_token_verification.md  # 事实锚点（防再踩坑）
cat docs/pr6_2_thread_loop_carried_tokens_prompt.md   # 本文件

git checkout jhlou/pr6-loop-carried-analysis
git log --oneline -5     # 期望看到 f47aff1 在 HEAD
```

---

## 10. v4 与 v3 计划差异速查

| 项 | v3（错误） | v4（正确） |
|---|---|---|
| affine.for 能否 yield !ADORA.token | 假设不能 | **实测可以** |
| PR6.2 职责 | promote affine.for → scf.for | **直接在 affine.for 上 thread** |
| PR6.3 职责 | 在 scf.for 上 thread token | 顺移为 lower-async-tokens 扩展 |
| PR 总数 | 6 | **5** |
| `affineForOuterToSCF` | 需要保留并适配 | **不再需要**（独立 cleanup） |
| sentinel 实现 | 新 `ADORA.null_token` op | **复用 `ADORA.event.create`** |

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

## 0.1 执行模型 + token 的定位（本次澄清）

本仓**没有 runtime 层**。并发由 host 端对 emit 产物（`EmitCGRACall` /
`EmitPytest` / `EmitVitisSDK`）做**静态依赖分析**后决定多 stream issue：
emit 出的指令流携带依赖标注，host 分析器读标注即可决定哪些指令可并行下发。

在此前提下，`!ADORA.token` 的定位是**编译期依赖凭证**，不是 runtime event：

1. **信息保真** — 把编译器在 IR 层做的符号依赖分析（AffineMap + IV shift +
   overlap）结果固化到 SSA 边上，避免 emit 后在已展开/下沉的形态上重算；
2. **多 emit backend 共享** — 三个 emitter 不各自重写依赖分析；
3. **IR pass 间共同词汇** — `RemoveRedundantBlockStoreLoadPair`（已在用 TaskGraph
   邻接表）等 pass 之间的依赖信息交换；
4. **可验证** — pipeline 内 `--verify-each` + FileCheck + `adora.dep_summary ↔
   token graph` 双向 cross-check（`verifyTokensMatchSummary`）。

### token 的生命周期（PR6.x 全景）

```
┌─────────────────────────┐
│ schedule-tasks (PR6.2)  │  产出 !ADORA.token SSA 值 + async [...] operand
│                         │  + affine.for iter_args / affine.yield
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ IR pass 层消费          │  RemoveRedundantBlockStoreLoadPair 等
│                         │  未来 DMA merge / prefetch hoist 等
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ lower-async-tokens      │  把 !ADORA.token SSA 值降级为 op attribute：
│ (PR6.3)                 │    op {adora.wait_ids = [1,2], adora.signal_id = 3,
│                         │        adora.lc_wait_ids = [4], adora.lc_iter_distance = 1}
│                         │  `ADORA.event.create/destroy` 消失
│                         │  `affine.for iter_args(!ADORA.token)` 塌回无 iter_args 形态
└──────────┬──────────────┘
           ↓
┌─────────────────────────┐
│ emit (PR6.4)            │  三个 emitter 只读上述属性，写依赖编号到各自输出格式
│                         │  emit 产物中完全没有 `!ADORA.token` 类型痕迹
└─────────────────────────┘
```

### 默认发射 token（本 PR 一并翻转）

`--adora-schedule-tasks` 此前的 `emit-token` 默认 `false` 属历史遗留。本 PR
**把默认翻转为 `true`**，理由：

1. 当前关掉 token 等于 pipeline 残缺，下游 PR6.3/6.4 都无法工作；
2. 与 MLIR 生态（`gpu.async.token` / `async.token`）一致，token 是默认产物；
3. 保留 option 作为调试 opt-out（回归 pre-PR6 baseline / 对照实验）。

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

## 3. 文件落点（v4.1 — 嵌入式集成）

> **关键设计修正**：不再新增独立 pass。功能**嵌入 `ScheduleADORATasksPass`**。
> 理由：LC 分析结果已在 ScheduleAdoraTasks 内部 in-memory 存在，拆独立 pass
> 会把 `adora.lc_dep_summary`（调试 dump）被迫升格为 pass 间稳定契约。
>
> 同时借机清理 `ScheduleAdoraTasks.cpp`（当前 822 行，混有死代码 + 分析 + 改写）：
> 纯分析 / 验证 / 小工具迁出到 `Analysis/` 或 `Utility/`，保留 IR 改写主线。

### 分三类动作

#### A. 删除 `ScheduleAdoraTasks.cpp` 内的死代码（~60 行）
- `generateTaskGraphFromBlock` 里被注释掉的 load-after-load 残块（约 line 117-125）
- 函数末尾注释掉的"first, load-after-store"整块（约 line 150-165）
- `ScheduleADORATasksInFunction` 开头对 `affineForOuterToSCF(func, 1)` 的调用
  — v4 决策已弃用；函数本体仍留 `ADORAToSCF.cpp` 供他处可能使用

#### B. 迁出到 `lib/Dialect/ADORA/Analysis/`（纯分析 / 验证）
| 当前位置 | 迁到 | 备注 |
|---|---|---|
| `ScheduleAdoraTasks.cpp:185 analyzeDependencyInGraph` | `Analysis/TaskGraphDepAnalysis.cpp` | O(N²) RAW/WAR/WAW 边分析，纯只读 |
| `ScheduleAdoraTasks.cpp:578 verifyTokensMatchSummary` | `Analysis/TaskGraphDepAnalysis.cpp` | CI cross-check，纯只读 |
| `ScheduleAdoraTasks.cpp:683 collectEnclosingLoopsWithKernel` | `Analysis/LoopCarriedDep.cpp` | 已是 PR6.1 配套的 loop 枚举工具 |
| `ScheduleAdoraTasks.cpp:263 appendDepEdgesToAttrList` | `Analysis/DepSummaryView.cpp`（已存在）| P4.0 序列化到 `adora.dep_summary` |
| `ScheduleAdoraTasks.cpp:437 dumpTokenGraphAsDot` | `Analysis/DepSummaryView.cpp` | 调试可视化 |

新增声明头文件：
- `include/ADORA/Dialect/ADORA/Analysis/TaskGraphDepAnalysis.h`
  — `analyzeDependencyInGraph(TaskGraph*)` + `verifyTokensMatchSummary(TaskGraph*)`

#### C. 迁出到 `lib/Dialect/ADORA/Utility/`（小工具）
| 当前位置 | 迁到 |
|---|---|
| `ScheduleAdoraTasks.cpp:298-326 loadBuiltInAttrs / storeBuiltInAttrs / kernelBuiltInAttrs / migrateAttrs` | `Utility/Utility.cpp` 或新增 `Utility/AttrMigration.cpp` |

#### D. 保留在 `ScheduleAdoraTasks.cpp`（IR 改写主线）
- `generateTaskGraphFromBlock`（TaskGraph 构建，紧贴 schedule 主线）
- `rebuildAsyncLoad / rebuildAsyncStore / rebuildAsyncKernel`（改 async deps，IR 改写）
- `threadTokensOnDMAs`（块内 intra-iter token 编织，PR2 核心）
- `RemoveRedundantBlockStoreLoadPair / RemoveRedundantBlockLoads`（冗余消除）
- `ScheduleADORATasksPass` 类 + `ScheduleADORATasksInFunction`

#### E. 新增 PR6.2 核心实现（嵌入 ScheduleAdoraTasks.cpp 或就近独立文件）
- `lib/Dialect/ADORA/Transforms/ThreadLoopCarriedTokensImpl.cpp`
  导出单函数：
  ```cpp
  LogicalResult threadLoopCarriedTokensOnAffineFor(
      affine::AffineForOp forOp,
      const analysis::LoopCarriedDepResult& result);
  ```
  由 `ScheduleAdoraTasksInFunction` 在 LC 分析拿到结果后**立即**调用。
- 声明放 `include/ADORA/Dialect/ADORA/Transforms/ThreadLoopCarriedTokens.h`
- `groupEdgesIntoChains` 放 `Analysis/LoopCarriedDep.{h,cpp}`（纯逻辑，属于分析）

#### F. Pass option（开关 + 回滚通道）
在 `Passes.td` 的 `ScheduleADORATasks` 加 / 改：
```tablegen
Option<"threadLCTokens", "thread-lc-tokens", "bool", /*default=*/"true",
       "Rewrite affine.for with iter_args/affine.yield to carry "
       "!ADORA.token across iterations (PR6.2).">,
// 本 PR 同时翻转已有 emit-token 默认值 false → true，option 本身保留：
Option<"emitToken", "emit-token", "bool", /*default=*/"true",
       "Emit !ADORA.token SSA values along async dependency edges. "
       "Default true; set false only to reproduce pre-PR6 baseline.">
```

`experiment/taskschedule/04_gemm_tiled/run.sh` 与
`experiment/taskschedule/05_loop_carried/run.sh` 中显式写的 `emit-token=true`
参数在默认翻转后可精简移除（保留亦向后兼容）。

### 不动
- `lib/Dialect/ADORA/Lowering/ADORAToSCF.cpp`（`affineForOuterToSCF` 留文件里，
  仅从 ScheduleAdoraTasks 的调用点移除）
- 现有 TaskGraph 数据结构 / API（瘦身留给独立 PR）
- PR6.1 属性 `adora.lc_dep_summary` 生成逻辑（保留为调试 dump，本 PR 不依赖它）

---

## 4. 集成位点

在 `ScheduleAdoraTasks.cpp::ScheduleADORATasksInFunction` 中，PR6.1 的 LC 分析
循环内**立即**调用改写：

```cpp
if (emitSummary || threadLCTokens) {
  SmallVector<mlir::Attribute> lcAttrs;
  int loopIdx = 0;
  for (Operation *loopOp : analysis::collectEnclosingLoopsWithKernel(func)) {
    auto r = analysis::analyzeLoopCarriedDeps(loopOp);
    if (r.empty()) { loopIdx++; continue; }

    // dump (可选)
    if (emitSummary)
      lcAttrs.push_back(
          analysis::serializeLoopCarriedDeps(r, loopIdx, func.getContext()));

    // PR6.2 — 原地 thread token
    if (threadLCTokens) {
      if (auto forOp = dyn_cast<affine::AffineForOp>(loopOp)) {
        if (failed(threadLoopCarriedTokensOnAffineFor(forOp, r))) {
          func.emitError("adora: failed to thread loop-carried tokens");
          signalPassFailure();
          return;
        }
      }
    }
    loopIdx++;
  }
  if (emitSummary && !lcAttrs.empty())
    func->setAttr("adora.lc_dep_summary",
                  ArrayAttr::get(func.getContext(), lcAttrs));
}
```

**重要顺序约束**：本步骤必须在 `threadTokensOnDMAs` 之后、在 `adora.dep_summary`
序列化之后。因为：
- `threadTokensOnDMAs` 会改 DMA 的 `async [...]` 列表，本 pass 还要再 append `%lc_tok`
- `adora.dep_summary` 序列化读的是 TaskGraph 的 dep edges，与 affine.for iter_args
  无关，可以更早做

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

### 5.5 默认开关回归（新增，对应 §0.1 默认翻转）

`test/ADORA/Transforms/schedule_tasks_default_emit_token.mlir`：
- **不传** `emit-token` / `thread-lc-tokens`，直接跑 `cgra-opt --adora-schedule-tasks`
- 期望默认就织出 `async [%tok]` + `iter_args(!ADORA.token)`

```mlir
// RUN: cgra-opt --adora-schedule-tasks %s | FileCheck %s
// CHECK: async [%{{.*}}]
// CHECK: iter_args({{.*}}: !ADORA.token)
// CHECK: affine.yield %{{.*}} : !ADORA.token
```

另加一条反向：`--adora-schedule-tasks='emit-token=false'` 仍能产出 pre-PR6
形态（CHECK-NOT token），保证 opt-out 通道畅通。

整链 smoke：`tools/adoracc/adoracc.py` 跑 04 任意 `.mlir` 不报 verifier
error（确认默认开关翻转没把下游 pass 击穿）。

---

## 6. 风险 & 限制

| 风险 | 说明 | 应对 |
|---|---|---|
| 一对 op 多 kind 边产生多 token | 首版每 unique (src,dst) 一条 chain，可能仍然过保守 | 后续优化，保留 TODO |
| body 内有非 ADORA op 持有 token | 不应该发生（token 只来自 ADORA ops），断言保护 | `assert isa<ADORA::TokenType>(producer.token.getType())` |
| `affine.for` 已经有 iter_args（非 token） | 当前 codebase 不存在这种情况；首版断言不允许 | TODO：未来支持时把 inits 拼到末尾 |
| sentinel token 的语义 | `ADORA.event.create` 产的 token 必须 "signaled-at-birth"，否则第 0 次迭代会等死 | PR3 已定义此语义；如未定义需补 attr，详 PR3 |
| `affineForOuterToSCF` 仍可能在 pipeline 内被调用 | 仅作用于"包住 kernel 的最外层"，与本 pass 处理的内层 loop 互不相干 | 跑通后再独立 cleanup PR |
| `emit-token` 默认翻转为 true | 下游 pass / emitter 若对 `!ADORA.token` 类型未声明合法性，可能新触发 verifier error | pipeline smoke（adoracc 跑 04/05）+ `--verify-each`；如击穿，给相应 op / region 补 TypeInterface 而非回滚开关 |
| PR6.3 / PR6.4 尚未落地时默认发 token | 目前已有消费者仅 `threadTokensOnDMAs` 自身 + `RemoveRedundantBlockStoreLoadPair`（不读 token，读 TaskGraph），整链可编可跑 | 验证：04_gemm_tiled 完整 adoracc pipeline 不回归；emit 阶段因未 lower 而忽略 async operand 属于预期（由 PR6.3 填坑）|

---

## 7. 验收清单

- [ ] `cgra-opt --adora-thread-loop-carried-tokens` 注册可见
- [ ] 04_gemm_tiled / 05_loop_carried run.sh + FileCheck 全绿
- [ ] 06_no_lc（可选）no-op 验证通过
- [ ] `ninja cgra-opt` 全量编译干净（no warning regression）
- [ ] commit message 引用 `f47aff1` (v4 plan) + `docs/affine_for_yield_token_verification.md`

---

## 8. 单 PR 交付（不拆）

按用户指示"不好拆就不拆了"，单 PR 一次交付，内容：
1. ScheduleAdoraTasks.cpp 死代码清理 + 调用 `affineForOuterToSCF` 移除
2. 迁出 analyze/verify/collect/dump/attr-migrate 到 Analysis/ 与 Utility/
3. `Analysis/LoopCarriedDep.{h,cpp}` 新增 `groupEdgesIntoChains` + `LCChain`
4. 新增 `Transforms/ThreadLoopCarriedTokensImpl.cpp` + 配套头
5. `Passes.td` 加 `threadLCTokens` option
6. `ScheduleAdoraTasksInFunction` 集成调用
7. FileCheck tests（04/05）
8. 编译 + run 全绿

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

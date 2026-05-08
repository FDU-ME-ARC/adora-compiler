# PR2 commit B / C 实施计划（定稿 · 待 review · 基于真实代码现状）

> 本文档是 PR2 剩余工作（commit B：pass threading；commit C：canonical + cross-check）
> 的可开工级别计划。review 通过后逐 commit 落地。
>
> 相关背景见 `docs/async_token_design.md` 的 PR1/PR2/PR3 章节与 "现状 & 下一步"。

---

## 1. 现状盘点（本次核验发现）

| 资产 | 状态 | 依据 |
|---|---|---|
| `TaskNode::getOperation/setOperation` | ✅ **已有**（非 virtual，但已足够） | `TaskNode.h:48-50` |
| `TaskGraph::depEdges() / addDepEdge` | ✅ 已有 | `TaskGraph.h:64-65` |
| `DataBlockLoadOp::build(..., asyncDeps, produceToken)` | ✅ 已有 | `ADORAOps.cpp:78-102` |
| `DataBlockStoreOp::build(..., asyncDeps, produceToken)` | ✅ 已有 | `ADORAOps.cpp:420-442` |
| `KernelOp::build(name, asyncDeps, produceToken)` | ⚠️ **有但不够用** —— 只建空 body | `KernelOp.cpp:49-64` |
| `ADORAOps.td` BlockLoad/Store async 字段 | ✅ 已有 | `ADORAOps.td:84-93, 227-235` |
| `assemblyFormat` 的 `async [...]` 语法 | ❌ 未配置（仅注释里出现 `async`） | grep 验证 |
| `Utility.h` accessor | ❌ 未补 | grep 零结果 |
| `ScheduleADORATasksBase` option | ❌ 未定义 | `Passes.td` 需改 |
| `threadTokensOnDMAs` / `rebuildAsync*` | ❌ 未实现 | grep 零结果 |
| Canonical `DropUnusedAsyncToken` / `DedupAsyncDeps` | ❌ 未实现 | grep 零结果 |
| Lit `schedule_cgra_tasks_tokens.mlir` | ❌ 未新增 | 文件不存在 |

**阻塞项**：现有 `KernelOp` 的 async builder 直接 `new Block` 建空 body，无法承载 rebuild（需要 `takeBody`）。必须新增一个带 body-take 的 rebuild-friendly builder，或改在 B 里直接 in-place mutate。

---

## 2. 关键决策（请 review）

### 决策 A：Kernel 如何从同步形态升级到异步形态？

**方案 A1：in-place mutate**
- 用 `op->insertOperands(op->getNumOperands(), asyncDeps)` 把 token operand 追加到末尾。
- 更新 `operandSegmentSizes` attr。
- ⚠️ MLIR 不支持在 op 上动态 append result；因此 **只能在 in-place 追加 operand**；若 `produceToken=true` 则必须 **rebuild**。

**方案 A2：统一 rebuild**（推荐）
- 新增 `KernelOp::build(name, asyncDeps, produceToken, bodyOwner)` 支持 `takeBody`。
- 统一用 `rebuildAsyncKernel`，与 Load/Store 对称。

**推荐**：**A2**。理由：Load/Store 必须 rebuild（因为 result 可能新增 token），Kernel 用同样范式可读性一致。KernelOp 的 IsolatedFromAbove + region 搬移由 `Region::takeBody` 原生支持。

### 决策 B：Load/Store 升级路径

Load/Store 也存在"只加 operand 不加 result"的场景（即 `produceToken=false` 只加 deps）。仍然统一走 rebuild 以保持范式一致。

### 决策 C：Pass option 接入方式

目前 `ScheduleADORATasksPass : public ScheduleADORATasksBase<...>`。option 必须加到 `Passes.td` 的 `def ScheduleADORATasks : Pass<...>` 里：

```tablegen
def ScheduleADORATasks : Pass<"adora-schedule-cgra-tasks", "ModuleOp"> {
  let options = [
    Option<"emitTokens",  "emit-token",    "bool", "true",  "Thread SSA !ADORA.token on DMAs/Kernels">,
    Option<"emitSummary", "emit-summary",  "bool", "true",  "Emit adora.dep_summary DictionaryAttr">,
    Option<"crossCheck",  "cross-check-summary-vs-token", "bool", "false",
           "Assert summary edges match SSA token edges (CI-only)">
  ];
}
```

### 决策 D：assemblyFormat 的 `async [...]` 语法

当前 `.td` 无自定义 assemblyFormat（应该走默认 generic form）。若 commit B 产出 IR 里含 `asyncToken`，打印会是 `"ADORA.datablock_load"(...) : ... -> (..., !ADORA.token)` 这种 generic form——**语法上合法，能 roundtrip**，但可读性差。

**推荐**：commit B 不动 printer/parser，先让 pass 跑通；commit C 之后若有时间再补 custom assemblyFormat（可作 A' 小 PR）。

---

## 3. commit B 详细实施（按文件）

### 3.1 `ADORAKernelOp.td` + `KernelOp.cpp`：新增 rebuild-friendly builder

```tablegen
// ADORAKernelOp.td — add a builder that lets rebuilder hand in a region.
let builders = [
  OpBuilder<(ins)>,                                           // 既有
  OpBuilder<(ins "std::string":$name)>,                       // 既有
  OpBuilder<(ins "std::string":$name,
                 "ValueRange":$asyncDeps,
                 "bool":$produceToken)>,                      // 既有（建空 body）
  OpBuilder<(ins "std::string":$name,
                 "ValueRange":$asyncDeps,
                 "bool":$produceToken,
                 "Region*":$takeFromRegion)>                  // PR2 新增
];
```

```cpp
// KernelOp.cpp
/// Rebuild-friendly async builder: constructs a new KernelOp carrying
/// `asyncDeps` and optionally producing an asyncToken, while *taking over*
/// the body region from `takeFromRegion` (typically the old KernelOp we
/// are replacing). Caller is responsible for RAUWing old results.
void KernelOp::build(OpBuilder &b, OperationState &result,
                     std::string name,
                     ValueRange asyncDeps, bool produceToken,
                     Region *takeFromRegion) {
  result.addOperands(asyncDeps);
  if (produceToken)
    result.addTypes(TokenType::get(b.getContext()));
  if (!name.empty())
    result.addAttribute(getKernelNameAttrStr(), b.getStringAttr(name));
  Region *kernelRegion = result.addRegion();
  if (takeFromRegion)
    kernelRegion->takeBody(*takeFromRegion);
  else {
    Block *body = new Block();
    for (unsigned i = 0; i < kNumConfigRegionAttributes; ++i)
      body->addArgument(b.getIndexType(), result.location);
    kernelRegion->push_back(body);
  }
}
```

### 3.2 `Utility.h` accessor

```cpp
// ADORA/Utility/Utility.h — append at the bottom of namespace mlir::ADORA

/// Returns the async token produced by `op`, or a null Value if `op` is
/// either not async-capable or is currently in its synchronous form.
inline Value getAsyncTokenOrNull(Operation *op) {
  if (auto l = dyn_cast_or_null<DataBlockLoadOp>(op))   return l.getAsyncToken();
  if (auto s = dyn_cast_or_null<DataBlockStoreOp>(op))  return s.getAsyncToken();
  if (auto k = dyn_cast_or_null<KernelOp>(op))          return k.getAsyncToken();
  return {};
}

inline OperandRange getAsyncDeps(Operation *op) {
  if (auto l = dyn_cast<DataBlockLoadOp>(op))   return l.getAsyncDependencies();
  if (auto s = dyn_cast<DataBlockStoreOp>(op))  return s.getAsyncDependencies();
  if (auto k = dyn_cast<KernelOp>(op))          return k.getAsyncDependencies();
  return op->getOperands().take_front(0);  // empty OperandRange fallback
}

inline bool isAsyncCapable(Operation *op) {
  return isa_and_nonnull<DataBlockLoadOp, DataBlockStoreOp, KernelOp>(op);
}
```

### 3.3 `ScheduleAdoraTasks.cpp`：三个 `rebuildAsync*` + `threadTokensOnDMAs`

```cpp
//===----------------------------------------------------------------------===//
// PR2 commit B — thread SSA !ADORA.token values through DMA/Kernel ops so
// that dependency edges discovered by analyzeDependencyInGraph become first
// class MLIR happens-before relations, replacing the PR1 adora.dep_summary
// string path when `emit-token` option is on.
//===----------------------------------------------------------------------===//

/// Attribute names that are already written by the op's builder; we skip
/// them during attribute migration to avoid double-setting.
static const llvm::StringSet<> kLoadBuiltInAttrs = {
    "map", "strides", "kernel_name", "operandSegmentSizes"};
static const llvm::StringSet<> kStoreBuiltInAttrs = {
    "map", "strides", "kernel_name", "operandSegmentSizes"};
static const llvm::StringSet<> kKernelBuiltInAttrs = {
    "kernel_name", "operandSegmentSizes"};

/// Migrate every user-set attribute (pingpong, tile_id, schedule_hint, etc.)
/// from the old op to the newly built async op.
static void migrateAttrs(Operation *oldOp, Operation *newOp,
                         const llvm::StringSet<> &builtIns) {
  for (NamedAttribute a : oldOp->getAttrs())
    if (!builtIns.contains(a.getName().strref()))
      newOp->setAttr(a.getName(), a.getValue());
}

/// Rebuild `old` in its async form. The old op is erased; its memref result
/// is RAUW'd onto the new op's result so downstream users keep working.
/// Returns { newAsyncToken (null if !produceTok), newOp }.
static std::pair<Value, Operation *>
rebuildAsyncLoad(ADORA::DataBlockLoadOp old, ValueRange deps, bool produceTok) {
  OpBuilder b(old);
  auto mapAttr = old->getAttrOfType<AffineMapAttr>(
                     ADORA::DataBlockLoadOp::getMapAttrStr());
  auto strides = old->getAttrOfType<DenseI64ArrayAttr>("strides");
  std::string kern = old.getKernelName().str();
  SmallVector<Value> mapOps(old.getIndices().begin(), old.getIndices().end());

  auto newOp = b.create<ADORA::DataBlockLoadOp>(
      old.getLoc(), old.getOriginalMemref(),
      mapAttr.getValue(), mapOps,
      old.getResult().getType().cast<MemRefType>(),
      strides, kern, deps, produceTok);

  migrateAttrs(old, newOp, kLoadBuiltInAttrs);
  old.getResult().replaceAllUsesWith(newOp.getResult());
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : Value(), newOp.getOperation()};
}

static std::pair<Value, Operation *>
rebuildAsyncStore(ADORA::DataBlockStoreOp old, ValueRange deps, bool produceTok) {
  OpBuilder b(old);
  auto mapAttr = old->getAttrOfType<AffineMapAttr>(
                     ADORA::DataBlockStoreOp::getMapAttrStr());
  auto strides = old->getAttrOfType<DenseI64ArrayAttr>("strides");
  std::string kern = old.getKernelName().str();
  SmallVector<Value> mapOps(old.getIndices().begin(), old.getIndices().end());

  auto newOp = b.create<ADORA::DataBlockStoreOp>(
      old.getLoc(), old.getSourceMemref(), old.getTargetMemref(),
      mapAttr.getValue(), mapOps, strides, kern, deps, produceTok);

  migrateAttrs(old, newOp, kStoreBuiltInAttrs);
  // Store has no memref result → nothing to RAUW.
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : Value(), newOp.getOperation()};
}

static std::pair<Value, Operation *>
rebuildAsyncKernel(ADORA::KernelOp old, ValueRange deps, bool produceTok) {
  OpBuilder b(old);
  std::string name = old.getKernelName().str();

  // Take body out of old so the new op inherits the region verbatim.
  auto newOp = b.create<ADORA::KernelOp>(
      old.getLoc(), name, deps, produceTok, &old.getBody());

  migrateAttrs(old, newOp, kKernelBuiltInAttrs);
  // KernelOp currently has no data results to RAUW; if it had, we would
  // map them positionally here (token stays last).
  old.erase();
  return {produceTok ? newOp.getAsyncToken() : Value(), newOp.getOperation()};
}

/// Thread tokens along dep edges recorded by analyzeDependencyInGraph.
///
/// Algorithm:
///   1. Collect preds[dst] = [src...] and the fan-out set `hasOut`.
///   2. Sort participating ops by lexical order (PR1 guarantees DAG).
///   3. Rebuild each op into its async form, wiring incoming tokens and
///      producing an outgoing token iff the op has any outgoing edge.
///   4. Patch TaskNode back-pointers so graph metadata stays valid.
static void threadTokensOnDMAs(TaskGraph *graph) {
  llvm::DenseMap<Operation *, SmallVector<Operation *>> preds;
  llvm::DenseSet<Operation *> hasOut;
  for (const auto &e : graph->depEdges()) {
    Operation *s = e.src ? e.src->getOperation() : nullptr;
    Operation *d = e.dst ? e.dst->getOperation() : nullptr;
    if (!s || !d || s == d) continue;
    preds[d].push_back(s);
    hasOut.insert(s);
  }
  if (preds.empty() && hasOut.empty()) return;

  llvm::SetVector<Operation *> all;
  for (auto &kv : preds) {
    all.insert(kv.first);
    for (auto *s : kv.second) all.insert(s);
  }
  for (auto *s : hasOut) all.insert(s);

  SmallVector<Operation *> ordered(all.begin(), all.end());
  llvm::sort(ordered, [](Operation *a, Operation *b) {
    if (a->getBlock() == b->getBlock()) return a->isBeforeInBlock(b);
    return a < b;  // different blocks: stable but arbitrary; not reached in practice.
  });

  llvm::DenseMap<Operation *, Value>      tokens;     // old op -> new token
  llvm::DenseMap<Operation *, Operation *> oldToNew;  // for TaskNode patch-up

  for (Operation *op : ordered) {
    // 3a. Assemble dedup'd dep list from already-rebuilt predecessors.
    llvm::SmallSetVector<Value, 4> depSet;
    if (auto it = preds.find(op); it != preds.end()) {
      for (Operation *p : it->second) {
        if (p == op) continue;
        if (Value t = tokens.lookup(p)) depSet.insert(t);
      }
    }
    SmallVector<Value> deps(depSet.begin(), depSet.end());
    bool produce = hasOut.contains(op);

    std::pair<Value, Operation *> rebuilt;
    if (auto l = dyn_cast<ADORA::DataBlockLoadOp>(op))
      rebuilt = rebuildAsyncLoad(l, deps, produce);
    else if (auto s = dyn_cast<ADORA::DataBlockStoreOp>(op))
      rebuilt = rebuildAsyncStore(s, deps, produce);
    else if (auto k = dyn_cast<ADORA::KernelOp>(op))
      rebuilt = rebuildAsyncKernel(k, deps, produce);
    else
      continue;  // e.g. LocalMemAllocOp — not async-capable, skip

    if (rebuilt.first)  tokens[op]   = rebuilt.first;
    if (rebuilt.second) oldToNew[op] = rebuilt.second;
  }

  // 4. Patch TaskNode back-pointers; erased ops would otherwise dangle.
  for (TaskNode *n : graph->getAllNodes()) {
    auto it = oldToNew.find(n->getOperation());
    if (it != oldToNew.end()) n->setOperation(it->second);
  }
}
```

### 3.4 `ScheduleAdoraTasks.cpp` 接入点

```cpp
void ScheduleADORATasksPass::runOnOperation() {
  ...
  for (auto func : module.getOps<func::FuncOp>()) {
    ...
    for (auto *block : /* blocks containing KernelOp */) {
      TaskGraph graph;
      generateTaskGraphFromBlock(&graph, block);
      analyzeDependencyInGraph(&graph);
      if (emitTokens)  threadTokensOnDMAs(&graph);       // PR2 commit B
      if (emitSummary) emitDepSummaryAttr(func, &graph); // PR1
      if (crossCheck)  verifyTokensMatchSummary(&graph); // PR2 commit C
    }
  }
  module->setAttr("adora.scheduled", UnitAttr::get(&getContext())); // PR1
}
```

### 3.5 `Passes.td` options

见"决策 C"代码段。

### 3.6 Lit `test/cgra-opt/kernel/schedule_cgra_tasks_tokens.mlir`

```mlir
// RUN: cgra-opt %s --adora-schedule-cgra-tasks | FileCheck %s
// RUN: cgra-opt %s --adora-schedule-cgra-tasks=emit-token=false | \
// RUN:     FileCheck %s --check-prefix=NOTOKEN

// CHECK-LABEL: func.func @chain
// CHECK: %[[T0:.*]] = "ADORA.datablock_load"
// CHECK-SAME: -> (memref<{{.*}}, 2>, !ADORA.token)
// CHECK: "ADORA.kernel"(%[[T0]])
// CHECK-SAME: -> !ADORA.token
// CHECK: "ADORA.datablock_store"(%{{.*}}, %{{.*}}, %[[TK:.*]])

// NOTOKEN-NOT: !ADORA.token
// NOTOKEN: adora.dep_summary
```

### 3.7 回归 `schedule_tasks_dep_summary.mlir`

无需改——pass 默认 `emit-token=true`，会让 IR 多出 `!ADORA.token`；但该 lit 只 CHECK `adora.dep_summary` 与 `adora.scheduled`，不会误伤。若 CHECK 失败，补一个 `emit-token=false` 的 RUN 行即可。

---

## 4. commit C 详细实施

### 4.1 `ADORAOps.cpp` — 两个 canonical pattern

```cpp
namespace {

/// Drop an unused asyncToken: if nobody consumes it, rebuild the op in
/// synchronous form. Reduces IR noise after aggressive DCE / inlining.
template <typename OpTy>
struct DropUnusedAsyncToken : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rw) const override {
    Value tok = op.getAsyncToken();
    if (!tok || !tok.use_empty()) return failure();
    // NOTE: rebuild helpers live in ScheduleAdoraTasks.cpp; duplicate the
    //       minimal inline version here to keep patterns self-contained.
    // [rebuild boilerplate identical to §3.3 but with produceTok=false]
    return success();
  }
};

/// Remove duplicate or self entries from asyncDependencies.
template <typename OpTy>
struct DedupAsyncDeps : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rw) const override {
    auto deps = op.getAsyncDependencies();
    llvm::SmallSetVector<Value, 4> uniq;
    for (Value v : deps)
      if (v.getDefiningOp() != op.getOperation()) uniq.insert(v);
    if (uniq.size() == deps.size()) return failure();
    rw.updateRootInPlace(op, [&] {
      op.getAsyncDependenciesMutable().assign(
          SmallVector<Value>(uniq.begin(), uniq.end()));
      // sync operandSegmentSizes
      // [segment update boilerplate]
    });
    return success();
  }
};

} // namespace

void DataBlockLoadOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *ctx) {
  results.add<DropUnusedAsyncToken<DataBlockLoadOp>,
              DedupAsyncDeps<DataBlockLoadOp>>(ctx);
}
// 同样为 DataBlockStoreOp / KernelOp 各挂一份
```

> `let hasCanonicalizer = 1;` 需在 `.td` 中三个 op 上补。

### 4.2 `verifyTokensMatchSummary`（cross-check）

```cpp
/// Walk graph edges + the DictionaryAttr summary side-by-side and assert
/// they describe the same set of (src, dst) pairs. CI-only; emits a fatal
/// pass failure in cross-check mode, a warning otherwise.
static LogicalResult verifyTokensMatchSummary(TaskGraph *graph /*, func FuncOp */) {
  // 1. Reconstruct expected edges from graph->depEdges() as unordered set.
  // 2. For each op in order, walk getAsyncDependencies() → defining op
  //    pairs, building actual edges.
  // 3. Diff the two sets; on mismatch either signalPassFailure or
  //    emitWarning at module level.
}
```

---

## 5. 命令与回归

```bash
# 构建
cd build && ninja

# 新 lit
llvm-lit -v test/cgra-opt/kernel/schedule_cgra_tasks_tokens.mlir

# NFC 回归证明（emit-token=false 下与 main 一致）
cgra-opt old.mlir --adora-schedule-cgra-tasks=emit-token=false > a.mlir
cgra-opt old.mlir --adora-schedule-cgra-tasks=emit-token=false > b.mlir # from main
diff a.mlir b.mlir     # 期望：空

# 全量
llvm-lit test/
```

---

## 6. 风险与兜底

| 风险 | 缓解 |
|---|---|
| `Region::takeBody` 在带 IsolatedFromAbove 的 KernelOp 上破坏外部捕获 | PR2 下游 kernel 已 isolated 且 inputs/outputs 非 operand，takeBody 仅搬 region，不动外部 SSA → 安全 |
| `operandSegmentSizes` 长度与 segment 数量不匹配导致 verify 失败 | rebuildAsync* 只 `result.addOperands(...)` + 统一 `operandSegmentSizes` 一次写入；通过新建 op 天然避免 out-of-sync |
| 现有 `.mlir` 测试隐含 CHECK 依赖 sync-form 打印 | pass option `emit-token` 默认 on，但存在风险；若 integration 红，改默认 off → 先只新 lit 覆盖，老 lit 保持 sync 打印 |
| `DropUnusedAsyncToken` 与 `threadTokensOnDMAs` 幂等性 | canonical 在 `--canonicalize` 里跑；`threadTokensOnDMAs` 只在 pass 内跑一次；两者不冲突 |

**默认策略建议**：commit B 初稿把 `emit-token` 默认设为 `false`，只在新 lit 用 `=true` 打开；稳定一轮后再翻转默认值，最稳。

---

## 7. 注释规范（commit B/C 必须遵守）

- 全英文 `//` / `///`，解释 **意图 / invariant / why**，不复述代码。
- 文件顶部 2–4 行 block 注释定位模块。
- `rebuildAsync*` 用 `///` doxygen 标 `@param / @return / side-effects: erases old op`。
- 算法主循环用 `// 1. / 2. / 3.` 分段。
- 易错点用 `// NOTE:` / `// WHY:`；TODO 带 scope，如 `TODO(PR3): lower to adora.event.wait`。

---

## 8. commit 切片与时间表

| commit | 内容 | 预期工时 | 预期 lit |
|---|---|---|---|
| **B.1** | §3.1 Kernel 新 builder + §3.2 Utility accessor | 0.5 日 | 老 lit 全绿 |
| **B.2** | §3.3 `rebuildAsync*` + `threadTokensOnDMAs` + §3.5 options + §3.6 新 lit | 1 日 | 新 lit 绿 + 老 lit 在 `emit-token=false` 绿 |
| **C.1** | §4.1 两 canonical pattern + `hasCanonicalizer` | 0.5 日 | `--canonicalize` 幂等 |
| **C.2** | §4.2 cross-check + 文档"PR2 合入总结" | 0.5 日 | CI `cross-check=true` 零警告 |

总工期：**2.5 日**。

---

## 9. 验收 checklist

```
[ ] Kernel rebuild-friendly builder + takeBody 路径
[ ] Utility.h 三个 accessor
[ ] rebuildAsyncLoad / Store / Kernel 实现（含 doxygen 英文注释）
[ ] threadTokensOnDMAs 实现
[ ] Passes.td emit-token / emit-summary / cross-check options
[ ] schedule_cgra_tasks_tokens.mlir 新增并通过
[ ] schedule_tasks_dep_summary.mlir 回归（emit-token=false 字节一致）
[ ] DropUnusedAsyncToken / DedupAsyncDeps 两个 canonical
[ ] verifyTokensMatchSummary cross-check
[ ] docs/async_token_design.md 追加 "PR2 合入总结" 小节
[ ] ninja 全绿 + llvm-lit 全绿
```

---

## 10. 需要 review 的决策点

1. **决策 A**：KernelOp 走 **rebuild + takeBody**（A2）还是 **in-place mutate**（A1）？ → 推荐 **A2**
2. **默认策略**：commit B 初稿 `emit-token` 默认值 —— `true`（激进，推动生态）还是 `false`（保守，老 lit 零风险）？ → 推荐 **初稿 `false`，稳定后翻转**
3. **assemblyFormat `async [...]` 自定义语法** —— PR2 内补还是挪到 A' 小 PR？ → 推荐 **挪到 A'**，PR2 走 generic form
4. **`hasCanonicalizer` 打开时机** —— commit C 打开还是继续压到 PR3？ → 推荐 **commit C 打开**
5. **cross-check 行为** —— 失败 `signalPassFailure` 还是 `emitWarning`？ → 推荐 **CI（`cross-check=true`）fail；开发默认 off**

---

## 11. Review 通过后下一步

1. 按第 10 节逐点确认决策。
2. 我按 §8 顺序落地：B.1 → B.2 → C.1 → C.2。
3. 每个 commit 独立可回滚；每个 commit 合入前跑 `ninja && llvm-lit test/`。
4. 合入完成后，在 `docs/async_token_design.md` 追加 "PR2 合入总结"，记录实际 commit hash、benchmark、未决事项。

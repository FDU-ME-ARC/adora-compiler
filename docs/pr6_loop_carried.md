# PR6.1 — Loop-Carried Dependency Analysis (Phase 6.1)

> **Status**: 代码完成，待环境恢复后构建+lit 验证
> **Branch**: `jhlou/pr6-loop-carried-analysis`
> **Base**: `jhlou/scheduletasks` @ `9e821ff` (PR5: scf.for emit support)

---

## 1. 目标

为 `--adora-schedule-tasks` pass 增加 **跨迭代依赖分析**，识别外层 / 内层循环（scf.for, affine.for）相邻迭代之间通过同一 DRAM 区域产生的 RAW/WAR/WAW/RAR 边，并将结果序列化到 FuncOp 属性 `adora.lc_dep_summary`，供下游消费：

- **PR6.2**：决定哪些 affine.for 需要 promote 到 scf.for
- **PR6.3**：`threadLoopCarriedTokens` —— 真正插入 `scf.for iter_args(!ADORA.token)`
- 测试/可视化/CI 交叉校验

> 本 PR **只产分析结果，不动 IR**。

---

## 2. 完成的工作（按 todo 顺序）

### 2.1 调研 dep_summary emission pipeline ✅

确认现有依赖分析 pipeline 4 阶段：
```
P0 generateTaskGraphFromBlock    建 TaskNode 图 + SSA RAW
P1 analyzeDependencyInGraph      O(N²) 发 WAR/WAW/RAR
P2 threadTokensOnDMAs            按边织 !ADORA.token
P4 appendDepEdgesToAttrList      序列化到 adora.dep_summary
```
所有逻辑集中在 `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`（815 行）。

TaskGraph 邻接表（`_innodes/_outnodes`）大部分死代码，唯一活读取点为 `RemoveRedundantBlockStoreLoadPair`；`KernelNode::addInNode` 有 IR 副作用（写入 `LocalMemAllocOp::addAnotherKernelName`），不能贸然删。故 **PR6.1 不动 TaskGraph**，瘦身留独立 PR。

### 2.2 新建 Analysis 库 + 旧接口迁移 ✅

新增目录 `lib/Dialect/ADORA/Analysis/` 和 `include/ADORA/Dialect/ADORA/Analysis/`：

```
include/ADORA/Dialect/ADORA/Analysis/
├── AccessRegion.h          # PR6.1 新增
├── LoopCarriedDep.h        # PR6.1 新增
├── DepKind.h               # 从 Transforms/TaskGraph/ 迁移
└── DepSummaryView.h        # 从 Transforms/TaskGraph/ 迁移

lib/Dialect/ADORA/Analysis/
├── AccessRegion.cpp        # PR6.1 新增 (~135 行)
├── LoopCarriedDep.cpp      # PR6.1 新增 (~130 行)
├── DepSummaryView.cpp      # 从 Transforms/TaskGraph/ 迁移
└── CMakeLists.txt          # 新 target: MLIRADORAAnalysis
```

旧路径头文件保留为 **shim**（一行 `#include` 转发），下游代码零改动：
```cpp
// include/ADORA/Dialect/ADORA/Transforms/TaskGraph/DepKind.h
#include "ADORA/Dialect/ADORA/Analysis/DepKind.h"
```

**CMake 变更**：
- 根 `CMakeLists.txt` — `install(TARGETS ... MLIRADORAAnalysis ...)`
- `lib/Dialect/ADORA/CMakeLists.txt` — `add_subdirectory(Analysis)`
- `lib/Dialect/ADORA/Transforms/CMakeLists.txt` — LINK_LIBS 加 `MLIRADORAAnalysis`，删 `TaskGraph/DepSummaryView.cpp`

### 2.3 `AccessRegion` — DRAM 区域抽象 ✅

```cpp
struct AccessRegion {
  Operation *op;
  Value memref;
  SmallVector<AffineExpr, 4> startExprs;  // op AffineMap 的 dim-space
  SmallVector<int64_t, 4>   sizes;
  SmallVector<int64_t, 4>   sourceShape;
  SmallVector<Value, 4>     operands;

  static FailureOr<AccessRegion> fromOp(Operation* op);
  static bool isLoad(Operation*);
  static bool isStore(Operation*);

  AccessRegion shiftedByIV(Value iv, int64_t delta) const;
  bool overlapsWith(const AccessRegion& other) const;
  bool sameAs(const AccessRegion& other) const;
};
```

**`shiftedByIV` 算法**：
1. 在 `operands` 中定位 `iv` 的 dim-index `ivPos`
2. 对每个 `startExpr` 调用 `replaceDims({{ivPos, dim[ivPos] + delta}})`
3. 返回新 region；若 `iv` 不在 operands 则恒等返回（说明该维度跨迭代不变 → 必 overlap）

**`overlapsWith` 算法**：
1. 不同 memref → false
2. rank 不同 → 保守 true
3. 双方 startExpr 都能折成 `AffineConstantExpr` 且 sizes 都是静态 → 计算 per-dim 闭区间交集；任何一维不交 → false
4. 否则保守 true（含 symbol 的复杂表达式）

复用了 `lib/Dialect/ADORA/Transforms/DependencyAnalysis.cpp:462` 起的 `canonicalizeAndEqual / mayOverlapBoxesByQuadruple` 思路，逻辑同源。

### 2.4 `LoopCarriedDep` — 主分析入口 ✅

```cpp
enum class LCKind { RAW, WAR, WAW, RAR };
struct LoopCarriedDepEdge {
  Operation *src, *dst, *enclosingLoop;
  LCKind     kind;
  Value      loopIV;
  int64_t    step;
  bool       exact;
};
struct LoopCarriedDepResult {
  Operation *loopOp;
  SmallVector<LoopCarriedDepEdge> edges;
};

LoopCarriedDepResult analyzeLoopCarriedDeps(Operation *loopOp);
Attribute serializeLoopCarriedDeps(const LoopCarriedDepResult&, int loopIdx, MLIRContext*);
```

**算法**：
```
body = loopOp.body
iv   = loopOp.inductionVar
step = loopOp.step               (scf.for 用 arith::ConstantIndexOp 抓取，抓不到 step=1 保守)

dmaOps = [op for op in body if isLoad(op) or isStore(op)]

for src in dmaOps:
  for dst in dmaOps:
    if src.memref != dst.memref: continue
    nextDst = dst.shiftedByIV(iv, step)
    if not src.overlapsWith(nextDst): continue
    emit LoopCarriedDepEdge {
      kind: classify(srcIsStore, dstIsStore),  // RAW/WAR/WAW/RAR
      enclosingLoop, iv, step,
      exact: src.sameAs(nextDst)
    }
```

**支持 scf::ForOp 和 affine::AffineForOp 两种**（helper `getLoopBody/getLoopIV/getLoopStep`）。

### 2.5 Driver 接线 ✅

在 `ScheduleAdoraTasks.cpp` 中：

**删除**：
- `findLoopCarriedStoreLoadPair`（旧 Store→Load 简易桩）
- `wireLoopCarriedToken`（旧 stderr 打印桩）
- driver Step D2（调用上面两个桩的位置）

**新增**：
- 静态 helper `collectEnclosingLoopsWithKernel(func)` —— `func.walk` 找所有含 KernelOp 的 scf.for / affine.for
- 在 `appendDepEdgesToAttrList` setAttr 之后，按需调用 `analyzeLoopCarriedDeps`，收集非空结果到 `adora.lc_dep_summary` ArrayAttr

```cpp
if (emitSummary) {
  SmallVector<Attribute> lcAttrs;
  int loopIdx = 0;
  for (Operation *loopOp : collectEnclosingLoopsWithKernel(func)) {
    auto r = analysis::analyzeLoopCarriedDeps(loopOp);
    if (r.empty()) continue;
    lcAttrs.push_back(analysis::serializeLoopCarriedDeps(r, loopIdx++, ctx));
  }
  if (!lcAttrs.empty())
    func->setAttr("adora.lc_dep_summary", ArrayAttr::get(ctx, lcAttrs));
}
```

**输出格式**（DictionaryAttr per loop）：
```mlir
"adora.lc_dep_summary" = [
  { loop_idx = 0 : i64,
    loop_op  = "scf.for",
    edges = [
      { kind = "LC-RAW", step = 1 : i64, exact = true },
      { kind = "LC-WAW", step = 1 : i64, exact = true },
      ...
    ]
  }
]
```

### 2.6 测试 ✅

**`experiment/taskschedule/04_gemm_tiled/input.mlir`**：在原有 PR2/PR3 的 intra-iter token CHECK 后追加：
```
// CHECK: "adora.lc_dep_summary"
// CHECK-SAME: LC-RAW
// CHECK-SAME: LC-WAW
```

**`experiment/taskschedule/05_loop_carried/`**（新建）：极简 reduction —— 单 memref + 空 AffineMap，相邻迭代必 overlap，必产 LC 边。覆盖 `shiftedByIV` 恒等 fallback 路径。

---

## 3. 关键设计决策

| 决策 | 理由 |
|---|---|
| LC 数据结构独立于 `DataBlockDepEdge` | LC 边有 enclosingLoop/iv/step 三个 intra-iter 没有的字段；混在一个 struct 会产生大量 nullable 字段 |
| LC 边不入 TaskGraph 邻接表 | LC 服务 token threading + 序列化，不服务 redundant elimination；保持 TaskGraph 现状不破坏 RemoveRedundantBlockStoreLoadPair |
| Analysis 库独立成 `MLIRADORAAnalysis` | 纯数据 + 纯函数，未来 mapper / 测试可独立 link，不被迫拖入 Transforms |
| 旧 DepKind / DepSummaryView 迁过来 + shim | 让 Analysis 立刻有"实质"内容，但下游零修改 |
| `shiftedByIV` 用 `replaceDims` 而不动 operands | operands 列表保持稳定，避免 LIVE SSA value 错位 |
| overlap 保守原则 | 任何无法常量折叠的情况返回 true → 不产生假阴性 → 下游 token 多等几个，永远安全 |

---

## 4. 风险 / 未决

| 项 | 等级 | 说明 |
|---|---|---|
| 构建未验证 | 🔴 高 | `build/` 被清理，`MLIR_DIR` 路径丢失；环境恢复后跑 `ninja cgra-opt + lit` 即可验证 |
| `AffineExpr::replaceDims(SmallDenseMap)` API | 🟡 中 | MLIR 18+ 支持该签名；若仓库用旧版可能要改成 `replace(DenseMap)` |
| `getStepAsInt()` API | 🟢 低 | PR5 已经在 `LowerPasses.h` 用过，OK |
| `arith::ConstantIndexOp` 抓 scf.for step | 🟢 低 | 抓不到 fallback step=1，分析保守 |
| LC 序列化字符串格式 | 🟡 中 | CHECK 行为依赖 ArrayAttr/DictionaryAttr 的 MLIR 打印格式，可能需要微调 CHECK-SAME |

---

## 5. 下一步路线图

### PR6.2 — 把带 LC 边的内层 affine.for promote 到 scf.for
读取 `adora.lc_dep_summary`，决定哪些 affine.for 需要 lowering 到 scf.for（为 iter_args 做准备）。

### PR6.3 — `threadLoopCarriedTokens`（真正插 iter_args）
1. 对每个有 LC 边的 scf.for，确定需要 carry 的 token 数（一条 chain 一个）
2. 循环前生成 `ADORA.null_token : !ADORA.token`（PR3 已有 event ops，可复用或新增）
3. 给 scf.for 追加 `iter_args(%lc_tok = %null)`
4. body 内把 `lc_tok` 加到后继 DMA 的 `async [...]` 依赖
5. body 尾把前驱 DMA 的 token 作为 `scf.yield` 操作数

### PR6.4 — `--lower-async-tokens` 扩展
识别 scf.for 的 token iter_args，降级到运行时 handle 传递。

### PR6.5 — emit 层支持 scf.for iter_args
EmitCGRACall / EmitPytest / EmitVitisSDK 三处。

### PR6.6（可选）— Load-after-Load 消除
`adora.lc_dep_summary` 里的 `LC-RAR` 边天然指出可优化的同 tile 重复 load，独立 pass 实现。

### 解耦的清理 PR
TaskGraph 瘦身 —— 重写 `RemoveRedundantBlockStoreLoadPair` → 删 TaskNode 派生类 + 邻接表。

---

## 6. 待办（commit 前）

- [ ] 待 `MLIR_DIR` 环境恢复
- [ ] `cd build && cmake .. && ninja cgra-opt`
- [ ] 手跑 `./build/bin/cgra-opt experiment/taskschedule/04_gemm_tiled/input.mlir --adora-schedule-tasks=emit-token=true`，肉眼确认 `adora.lc_dep_summary` 出现
- [ ] 可能微调 CHECK 行（FileCheck 格式适配）
- [ ] `lit experiment/taskschedule -v` 全过
- [ ] `git commit -m "PR6.1 — loop-carried dep analysis + MLIRADORAAnalysis library"`

---

## 7. 关键文件速查

| 用途 | 文件:行 |
|---|---|
| AccessRegion 接口 | `include/ADORA/Dialect/ADORA/Analysis/AccessRegion.h` |
| AccessRegion 实现 | `lib/Dialect/ADORA/Analysis/AccessRegion.cpp` |
| LoopCarriedDep 接口 | `include/ADORA/Dialect/ADORA/Analysis/LoopCarriedDep.h` |
| LoopCarriedDep 实现 | `lib/Dialect/ADORA/Analysis/LoopCarriedDep.cpp` |
| Driver 接线 | `lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp:670-693, 769-792` |
| 04 测试更新 | `experiment/taskschedule/04_gemm_tiled/input.mlir:35-78` |
| 05 测试新增 | `experiment/taskschedule/05_loop_carried/{input.mlir,run.sh}` |

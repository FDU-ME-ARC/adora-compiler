# DataBlock 级依赖分析 — 设计文档 (P1 + P4)

> 归属: `adora-compiler/lib/Dialect/ADORA/Transforms/TaskGraph`
> 状态: P1 基建 / P4 桥梁初版
> 日期: 2026-05-07
> 关联文档: `docs/gap.md` (缺口 1)

---

## 1. 动机

当前 `ScheduleAdoraTasks.cpp:160-166` 的 `analyzeDependencyInGraph` 是空 stub。
上层多处调度决策（schedule node reorder、pingpong cover、`RemoveRedundantBlockLoads`）都
需要节点两两之间的精确依赖分类（RAW / WAR / WAW / Reuse × Must / May），但现有代码只有
粗粒度布尔 `AccessSameDataBlock`（`DependencyAnalysis.h:246-247`）。

本设计新增 DataBlock 级依赖分析工具 `checkDataBlockAccessDependence`，并把结果沉淀到
`TaskGraph` 的依赖边上，供后续 pass 复用。

---

## 2. 参考实现与致谢

本依赖分析的**架构范式**参考 MLIR Affine Dialect 的 Dependency Analysis 子系统：

| MLIR 源文件（本机路径 `/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/mlir/`） | 借鉴的模式 |
|---|---|
| `lib/Dialect/Affine/Analysis/AffineAnalysis.{h,cpp}` | `checkMemrefAccessDependence` 的外壳流程：提取访问 → 建约束系统 → 加等式"访问位置相同" → 判可行性；`MemRefAccess` 数据结构分层；`DependenceResult / DependenceComponent` 返回结构 |
| `lib/Dialect/Affine/Analysis/Utils.{h,cpp}` | `MemRefRegion` 把循环嵌套内访问投影为 memref 子区域（hyper-rectangle）的思路 |
| `lib/Dialect/Affine/Analysis/AffineStructures.{h,cpp}` | `FlatAffineValueConstraints` 约束系统 API（预留给仿射路径） |
| `lib/Dialect/Affine/Analysis/LoopAnalysis.{h,cpp}` | `getInvariantAccesses` / loop-bounds 提取 |
| `include/mlir/Dialect/Affine/Analysis/AffineAnalysis.h` | 返回结构的公共 API 风格 |

**与原版差异**：
1. 粒度：**DataBlock 级**（`adora.BlockLoad / adora.BlockStore` 的 offsets + sizes
   矩形区域），不是逐元素 `affine.load/store`。
2. 输入 op：`ADORA::DataBlockLoadOp / DataBlockStoreOp`，而非 `affine.load/store`。
3. 输出：简化的 `{DepKind, DepStrength, carrierLoopDepth, distance?}`，不返回
   per-loop `DependenceComponent`。
4. 初版只实现**常量偏移的 hyper-rectangle 路径**；仿射表达式路径预留接口，标
   `TODO(P1.5)`。
5. 保留 `NoDep` + `Reuse` 两个"负依赖"（non-hazard）分类。Reuse 边（两个 Load 访问
   同一 block）在 `RemoveRedundantBlockLoads` 和 pipeline cover 阶段有用，MLIR 原
   版 affine 分析不区分此项。

---

## 3. 数据结构

### 3.1 `DataBlockDep`

```cpp
enum class DepKind : uint8_t {
  NoDep = 0,   // 无依赖，可自由调度
  RAW,         // src 写, dst 读
  WAR,         // src 读, dst 写
  WAW,         // src 写, dst 写
  Reuse        // src/dst 都是 Load 同一 block (非依赖, 用于复用识别)
};

enum class DepStrength : uint8_t {
  Must = 0,    // 访问区域完全重叠或可证必然相交
  May          // 存在潜在重叠但不能证明必然发生
};

struct DataBlockDep {
  DepKind     kind     = DepKind::NoDep;
  DepStrength strength = DepStrength::May;
  int  carrierLoopDepth = -1;   // -1 = loop-independent
  std::optional<llvm::SmallVector<int64_t, 4>> distance = std::nullopt;
};
```

### 3.2 `TaskGraph` 扩展

新增一张依赖边表（不修改原有 `_nodes` 结构）：

```cpp
using DepEdge = std::pair<TaskNode*, TaskNode*>;
struct DepEdgeHash { size_t operator()(const DepEdge&) const; };

class TaskGraph {
  // ... 原有成员 ...
  std::unordered_map<DepEdge, DataBlockDep, DepEdgeHash> _depEdges;
public:
  void addDepEdge(TaskNode* src, TaskNode* dst, DataBlockDep dep);
  const DataBlockDep* getDepEdge(TaskNode* src, TaskNode* dst) const;
  const auto& getAllDepEdges() const { return _depEdges; }
  void clearDepEdges();
};
```

`TaskNode` 本身不加字段（避免破坏序列化），所需 `offsets/sizes` 每次调用时从 op 实时
提取。若后续 P2 证明有缓存必要再加。

---

## 4. `checkDataBlockAccessDependence` 算法

### 4.1 签名

```cpp
DataBlockDep checkDataBlockAccessDependence(
    TaskNode* src, TaskNode* dst,
    bool checkProgramOrder = true);
```

### 4.2 流程（对照 MLIR `checkMemrefAccessDependence`）

```
Step 1. Dispatch by node kind (new):
   src/dst ∈ {BlockLoadNode, BlockStoreNode}
   其他组合 (KernelNode, LocalAllocNode, ...) → NoDep 返回

Step 2. Same-memref check (仿 AffineAnalysis):
   提取 src 的 OriginalMemref/TargetMemref 与 dst 的对应字段
   若 Value 不相等 (无 alias) → NoDep

Step 3. Access kind classification (DataBlock 特有):
   (Store, Load)  → 候选 RAW
   (Load,  Store) → 候选 WAR
   (Store, Store) → 候选 WAW
   (Load,  Load)  → 候选 Reuse

Step 4. Box intersection (仿 MemRefRegion 的简化版):
   从 op 的 AffineMap + indices + result-shape 投影出
   常量偏移向量 O 和尺寸向量 S。
   若两侧均为常量 → 矩形相交判定:
     overlap_empty → NoDep
     overlap == 两边完整 → Must
     overlap ⊂ 任一 → May

Step 5. Affine fallback (TODO P1.5):
   offsets 含 BlockArgument/symbol 时，构建
   FlatAffineValueConstraints，加等式 src_access == dst_access，
   调 isIntegerEmpty() 判可行性。
   初版直接返回 {候选kind, May} 保证 legality 不破坏。

Step 6. Program-order pruning (仿 AffineAnalysis):
   若 checkProgramOrder=true 且 src.op 在 dst.op 之后 → 交换 / 或返回 NoDep
   （取决于是否做双向边；初版只记前向边）。

Step 7. Carrier loop depth & distance (TODO P1.5):
   初版不填，留占位。
```

### 4.3 op 字段提取

**`DataBlockLoadOp`** (`ADORAOps.td:40-173`):
- `getOriginalMemref()` → 底层大 memref (源)
- `getIndices()` → `ValueRange`, affine map 的 operand
- `getAffineMap()` → map, `indices` 喂它得到基地址
- `getResult().getType().cast<MemRefType>().getShape()` → 块尺寸
- `getStrides()` (可选, `DenseI64ArrayAttr`) → stride 模式

**`DataBlockStoreOp`** (`ADORAOps.td:176+`):
- `getSourceMemref()` → 小 memref (块)
- `getTargetMemref()` → 底层大 memref
- `getIndices()`, `getAffineMap()`, `getStrides()` 同上
- 块尺寸来自 `getSourceMemref().getType().cast<MemRefType>().getShape()`

**Same memref 判定**：Load.OriginalMemref 与 Store.TargetMemref 的 `Value` 相等。

**Offset 提取（常量路径）**：若 `indices` 都是 `arith.constant` 或 SSA value
能通过 `getConstantIntValue()` 解出 → 把 affine map 代入得偏移向量；否则进 Step 5。

---

## 5. `analyzeDependencyInGraph(TaskGraph*)`

```cpp
void analyzeDependencyInGraph(TaskGraph* graph) {
  auto allNodes = graph->getAllNodes();

  // 1. 收集访问类节点 (BlockLoad, BlockStore)
  llvm::SmallVector<TaskNode*, 16> accesses;
  for (auto* n : allNodes) {
    if (dyn_cast<BlockLoadNode>(n) || dyn_cast<BlockStoreNode>(n))
      accesses.push_back(n);
  }

  // 2. 按 program order 两两比较
  graph->clearDepEdges();
  for (size_t i = 0; i < accesses.size(); ++i) {
    for (size_t j = i + 1; j < accesses.size(); ++j) {
      auto dep = checkDataBlockAccessDependence(
          accesses[i], accesses[j], /*checkProgramOrder=*/true);
      if (dep.kind != DepKind::NoDep)
        graph->addDepEdge(accesses[i], accesses[j], dep);
    }
  }
}
```

Kernel 节点的依赖暂通过"其消费的 Load 与产生的 Store 之间的 use-def 链"隐式表达（未
来 P2 需要显式化，初版够用）。

---

## 6. P4 桥梁（pass→mapper 依赖透传）

### 6.1 序列化 attribute

在 `ScheduleAdoraTasks` 末尾将 `_depEdges` 序列化为 `DictionaryAttr`，挂到对应
`func::FuncOp` 上：

```
adora.dep_summary = {
  nodes = [{id = 0, kind = "load",  memref = "%arg0"},
           {id = 1, kind = "store", memref = "%out"}],
  edges = [{src = 0, dst = 1, kind = "raw", strength = "must"}]
}
```

用原生 `ArrayAttr / DictionaryAttr / StringAttr / IntegerAttr`，不引 JSON。

### 6.2 mapper 侧视图

`mapper/include/mapper/taskgraph_dep_view.h`:
```cpp
struct DataBlockDepView {
  std::string kind;       // "raw" / "war" / "waw" / "reuse"
  std::string strength;   // "must" / "may"
  int src_id;
  int dst_id;
};
std::vector<DataBlockDepView> parseDepSummary(mlir::Attribute attr);
```

mapGemm.cpp / mapConv.cpp 在 lowering 入口调 `parseDepSummary(funcOp.getAttr(...))`
拿到视图，暂只 log，不消费 (等 P2/P3)。

### 6.3 CLI

- `cgra-opt --adora-dump-taskgraph-deps`：pass 侧 dump 依赖表到 stderr
- `cgra-mapper --dump-dep-view`：mapper 侧 dump 反序列化结果

两个 dump 的文本格式一致，用于跨阶段一致性检查。

---

## 7. 验证

### 7.1 单元测试 (litest, `test/ADORA/`)
1. `dep_raw_gemm_relu.mlir` — store→load 同 block → RAW Must
2. `dep_waw_disjoint.mlir` — 两 store 不相交 offset → NoDep
3. `dep_reuse_two_loads.mlir` — 两 load 同 block → Reuse
4. `dep_passmapper_handoff.mlir` — pass 侧 dump 与 mapper 侧 dump 一致

### 7.2 回归
`ninja check-adora` 除已知 7 个预存在失败 (缺口 8 `DFGgen.cpp:2396`) 外不回归。

---

## 8. 路线图

| 阶段 | 内容 | 状态 |
|---|---|---|
| P1.0 | 常量矩形路径 + `analyzeDependencyInGraph` 空壳填实 | **本次** |
| P1.5 | 仿射路径 (`FlatAffineValueConstraints`) + carrier loop depth | 后续 |
| P2 | 基于 DepMap 的 Pluto-lite 调度 ILP | 后续 |
| P3 | Pipeline cover LP (makespan) | 后续 |
| P4.0 | `adora.dep_summary` attribute + mapper 视图, 只 log | **本次** |
| P4.1 | mapper 消费 dep_summary 做 schedule | 等 P2/P3 |

---

## 9. 非目标

- **不重写 CDFG 生成**：缺口 8 (`DFGgen.cpp:2396`) 单独处理；
- **不动 `RemoveRedundantBlockLoads`**：等 P1.5；
- **不动 paper 正文**：等 P3 有数据；
- **不 git push**：全程本地。

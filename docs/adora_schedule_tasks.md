# `--adora-schedule-tasks` Pass 使用说明

**状态**：PR4 已完成并归档。分支 `jhlou/scheduletasks`（最新 `d825b2d`）。

---

## 功能清单

| # | 功能 | 状态 |
|---|------|------|
| 1 | Task Graph 构建（BlockLoad / Kernel / BlockStore / LocalMemAlloc） | ✅ 已实现 |
| 2 | 数据块依赖分析（RAW / WAR / WAW / RAR） | ✅ 已实现 |
| 3 | Buffer reuse（冗余 BlockLoad 消除 + `replaceAllUsesWith`） | ✅ 已实现 |
| 4 | `!ADORA.token` async chain（BlockLoad → Kernel → BlockStore 全程） | ✅ 已实现 |
| 5 | `adora.dep_summary` dict attr（供 mapper 消费） | ✅ 已实现 |
| 6 | Loop-carried dep 检测 | ✅ 已实现（仅诊断） |
| 7 | Loop-carried token yield（`affine.for` → `scf.for iter_args`） | ⏳ PR6 待实现 |
| 8 | Load-after-Load 消除 | ⏳ 待实现（stub） |

---

## 基本用法

```bash
# 默认：生成 dep_summary（不生成 token）
cgra-opt input.mlir --adora-schedule-tasks

# 推荐：生成完整 async token chain
cgra-opt input.mlir --adora-schedule-tasks="emit-token=true"

# 带可视化
cgra-opt input.mlir --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/g.dot"
```

---

## Options

| Option | Type | Default | 说明 |
|--------|------|---------|------|
| `emit-token` | bool | `false` | 生成 SSA `!ADORA.token` async chain（后续 pipeline 前提） |
| `emit-summary` | bool | `true` | 在 func 上发射 `adora.dep_summary` |
| `cross-check-summary-vs-token` | bool | `false` | CI-only，断言 token edges 与 summary 一致 |
| `dump-token-graph` | string | `""` | 非空时写 Graphviz DOT |

---

## 完整 Pipeline

```bash
cgra-opt input.mlir \
  --adora-schedule-tasks="emit-token=true" \
  --adora-assign-streams \
  --adora-lower-async-tokens \
  --adora-to-llvm-async-runtime
```

| Pass | 作用 | 状态 |
|------|------|------|
| `adora-schedule-tasks` | Task graph + 依赖分析 + buffer reuse + token 链接 | ✅ 必需 |
| `adora-buffer-reuse` | 冗余 BlockLoad 消除 | ⚠️ 冗余（功能已被 schedule-tasks 吸收，现为 no-op） |
| `adora-assign-streams` | 分配硬件 stream ID | ✅ 必需 |
| `adora-lower-async-tokens` | `!ADORA.token` → `ADORA.event.*` | ✅ 必需 |
| `adora-to-llvm-async-runtime` | `ADORA.event.*` → `llvm.call @adoraEvent*` | ✅ 必需 |

---

## Token Chain 实例

**输入**（fanin 模式）：
```mlir
%a = ADORA.BlockLoad %arg0 [0, 0] ...
%b = ADORA.BlockLoad %arg1 [0, 0] ...
%local = ADORA.LocalMemAlloc
ADORA.kernel { ... }
ADORA.BlockStore %local, %arg2 [0, 0]
```

**输出**（`emit-token=true`）：
```mlir
%a, %tok0 = ADORA.BlockLoad %arg0 [0, 0]  -> !ADORA.token
%b, %tok1 = ADORA.BlockLoad %arg1 [0, 0]  -> !ADORA.token
%local    = ADORA.LocalMemAlloc
%tokK     = ADORA.kernel async [%tok0, %tok1] { ... }
ADORA.BlockStore async [%tokK] %local, %arg2 [0, 0]
```

---

## 验证方式

### Lit 测试

```bash
# 4 个 schedule 专用测试
test/cgra-opt/schedule/schedule_cgra_tasks_tokens.mlir     # emit-token TOKEN/NOTOKEN 双检
test/cgra-opt/schedule/schedule_3mm.mlir                   # 多 kernel + buffer reuse
test/cgra-opt/schedule/schedule_tasks_dep_summary.mlir     # dep_summary attr
test/cgra-opt/schedule/schedule_gemm_tiled.mlir            # 64x64x64 tiled GEMM
```

### Experiment 示例

```bash
cd experiment/taskschedule
bash review.sh            # 4 个例子单 pass 验证
bash e2e_pipeline.sh      # 4 个例子完整 4-pass pipeline 验证
```

### End-to-End 验证结果（最新）

```
01_linear_chain:  PASS  (create=3 record=3 wait=3 destroy=3)
02_fanin:         PASS  (create=4 record=4 wait=4 destroy=4)
03_3mm:           PASS  (create=8 record=8 wait=8 destroy=8)   ← buffer reuse 生效
04_gemm_tiled:    PASS  (create=5 record=5 wait=6 destroy=5)
```

**验收**：最终 IR 无残留 `!ADORA.token`，全部下沉到 `llvm.call @adoraEvent{Create,Record,Wait,Destroy}`。

---

## 内部实现

`ScheduleADORATasksInFunction` 对每个含 kernel 的 block 依次执行：

| 步骤 | 函数 | 作用 |
|------|------|------|
| A | `generateTaskGraphFromBlock` | 建 node + KernelName 匹配的 default dep |
| B | `analyzeDependencyInGraph` | O(N²) 扫 RAW/WAR/WAW |
| C | `RemoveRedundantBlockStoreLoadPair` | 消除冗余 Load（`replaceAllUsesWith` + erase） |
| D | `threadTokensOnDMAs` | 重建 op 为 async，穿 `!ADORA.token` |
| D2 | `findLoopCarriedStoreLoadPair` | 检测 loop-carried dep（PR6 stub） |
| E | `verifyTokensMatchSummary` | `cross-check=true` 时验证 |

---

## 已知限制

| # | 问题 | 归属 |
|---|------|------|
| 1 | Loop-carried token yield 未实现 | 本 pass（PR6） |
| 2 | Load-after-Load 消除未实现 | 本 pass |
| 3 | `AccessSameDataBlock` 对动态 shape 保守返回 true | 本 pass |
| 4 | `adora-adjust-kernel-mem-footprint` SIGSEGV | **独立 bug**，与本 pass 无关（在 schedule-tasks 之前运行） |

---

## 参考

- 实现源码：`lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`
- TaskGraph：`lib/Dialect/ADORA/Transforms/TaskGraph/`
- Pass 注册：`include/ADORA/Dialect/ADORA/Transforms/Passes.td`
- 测试：`test/cgra-opt/schedule/`
- Experiment：`experiment/taskschedule/`
- PR 历史：`docs/pr4_review.md`
- 设计文档：`docs/async_token_design.md`

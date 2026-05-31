# Task Schedule 进展归档（2026-05-29）

本文件归档 task schedule 工作截至 2026-05-29 的完整状态，覆盖 schedule pass、
mapper async 出码、测试、experiment，以及本轮修正的过时结论。

相关文档：
- 使用说明：`docs/adora_schedule_tasks.md`
- 历史 PR / 设计：`docs/archive/`（pr4_review、async_token_design、pr6_* 等）

---

## 1. 一句话状态

核心链路 **schedule-tasks → assign-streams → lower-async-tokens → emit** 已全部
实现并通过测试。mapper 通过 `--enable-async` 消费 schedule 产出的 async token，
出码（C / Vitis SDK / pytest）正确携带依赖。所有相关 lit / experiment 测试全绿。

---

## 2. 端到端链路

```
adoracc / 手写 MLIR
   │
   ▼  --adora-schedule-tasks        (lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp)
build TaskGraph → 依赖分析 → 消除冗余 Load → 线程化 async token
   │
   ▼  --adora-assign-streams        (并行 DMA 分流，stream 0/1/…)
   │
   ▼  --adora-lower-async-tokens    (dep_summary → await-gather)
   │
   ▼  cgra-mapper --enable-async    (上述三 pass 串在 emit 前)
   │
   ▼  emit
   ├── --output-type=c       EmitCGRACall.cpp   → execute(..., EX_DEP_ST_LAST_TASK)
   ├── --output-type=sdk     EmitVitisSDK.cpp
   └── --output-type=pytest  EmitPytest.cpp     → test_runif depend_type=2
```

`cgra-mapper --help` 对 `--enable-async` 的说明：
> Run `--adora-schedule-tasks` + `--adora-assign-streams` +
> `--adora-lower-async-tokens` before emit so PR6.4 dep_summary path drives
> BlockStore await-gather.

---

## 3. 功能实现状态

| 能力 | 状态 | 证据 |
|------|------|------|
| Task graph 依赖分析（RAW/WAR/WAW） | ✅ | `ScheduleAdoraTasks.cpp`，schedule lit 套件 |
| async token 线程化（`BlockLoad/Store async`） | ✅ | schedule 输出含 `async [...]` + `!ADORA.token` |
| 冗余 Load 消除（buffer reuse） | ✅ | `03_3mm` 消掉 Load(%arg0)/Load(%arg3) |
| 并行 DMA stream 分配 | ✅ | `02_fanin` LoadA→stream0、LoadB→stream1 |
| 外层 affine.for → scf.for | ✅ | `affineForOuterToSCF`，`schedule_gemm_tiled` |
| **Loop-carried token yield（原 PR6 预留）** | ✅ **本轮确认已实现** | `05_loop_carried` iter_args 带 3×`!ADORA.token` |
| mapper 消费 async token（`--enable-async`） | ✅ | C 出 `EX_DEP_ST_LAST_TASK`，pytest 出 `depend_type=2` |
| emit scf.for / scf.yield 支持 | ✅ | OpVisitor + 三个 Emit 文件 |

### 仍存在的限制（未做，非 bug）
- Load-after-Load 消除未实现（`docs/adora_schedule_tasks.md` 限制 #2）。
- `AccessSameDataBlock` 对动态 shape 保守返回 true（限制 #3，设计取舍）。
- `adora-adjust-kernel-mem-footprint` SIGSEGV（限制 #4，独立 bug，与本 pass 无关）。
- 仿真器实跑验证未做：生成的 pytest 对接 `CGRA-Cocotb-Sim/server/test_runif.py`，
  但需 cocotb 环境（`conda env cocotb` + `cocotb~=1.9.2` + RTL）才能真正执行。

---

## 4. 测试清单

| 套件 | 数量 | 状态 | 运行方式 |
|------|------|------|----------|
| `test/cgra-opt/schedule/` | 8 | 全 PASS | `llvm-lit test/cgra-opt/schedule/` |
| `test/cgra-mapper/`（含 async） | 4 | 全 PASS | `llvm-lit test/cgra-mapper/` |
| `experiment/taskschedule/01..06` | 6 | 全 PASS | `bash <dir>/run.sh` |

运行前需 `source /data00/home/loujiahang/adora/env.sh`（注入 build/bin + LLVM 工具）。

### 本轮新增 / 修正的测试
- `test/cgra-mapper/emit/mvt_async_emit.mlir` — 新增正式 lit 测试，验证
  `--enable-async` 下 C / pytest 出码携带 async 依赖（用 `CHECK-DAG` 容忍 mapper
  placement 搜索的非确定 emit 顺序）。
- `experiment/taskschedule/06_mapper_async/` — 新增实验脚本，跑 baseline vs async
  对照，方便 diff review。
- `experiment/taskschedule/01..05/check.mlir` — 全部从过时的 `-> !ADORA.token`
  返回类型语法更新到当前 async + asyncToken SSA 值形态。

---

## 5. 本轮修正的过时结论（重要）

1. **「mapper 完全不消费 task schedule」——错误，已修正。**
   该结论源于只在 `mapper/` 子目录搜索；async / dep_summary 逻辑实际在
   `lib/Dialect/ADORA/Transforms`，mapper 通过 `--enable-async` 开关串入。

2. **「emit scf.for 支持待实现」——文档过时，已更新。**
   三个 Emit 文件均已落地，`docs/adora_schedule_tasks.md` 表格已补全 file:line。

3. **「限制 #1 loop-carried token yield 未实现」——文档过时，已更新。**
   `05_loop_carried` 输出已含 loop-carried token，`[PR6-TODO]` 诊断不再出现。

4. **`EX_DEP_ST_LAST_TASK` 判据陷阱**：该符号在 baseline C 里也作为 `#define`
   出现 1 次；区分 async 路径要看是否有**实际 `execute(...)` 调用**携带它
   （baseline 无任何 `BlockLoad async`，是更干净的判据）。

---

## 6. docs/ 整理（2026-05-29）

- 移入 `archive/`：`next_steps.md`、`pr6_session_handover.md`、
  `review_current_state.md`、`psg_next_phase.md`、`EMIT_BUG_20260518.md`。
- 删除：`adoracc_refactor_plan.md`（重构已不做）。
- 保留顶层：`adora_schedule_tasks.md`、`pipeline_overview.md`、
  `pipeline_detailed.md`、`psg_design.md`、`gap.md`、本文件。

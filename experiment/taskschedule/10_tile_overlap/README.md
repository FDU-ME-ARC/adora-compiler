# 10_tile_overlap — 一键端到端验证：两个独立 kernel 分到不同 tile

证明 **LLM tile assignment（Stage 1）+ mapper tile-约束落位（Stage 2）** 真正打通：
两个互不依赖的 kernel 被分到不同 tile，且它们的**计算节点**物理上落在不同 tile 的 GPE 上（可交叠并行的硬件前提）。

## 一键跑

```bash
./run.sh
```

预期：`ALL CHECKS PASSED ✓`，exit 0。产物在 `_work/`（已 gitignore）。

## 前置条件

- 已 build：`adora-compiler/build/bin/cgra-opt`、`cgra-mapper`
- Python3（解析 trace）；ranker 脚本在 `aicb-agent/experiments/llm_pipeline_tuning/task_schedule_ranker.py`
- 用 `--backend dryrule`（轮转分 tile），**不依赖在线 LLM**；想用真实 LLM 把 ranker 的 backend 换成 `openai` 并配好 endpoint 即可。

## 文件

| 文件 | 作用 |
|------|------|
| `two_kernels.mlir` | 两个独立 kernel（读写互不相交 memref，无跨 kernel 依赖），各含 `for { load; mulf; addf; store }` |
| `run.sh` | 一键脚本：3 项自检，全 PASS 才 exit 0 |
| `_work/` | 运行产物（dryrun IR、map 日志、agent trace、trace_check）—— gitignore |

复用的夹具（在 repo 内，非本目录）：
- ADG：`test/spec/cgra_fp32/cgra_adg_fp32_2tile.json`（把单 tile fp32 ADG 打补丁成 2 tile：前 16 GPE→tile0、后 16→tile1，`cgra_tile_num:2`）。
  > 真实多 tile ADG（如 `vitra_cgra_adg.json`）本身带 tile 字段、无需补丁；这里用 fp32 2-tile 是因为它**既多 tile 又支持 f32**（bf16 8-tile ADG 不支持 f32 op）。
- ops：`test/spec/cgra_fp32/operations_fp32.json`

## 三项自检

1. **Check 1 — Stage 1（dry-run，确定性）**：`cgra-opt --llm-pipeline-schedule --llm-pipeline-schedule-dry-run`
   两个 kernel 都写出 `adora.tile_set`。dry-run 跳过 LLM，给保守的 `[0..minTiles-1]`，所以两个都是 `[0]`（验证"属性被写出"）。

2. **Check 2 — Stage 1+2 端到端（dryrule ranker，轮转分 tile）**：`cgra-mapper --tile=2 --enable-async --enable-llm-schedule --llm-pipeline-schedule-ranker-cmd="... --backend dryrule"`
   - `cgra-mapper` exit 0；
   - IR dump 里 `k0: tile_set=[0]`、`k1: tile_set=[1]`（不同 tile）；
   - agent trace 有 **2 条** `tile_constraints_applied`（`k0 tile_set=[0] constrained_nodes=2`、`k1 tile_set=[1] constrained_nodes=2`）；
   - 解析 trace：k0 的 FMUL32/FADD32 落在 tile0 的 GPE、k1 落在 tile1 的 GPE，**计算节点零重叠**。
     （INPUT/OUTPUT 等 IO 节点按设计不加 tile 约束，可共享路由/IOB，不计入。）

## 相关 lit 测试（CI，确定性）

- `test/cgra-opt/schedule/tile_assignment_dryrun.mlir` — Stage 1 dry-run 写 tile_set 的 FileCheck。

## 实现位置

- Stage 1：`lib/Dialect/ADORA/Transforms/TaskPipeline/TileAssignment.{h,cpp}` + `LLMPipelineSchedule.cpp`（Step0 调 assignTiles，写 `adora.tile_set`）
- Stage 2：`tools/cgra-mapper/cgra-mapper.cpp` 的 `map_kernel` lambda（`if (subadg->isMultipleTile())` 门控 → 读 `adora.tile_set` → 对非 IO DFG 节点 `preestablishPlacementConstraints` 到该 tile 的 GPE）
- 设计与验证详情：`Agent-Compiler-notes/30_llm_and_api/llm_pipeline_schedule_status.md`（§3.4 / §8.3）

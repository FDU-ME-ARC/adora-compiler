# 事件级仿真器 — 状态（DONE v1，含严谨重写）

> 更新于重写完成。**代码全部本地未 commit**（等用户 review）。

## ✅ 已完成并验证（三个真实例子全过）

### 文件
- `arch/spec.py` — 读 vitra_spec.json（DMA_BPC=16、4bank×16KB、16PE×8tile）。修 adg.py 读错文件 bug。
- `core/event.py` — Event/Opcode/ResKind/Timeline 依赖驱动模型。
- `core/sram_track.py` — SRAMAccess + BankAllocator（cur/old/older 三档轮转，容量感知）。
- `extract/event_build.py` — **【已严谨重写】** SSA 环境解释器：递归按 affine 嵌套展开；
  func 层 affine.for→事件展开，kernel 体内 affine.for→cost；iter_args/yield→loop-carried token；
  event.create→零代价 marker；bank slot-recycle 边。
- `core/event_sim.py` — 离散事件 list-scheduler（最早可行 start 填资源）+ hang 检测。
- `viz/timeline.py` — render_event_gantt（DMA/PE 行）+ render_event_sram（addr×time×buf）。旧公式函数保留。
- `run.py` — `--event-sim`（+ --spec/--dma-bpc/--dma-setup/--max-cycles/--viz/--viz-sram）。
- `tests/test_event_sim.py` — 4 个回归全 PASS。
- 文档：EVENT_SIM_ARCH.md（架构）、EVENT_SIM_REDESIGN.md（重写设计）、EVENT_SIM_PLAN.md（计划）。

### 三例子结果（spec=cgra_bf16，DMA_BPC=16）

- gesummv 8737 vs 旧公式 4480（compute-only）→ 迭代间 load 已计入。
- 跨 kernel RAR（E8←E2）、loop-carried token（E25←E9/E13）均从 IR 直接读出、验证通过。

### ★ 全部 8 个正式实验例子已跑通并出图（明天检查用）
路径：`experiment/taskschedule/complex/<name>/_gantt/_cc/adora-cc-ir/3_task-schedule/<name>.final.mlir`
完整数字见 `docs/EVENT_SIM_RESULTS.md`；图在 `/tmp/evsim_all/<name>_{gantt,sram}.png`（共 16 张，0 冲突）。

| 例子 | trip | events | makespan | overlap |
|---|---|---|---|---|
| atax | 1 | 13 | 592 | 1.12x |
| attn | 8 | 132 | 2248 | 1.26x |
| cholesky | 16 | 195 | 4620 | 1.05x |
| ffn | 1 | 7 | 33412 | 1.01x (单大 kernel) |
| gesummv | 64 | 768 | 8737 | 1.49x |
| jacobi1d | 8 | 100 | 992 | 1.00x (token 链强制串行,非bug) |
| sobel | 62 | 1126 | 18732 | 1.54x |
| viterbi | 1 | 10 | 88 | 1.06x |

批量重跑：`tools/cycle-estimator/` 下对每个 name 跑
`python3 run.py --event-sim --mlir <f> --spec <vitra_spec.json> --viz /tmp/o/<n>_gantt.png --viz-sram /tmp/o/<n>_sram.png`
- 6 张图在 /tmp/evsim_out/（gesummv/tri/attn × gantt/sram）。

### 重跑命令
```
SPEC=/data00/home/loujiahang/adora/adora-compiler/test/spec/cgra_bf16/vitra_spec.json
python3 run.py --event-sim --mlir <scheduled.mlir> --spec $SPEC \
   --viz out_gantt.png --viz-sram out_sram.png [--max-cycles 400]
python3 tests/test_event_sim.py
```
三个 scheduled MLIR：
- /tmp/g_full_cc/adora-cc-ir/3_task-schedule/gesummv.final.mlir
- /tmp/tri_test/adora-cc-ir/3_task-schedule/tri.final.mlir
- /tmp/adoracc_attn/adora-cc-ir/3_task-schedule/attn.final.mlir

## 待办 / 已知限制（v2）
1. **dep_summary 内存依赖未连**：v1 只用 SSA token（三例子够用）。WAR/WAW 兜底待接
   `adora.dep_summary`（需先核 ScheduleAdoraTasks 的 op 序号口径）。
2. **三角循环 trip 低估**：tri 的 k_1 `affine.for to #map(%i)` 非常量上界，当前近似（见
   `_approx_dynamic_trip`），需 affine 域积分精确化。
3. **kernel cost II/drain 占位**：II=1/drain=4，可接 ii_model+dot 精化。
4. **DMA_SETUP=0** 是唯一未标定残差（待 RTL cocotb 标定；DMA_BPC 已 spec 锚定）。
5. 性能：大 trip 全展开事件多 + list-sched O(N²)；需要时换最小堆。

## 关键坑（已踩）
- async token operand **不是** python isinstance(OpResult)；用 `OpResult.isinstance(v)`+`v.owner`。
- list-sched 会抢未释放 slot → bank slot-recycle 显式边（reader(seq-depth)→load(seq)）。
- 区分 func 层 for（展开）vs kernel 体内 for（cost）：解释器遇 ADORA.kernel 即停，不递归其 region。

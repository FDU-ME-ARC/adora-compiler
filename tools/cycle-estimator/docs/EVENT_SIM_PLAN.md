# 事件级仿真器 — 问题记录 + 设计计划

> 创建：2026-06 ｜ 状态：**计划中（当前是公式模型，拟改为事件级仿真）**
> 关联：`DESIGN.md`（现公式模型）、`core/cycle_model.py`（现 estimate_cycles）、
> `viz/timeline.py`（甘特/SRAM 出图）

---

## 0. 为什么要改（一句话）

当前周期模型是**公式凑出来的近似**（`total = cfg + max(load+store, outer*(II*inner+drain))`），
对"外层 for 循环 + kernel 前后 op + op 间依赖"的真实执行**建模不准**，已暴露多个语义矛盾。
正确做法是**事件级仿真**：按真实 op 序 + 依赖逐事件推进时间轴，让迭代间 load、依赖等待、
资源占用都被仿真出来，而不是用公式假设。

---

## 1. 当前公式模型已确认的缺陷

### B-cyc-1：迭代间的 load/store 漏算（核心，本次发现）
`core/cycle_model.py` 第 ~150-160 行：
```python
compute = outer * (II * inner + drain)        # compute 乘了 outer ✓
total = cfg_c + max(load_c + store_c, compute) # 但 load/store 只算 1 次 ✗
```
注释声称 `dot 的 size = whole-nest 总字节`，所以 load 只算一次。**但实测不符**：
- gesummv_0（`tmp = A·x`，A 是 64×64=4096 元素）的 dot `Input.size = 256`（一个 tile，不是 4096）
- 配合 adoracc 的 `--adora-adjust-kernel-mem-footprint=... explicit-datablock`（把大数组分块流式搬）
- → 实际是**每次外层迭代 load 一个 tile**，load 发生 `outer` 次，不是 1 次

后果：mem-bound 的 kernel 周期被**严重低估**（迭代间反复 load 全漏了，还被 `max(mem, compute)`
里的 compute 吃掉）。例：gesummv_0 当前 total=4480（=compute），真实若 mem-bound 应是
`outer × dma_cycles(tile)` 量级，可能上万。

> **size 语义矛盾**：`io_bytes_from_dot` 注释写 "total bytes across the whole loop nest"，
> 但实测 size 是 per-iteration tile。注释与数据不符，是 B-cyc-1 的根。

### B-cyc-2：enclosing 外层循环曾被漏算（已修，但暴露模型脆弱）
`extract_kernel_loops` 原本只统计 kernel **内部** 的 affine.for，漏了**包裹 kernel 的外层
affine.for**（让 kernel 重复执行 N 次的那个）。已加 `_enclosing_loop_trips()`（用 MLIR-py
`.parent` 向上走）修复 → gesummv_0 从 132 → 4480 周期。
但这个修复让 B-cyc-1 更突出：outer 现在正确了（×64），可 load 还是 ×1，比例更失衡。

### B-cyc-3：甘特 phase 时序是"摆"出来的，不是仿出来的
`viz/timeline.py::_kernel_phases` 用规则摆放 config→load→compute→store（store 贴 compute 尾部
是上次修的）。但真实硬件里 load/compute/store 的重叠、迭代间的流水搭接，是**调度/依赖决定的**，
不该用固定规则摆。

### B-cyc-4：跨 kernel 调度是 list-scheduling 近似
`viz/timeline.py::schedule` 用粗粒度 list-scheduling（同 tile 串行 / dep 串行 / 否则并行）。
依赖来自 `hw_dep_type`，而 hw_dep_type 本身受限于 ScheduleAdoraTasks 只建线性链依赖
（fork-join 跨 affine.for 依赖建不出，见 SWEEP_RESULTS B4/gesummv 调查）。

---

## 2. 事件级仿真器设计

### 2.1 核心思想
不再用公式，而是**离散事件仿真**：把每个 kernel 的执行展开成时间轴上的事件序列，
按依赖和资源约束推进，记录每个事件的 [start, end) 和占用的资源（PE 阵列 / SPAD bank）。

### 2.2 输入（都已有）
- **op 序与嵌套**：从 scheduled MLIR 读 func body 的 op 序 —— 外层 `affine.for`、
  kernel 前后的 `ADORA.BlockLoad` / `ADORA.LocalMemAlloc` / `ADORA.kernel` / `ADORA.BlockStore`
- **依赖**：`ADORA.BlockLoad async [...]` 的 async token deps（谁等谁）+ SSA（load 读的 memref
  是哪个 store/alloc 写的）
- **每 op 的代价**：
  - BlockLoad/Store：`dma_cycles(per-iter tile bytes)`（dot 的 per-buffer size）
  - kernel：`II*inner + drain`（单次迭代的流水代价，复用现有 II 估计）
- **外层 trip count**：`_for_trip`（已有），决定每个 op 在外层循环里重复几次
- **资源**：tile 的 PE 数（num_alus）、SPAD bank 数/容量（arch/adg.py）

### 2.3 仿真模型
```
对每个 func：
  事件队列 = 按程序序展开 (含外层 for 的每次迭代):
    for it in range(outer_trip):           # 外层迭代
      for op in [BlockLoad..., kernel, BlockStore...]:  # kernel 前后的 op
        emit Event(op, iter=it, deps=async_deps(op), cost=cost(op), res=resource(op))
  调度:
    每个 Event.start = max(deps 的 end, 它占的资源最早空闲时间)
    Event.end = start + cost
    支持 overlap: 不同资源(DMA 引擎 / PE 阵列 / SPAD bank)可并发
  输出: 每个 Event 的 [start,end) + 资源 → 直接喂给甘特(图A) 和 SRAM 占用(图B)
```

关键：**迭代间的 load 自然就被展开了**（外层 it 循环里每次都 emit BlockLoad 事件），
B-cyc-1 不复存在。依赖等待、资源冲突也都是仿出来的，B-cyc-3/4 一并解决。

### 2.4 分阶段实现
1. **Stage A**：单 func 内、单 kernel + 它的 BlockLoad/Store，按外层 for 展开成事件，
   仿出 [start,end]。先验证 gesummv_0 的周期变合理（含迭代间 load）。
2. **Stage B**：多 kernel + 跨 kernel 依赖（async token / SSA），仿 fork-join / 链式。
3. **Stage C**：资源建模（DMA 引擎并发数、PE 阵列、SPAD bank 容量），让 overlap 真实。
4. **Stage D**：出图改为吃事件序列（甘特按事件画，SRAM 按 buffer 生命周期画）。

### 2.5 与现有代码的关系
- 复用：`extract/loop_info.py`（trip）、`core/dot_parser.py`（DFG）、`arch/adg.py`（资源）、
  `core/latency_table.py`（op 延迟）、`dma_cycles`
- 替换：`core/cycle_model.py::estimate_cycles`（公式）→ 新 `core/event_sim.py`（仿真）
- 替换：`viz/timeline.py::schedule`（list-sched）→ 吃事件序列

---

## 3. 暂不改的说明（当前决定）

按用户决定：**先记录问题（本文档），暂不改周期模型**。当前甘特图的周期数对
mem-bound kernel 偏低（B-cyc-1），看图时需知道这是公式近似、非事件级仿真结果。
事件级仿真器是后续独立任务，需 context 充足时专门做（涉及重写 cycle_model + timeline）。

---

## 4. 给接手人的 TL;DR
- 现在的周期 = 公式近似，**迭代间 load 漏算**（gesummv 类 mem-bound 偏低）
- 根因：dot 的 size 是 per-iter tile，但 cycle_model 当 whole-nest 只算一次
- 正解：改**事件级仿真**（按外层 for + op 序 + 依赖逐事件推进），见 §2 / §5
- 入口数据全都有（op 序/依赖/代价/资源），主要是写 `core/event_sim.py` + 改出图

---

## 5. 参考实现：AdaTileSim（只借范式，不照抄、不做那么精细）

参考项目：`/data00/home/loujiahang/agent/AdaTileSim`（一个成熟的 cycle-accurate NPU 仿真器）。

> **定位（重要）**：我们**只借它的调度范式**（依赖驱动 + 逐拍推进 + SRAM 占用记录），
> **不照抄代码、不做它那么精细**。AdaTileSim 是 cycle-accurate 全功能仿真器（NoC/多 device/
> DSL/分支等），ADORA 这边只要一个**轻量事件级仿真**：够把「迭代间 load、依赖等待、资源占用」
> 仿出来、让甘特/SRAM 图可信即可。**够用就停，不追求 cycle 级精确**。
> 一句话：要的是「事件级近似」而非「cycle-accurate 仿真」——比现在的公式准、比 AdaTileSim 简单。

### 5.1 借鉴这三个核心范式（看懂思路，自己写精简版，别照抄）（路径 `python/adt/simulator/`）

**(a) `instruction.py::Instruction` — 依赖驱动的事件模型**
```python
@dataclass
class Instruction:
    opcode: Opcode                 # LOAD / STORE / TCORE_OP(compute) ...
    ready_counter: int = 0         # 还有几个前驱没完成；==0 才能执行
    children: Set[Instruction]     # 我完成后通知谁
    start_cycle: int = -1
    finish_cycle: int = -1
    def is_ready(self): return self.ready_counter == 0 and not self.finished
    def finish(self, cyc):         # 完成时驱动依赖图前进
        self.finish_cycle = cyc; self.finished = True
        for c in self.children: c.ready_counter -= 1
```
→ ADORA 版：每个 `BlockLoad/LocalMemAlloc/Kernel/BlockStore`（×外层 for 每次迭代）= 一个
  Instruction；依赖来自 async token / SSA（load 读的 memref 是哪个 store/alloc 写的）。

**(b) `scheduler.py::_simulate` — 逐拍事件推进主循环**
```python
current_cycle = 0
while current_cycle < max_cycles:
    for pe in pes: pe.cycle(current_cycle)   # 每个资源单元推进就绪指令
    if all(pe.all_finished()): break
    if no_progress > hang_threshold: report_hang()  # 依赖死锁能报出来
    current_cycle += 1
```
→ ADORA 版：单 tile 至少有 {DMA 引擎, PE 阵列} 两类资源单元；每拍各自挑就绪指令执行，
  受资源占用 + 依赖约束。迭代间 load 自然被展开成多个 LOAD 事件 → B-cyc-1 解决。

**(c) `sram_tracker.py::SRAMAccess` — 图 B「数据段被占据」的现成数据结构**
```python
@dataclass
class SRAMAccess:
    sram_name: str
    addr_lo: int; addr_hi: int      # 地址区间（byte）
    start_cycle: int; finish_cycle: int   # 时间窗
    access_type: str                # load/store/tcore_read/tcore_write
    inst_name: str                  # 对应哪条指令
```
→ 这正是用户要的：**横轴时间 × 纵轴地址段 × 谁占 × 对应哪条 MLIR 指令**。仿真时每个
  LOAD/STORE/Kernel 产生一条 SRAMAccess，直接喂给图 B。bank 维度 = 按 addr 段划分。

### 5.2 ADORA 适配要点（与 AdaTileSim 的差异）
- **输入**：AdaTileSim 从它自己的 DSL/task-IR 构 TileGraph；ADORA 改为从 scheduled MLIR
  读 op 序 + async deps + memref SSA（复用现有 `extract/loop_info.py` 的 MLIR-py 遍历）。
- **代价**：LOAD/STORE = `dma_cycles(per-iter tile bytes)`（dot per-buffer size）；
  Kernel = `II*inner + drain`（复用现有 II 估计）。
- **资源**：先做 {1 个 DMA 引擎串行 + 1 个 PE 阵列} 最简模型（Stage C 再细化 bank/多 DMA）。
- **去掉**：NoC、多 device、inter-device link、nocsim C++ bridge、broadcast/multicast、
  RUNTIME_BRANCH —— ADORA 单芯片任务调度用不到。

### 5.3 落地映射（新文件）
| AdaTileSim | ADORA cycle-estimator 新文件 |
|---|---|
| `instruction.py::Instruction` | `core/event.py::Event`（精简版：opcode/ready_counter/children/cost/res/sram_access） |
| `scheduler.py::_simulate` | `core/event_sim.py::simulate(events, resources) -> timeline` |
| `sram_tracker.py` | `core/sram_track.py`（借鉴 SRAMAccess 结构，自己写精简版） |
| `loader`（DSL→graph） | `extract/event_build.py`：scheduled MLIR → Event 图（按外层 for 展开 + 连依赖） |
| 出图 | `viz/timeline.py` 改吃 Event timeline（甘特按 Event、SRAM 按 SRAMAccess） |

> 建议从 Stage A 起步：只跑 gesummv_0 单 kernel，参照 (a)(b)(c) 的思路写最简版，验证总周期
> 含迭代间 load（不再是 4480 这种 compute-only 低估值）。跑通后再加多 kernel/资源。


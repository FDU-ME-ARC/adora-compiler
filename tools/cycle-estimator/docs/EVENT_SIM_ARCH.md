# 事件级仿真器 — 架构设计（待 review）

> 创建：2026-06 ｜ 状态：**设计待批准，未写代码**
> 上游：`EVENT_SIM_PLAN.md`（高层计划 + AdaTileSim 范式参考）
> 目标：用**结构 cycle-accurate** 的离散事件仿真替换 `core/cycle_model.py` 的公式近似，
> 消除 B-cyc-1（迭代间 load 漏算）/ B-cyc-3（甘特摆放）/ B-cyc-4（list-sched 近似）。

---

## 0. 真实输入样本（设计的事实锚点）

本设计**不凭空假设**，全部锚定在已落盘的真实数据：
`/tmp/g_full_cc/adora-cc-ir/3_task-schedule/gesummv.final.mlir`（`module attributes {adora.scheduled}`）。

外层 `affine.for %arg4 = 0 to 64`（**trip=64**）每次迭代内的 op 序：

```
gesummv_0:
  BlockLoad Id=0  %alloca_1 []          -> memref<2xi32>     async token=t0    (8B,  scalar acc)
  BlockLoad Id=1  %arg0[%arg4, 0]       -> memref<1x64xi32>  async token=t1    (256B, A 的一行)
  BlockLoad Id=2  %arg2[0]              -> memref<64xi32>    async token=t2    (256B, x 向量)
  LocalMemAlloc                          -> memref<2xi32>     (Id=3)            (输出 buffer)
  kernel  async [t2, t1, t0] { 内层 affine.for 0..64: mul+add 累加 }            (II×64 + drain)
  BlockStore Id=3 %1 -> %alloca_1        (写回 scalar)                          (8B)
gesummv_1:
  BlockLoad Id=0  %alloca []            -> memref<2xi32>     async token=t7    (8B)
  BlockLoad Id=1  %arg1[%arg4, 0]       -> memref<1x64xi32>  async token=t9    (256B, B 的一行)
  BlockLoad Id=2  %arg2[0] async [t2]   -> memref<64xi32>    async token=t11   (256B, ★跨 kernel 依赖 t2)
  LocalMemAlloc                          -> memref<2xi32>     (Id=3)
  kernel  async [t11, t9, t7] { ... }
  BlockStore Id=3 %4 -> %alloca          (8B)
  ... 标量尾巴: 3*y0 + 2*y1 -> arg3[arg4] ...
```

依赖关系由 `gesummv.token_graph.dot` **独立印证**（不是我推的）：
```
load(t1) -> kernel0,  load(t2) -> kernel0,  load(t0)->kernel0
load(t0) -> store0    kernel0 -> store0
load(t2_g0) -> load(t2_g1)        ← ★跨 kernel RAR：gesummv_1 的 Id=2 load 等 gesummv_0 的 Id=2 load
load(t9)->kernel1, load(t11)->kernel1, load(t7)->kernel1
load(t7)->store1   kernel1 -> store1
```

**关键观察**：async token 是**显式**的，`kernel async [...]` / `BlockLoad async [...]` 的方括号里
就是依赖列表。仿真器不需要猜依赖，直接读 SSA token 的 def-use。

---

## 1. 范围与诚实的标定边界

### 1.1 能做到 cycle-accurate 的（结构 + 权威常数）
| 维度 | 来源 | 权威性 |
|---|---|---|
| op 序 / 外层 for 展开 | scheduled MLIR func body | ✅ 直读 |
| 依赖（谁等谁） | async token def-use + memref SSA | ✅ token_graph 印证 |
| op 延迟（kernel 内算子） | `latency_table.py`（Operations.scala） | ✅ 唯一事实源 |
| II / 关键路径 drain | `ii_model.py` + `dot_parser.py` | ✅ 复刻 mapper |
| config 周期 | `cfg_num // 3`（test_runif.py:1210） | ✅ 确认 |
| **DMA 带宽 DMA_BPC** | `vitra_spec.json::system_bus_beat_bits=128` → **16 B/cyc** | ✅ spec 锚定 |
| 资源结构（bank/PE/tile 数） | `vitra_spec.json`（**非** vitra_cgra_adg.json 连接图） | ✅ 直读 |

> ⚠️ **现存 bug**：`arch/adg.py` 一直读 `vitra_cgra_adg.json`——那只是 GIB 连接/布局图，
> **没有任何 width 字段**，于是 `num_input/data_width/iob_spad_bank_size` 全部 fallback 到默认/0。
> 真实硬件参数在 **`vitra_spec.json`**。本任务新增 `arch/spec.py` 读它，修掉这个空参数 bug。

### 1.2 DMA 带宽：**已锚定**（不再是自由标定常数）
`vitra_spec.json` 直接给了硬件常数（cgra_bf16 spec）：

| spec 字段 | 值 | 含义 / 用途 |
|---|---|---|
| `system_bus_beat_bits` | 128 | 系统总线每拍 128 bit = **DMA_BPC = 16 B/cyc** |
| `spad_data_width` | 128 | SPAD 数据通路 16 B/cyc（片上侧印证同值） |
| `dma_num_req_in_flight` | 8 | DMA 可 8 请求在飞（流水隐藏延迟） |
| `dma_lg_max_burst_size` | 6 | 最大 burst = 2⁶ beat |
| `tile_spad_num_banks` | 4 | 每 tile 4 个 SPAD bank（→ §5.2 bank 分配真实数） |
| `spad_bank_lg_size` | 14 | 每 bank = 2¹⁴ = 16 KB |
| `tile_num_row × tile_num_column` | 8 × 2 = 16 | 每 tile 16 PE（ResMII 用） |
| `cgra_tile_num` | 8 | 8 个 tile |
| `cgra_cfg_data_width` | 32 | config 字宽 |
| `cgra_data_width` | 16 | datapath 元素位宽（bf16=2B；i32 kernel 仍按 memref 类型取 elt bytes） |

→ **DMA cost = ceil(tile_bytes / 16)**（+ 小的 setup 延迟，见 §1.3）。
   例：gesummv 的 256B load → ceil(256/16) = **16 拍**。可 `--dma-bpc` 覆盖做 sweep。

### 1.3 唯一残留的标定残差（小且有界）
带宽已锚定；**未锚定的只剩 DMA setup 延迟 / 持续带宽效率**——总线能否真维持 1 beat/cycle、
burst 启动开销（`dma_lg_max_burst_size`/`dma_num_req_in_flight` 决定稳态吞吐能否打满）。
建模为 `dma_cost = ceil(bytes/DMA_BPC) + DMA_SETUP`，`DMA_SETUP` 默认 0，**唯一待 RTL 标定项**。
这是个小常数，不影响"迭代间 load 被正确展开 + 与 compute 按 bank 约束重叠"的结构正确性。

### 1.4 缓冲深度：**不是常数，复刻 io_scheduler 的 bank 分配（已定方案 A）**
缓冲深度（双/三缓冲）**不在我们喂的 `3_task-schedule` IR 里**——它是 mapper `io_scheduler.cpp`
在 *mapping 阶段* 决定的：`_cur_bank_status` / `_old_bank_status` / `_older_bank_status` 三档，
即**最多三缓冲（cur/old/older）**，按 `tile_spad_num_banks=4` 个物理 bank + 16KB/bank 容量贪心分配。

→ **本仿真器自己复刻这套 bank 分配器**（见 §5），缓冲深度是 bank 可用性的**自然结果**而非旋钮：
bank 够 → 自然双/三缓冲并发；bank 紧张 → 自然退化串行。这才真正"按调度逻辑模拟"，
且图 B 的 bank 窗口 = io_scheduler 会给的同一套。**不跑 mapper**，只复刻其 bank 规则。

> **一句话定位**：结构 cycle-accurate（事件驱动 + DMA 并发 + bank 冲突 + PE 占用 + 迭代间 load
> 真实展开），绝对周期里 DMA 那一项的常数待标定。比公式准、依赖/资源都是仿出来的，
> 但**不号称绝对 cycle 数已校准**——DMA_BPC 没标定前，绝对值是"结构正确、尺度待定"。

---

## 2. 数据模型

### 2.1 `core/event.py::Event`
```python
class ResKind(Enum):
    DMA = auto()      # BlockLoad / BlockStore 占 DMA 引擎
    PE  = auto()      # kernel 占某 tile 的 PE 阵列
    NONE = auto()     # LocalMemAlloc / config：零代价或只占 bank

@dataclass
class Event:
    eid: int
    opcode: str           # "LOAD" | "STORE" | "KERNEL" | "ALLOC" | "CONFIG"
    kernel: str           # 所属 KernelName（"gesummv_0" ...）
    it: int               # 外层迭代号（0..outer_trip-1）
    cost: int             # 周期数（见 §4）
    res_kind: ResKind
    tile: int = 0         # PE 事件落哪个 tile
    # 依赖驱动（借 AdaTileSim Instruction 范式）
    ready_counter: int = 0
    children: set[int] = field(default_factory=set)   # 完成后通知谁
    # SRAM 占用（喂图 B）
    sram: Optional[SRAMAccess] = None
    # 仿真填充
    start: int = -1
    finish: int = -1
    def is_ready(self): return self.ready_counter == 0 and self.start < 0
    def do_finish(self, cyc, events):
        self.finish = cyc
        for c in self.children: events[c].ready_counter -= 1
```

### 2.2 `core/sram_track.py::SRAMAccess`（借 AdaTileSim，精简）
```python
@dataclass
class SRAMAccess:
    bank: int             # SPAD bank 号（buffer -> bank 映射，见 §5）
    addr_lo: int          # byte 起（含）
    addr_hi: int          # byte 止（不含）
    start: int; finish: int
    kind: str             # "load" | "store" | "kernel_read" | "kernel_write"
    buf: str              # 逻辑 buffer 名（如 "gesummv_0.Id1"）— 对应哪条 MLIR 指令
```
冲突规则：同 bank + 地址重叠 + 时间窗重叠 → 串行（后者 start 推到前者 finish）。
双缓冲天然豁免：iter i 与 iter i+1 落不同 bank（i%n_buffers），不冲突 → load 与 compute 跨迭代重叠。

---

## 3. 输入抽取 `extract/event_build.py`

### 3.1 遍历（复用 loop_info.py 的 MLIR-py 基建）
```
parse scheduled MLIR -> 找外层 affine.for（trip=T）
for it in range(T):                       # 按外层迭代展开
  按 func body 程序序遍历 affine.for 体内的 ADORA.* op:
    BlockLoad      -> Event(LOAD,  cost=dma(tile_bytes),  sram=写入 buffer 的 bank 窗口)
    LocalMemAlloc  -> Event(ALLOC, cost=0,                sram=预留输出 bank)
    kernel         -> Event(KERNEL,cost=II*inner+drain,   sram=读输入+写输出 bank)
    BlockStore     -> Event(STORE, cost=dma(tile_bytes),  sram=读输出 bank)
  连依赖（§3.2）
```

### 3.2 依赖连边（三类，全部可从 IR 直读）
1. **async token**：`kernel async [%t2,%t1,%t0]` → 解析方括号里每个 SSA token，
   查"哪个 op 的 result token 是它"→ 连 producer→consumer。**含跨 kernel**（gesummv_1
   的 Id=2 load `async [%t2]` 自动连到 gesummv_0 的 Id=2 load）。
2. **BlockStore→kernel**：store 写的 memref（`%1`）是 kernel 内 `affine.store` 写的，
   连 kernel→store（token_graph 印证有这条边）。
3. **迭代间（同 buffer）**：iter i+1 对 buffer B 的 LOAD 依赖 iter i 对 B 的最后 reader 完成
   —— 但只在双缓冲耗尽时才成为真约束（由 §5 bank 仲裁隐式产生，**不显式连边**）。

> token 的 def-use 用 MLIR-py：`op.results` 里类型是 `!ADORA.token`（或 async token）的值，
> 记到 `token_def[value_id] = event`；消费侧遍历 `op.operands` 找 token 类型的，连边。

---

## 4. 代价模型 `cost(event)`

| op | cost 公式 | 数据来源 |
|---|---|---|
| LOAD / STORE | `ceil(tile_bytes / DMA_BPC)` | tile_bytes = result memref shape 连乘 × elt_bytes；DMA_BPC 见 §1.2 |
| KERNEL | `II × inner_trip + drain` | II=`ii_model`，inner_trip=kernel 内 affine.for，drain=dot 关键路径 |
| ALLOC | `0` | 只预留 bank |
| CONFIG | `cfg_num // 3` | 每 tile 首次用前发一次 |

gesummv 实例（DMA_BPC = 16 B/cyc，spec 锚定）：
- Id1/Id2 load：256B → ceil(256/16) = **16 cyc**；Id0 load：8B → 1 cyc
- kernel：II=1 × inner=64 + drain≈5 → ≈69 cyc
- store：8B → 1 cyc

---

## 5. 资源模型（cycle-accurate 关键子集）— **三类资源单元**

### 5.1 三类资源单元（甘特行 = 这三类）
- **DMA 引擎 ×1（串行，已定）**：每个 LOAD/STORE 占 DMA `cost` 拍，先到先排队。
- **PE 阵列 ×tile**：每个 KERNEL 占其 `tile` 的 PE `cost` 拍；同 tile 的 kernel 串行。
- **SPAD（bank 集合）**：复刻 io_scheduler 的 bank 分配器（§5.2），缓冲深度自然涌现。

### 5.2 SPAD bank 分配 = 复刻 io_scheduler（缓冲深度由此涌现，非旋钮）
按 `io_scheduler.cpp` 的模型 + `vitra_spec.json` 真实参数：
- bank 总数 = `tile_spad_num_banks` = **4**（每 tile）；每 bank 容量 = 2^`spad_bank_lg_size` = **16 KB**。
- 每个 buffer（一个 BlockLoad 的目标 memref / 一个 LocalMemAlloc）需占 `ceil(bytes/16KB)` 个 bank。
  （gesummv 的 buffer 全是 8B/256B，每个 1 bank 绰绰有余。）
- **三档轮转 cur/old/older**：给 buffer 分 bank 时优先选"两档都空"的，其次"old 空"，
  再次"older 空"——这正是 io_scheduler 的 `_cur/_old/_older_bank_status` 三档逻辑，
  使同一逻辑 buffer 在连续迭代里轮转 **最多 3 个物理 bank**（三缓冲上限，受 4 bank 总量约束）。
- bank 占用窗口 = 从该 buffer 的 LOAD.start 到它最后 reader(kernel).finish。
- bank 不够时（4 bank 全占）→ 后续 LOAD 必须等某 bank 释放 → **自然退化为串行**。

→ 缓冲深度（双/三/退化串行）是 bank 可用性的**结果**，忠实于真实调度逻辑。

### 5.3 迭代间重叠 = bank 轮转的来源（B-cyc-1 的真正解法）
iter i 的 kernel 还在读 bank A 时，iter i+1 的 LOAD 若分到空闲 bank B → **DMA(i+1) 与
compute(i) 并发**。bank 充足时这是稳态流水；bank 紧张时 io_scheduler 式分配自动串行化。
**这一条让"迭代间 load"既被展开、又按真实 bank 约束与计算搭接**，而非公式 `max(load,compute)` 一刀切。

---

## 6. 仿真主循环 `core/event_sim.py::simulate`

离散事件（非逐拍，省时间；640 事件级别）：
```
res_free = {DMA_k: 0, PE_tile: 0}            # 每个资源单元最早空闲时刻
bank_free = {bank: 0}                         # 每个 bank 最早空闲时刻
ready = [e for e in events if e.ready_counter == 0]   # 按 (it, 程序序) 排
while ready or 还有未完成:
    e = 取 ready 中"可最早开始"的事件（earliest feasible start）
    dep_ready = max(parents.finish)           # 依赖就绪
    if e.res_kind == DMA:  r = argmin_k res_free[DMA_k]   # 最早空闲引擎
    elif e.res_kind == PE: r = PE[e.tile]
    res_ready = res_free[r]
    bank_ready = bank_free[e.sram.bank] if e.sram else 0
    e.start  = max(dep_ready, res_ready, bank_ready)
    e.finish = e.start + e.cost
    res_free[r] = e.finish
    if e.sram: bank_free[e.sram.bank] = e.finish; register SRAMAccess
    e.do_finish(e.finish, events)             # 解锁 children -> 入 ready
makespan = max(e.finish)
return Timeline(events, makespan, sram_accesses)
```

正确性保障（针对"之前 bug 太多"）：
- **依赖只来自 IR 显式 token**，不靠规则摆放 → 消 B-cyc-3。
- **跨 kernel 依赖自动连**（token def-use）→ 消 B-cyc-4 的 fork-join 建不出问题。
- **每个外层迭代真 emit LOAD 事件** → 迭代间 load 不可能漏 → 消 B-cyc-1。
- 死锁检测：若一轮 ready 为空但仍有未完成事件 → 报 hang（借 AdaTileSim `_report_hang`）。

---

## 7. 输出与出图

`Timeline` 直接喂现有 `viz/timeline.py`（改吃事件，不再吃公式 phase）：
- **图 A 甘特**：行 = 资源单元（`DMA0..`, `PE-tile0..`）；每个 Event 画 [start,finish)，
  颜色按 opcode。迭代间 load 与 compute 的重叠**直接可见**（不再是摆出来的）。
- **图 B SRAM**：每条 `SRAMAccess` = 纵轴 bank/地址段 × 横轴时间窗 × buf 标签（哪条 MLIR 指令）。
  双缓冲下两 bank 交替占用一目了然。

---

## 8. 文件清单与改动

| 新文件 | 行数估计 | 替换/复用 |
|---|---|---|
| `arch/spec.py` | ~70 | **新增** 读 vitra_spec.json（修 adg.py 读错文件的空参数 bug） |
| `core/event.py` | ~60 | 新增 Event/ResKind |
| `core/sram_track.py` | ~60 | 借 AdaTileSim SRAMAccess + cur/old/older 分配 |
| `extract/event_build.py` | ~180 | 复用 loop_info MLIR-py 基建 |
| `core/event_sim.py` | ~120 | **替换** cycle_model.estimate_cycles 的地位 |
| `viz/timeline.py` | 改 ~80 | schedule()/render() 改吃 Timeline |
| `run.py` | 改 ~30 | 加 `--event-sim` 走新路径 |

`cycle_model.py` **保留**（公式快估仍可用作 sanity/对照），不删。

---

## 9. 验证计划（Stage A/B 合一，样本本就 2 kernel）

1. `event_build(gesummv.final.mlir)` → 打印事件图：64 iter × (3 load+alloc+kernel+store)×2
   = 64×12 ≈ 768 事件；断言跨 kernel 边（g1.Id2 ← g0.Id2）存在。
2. `simulate` → makespan **远大于 4480**（含 64×迭代间 load），且 `n_buffers=2` 时
   DMA 与 compute 跨迭代重叠（makespan < 纯串行 sum）。
3. 出图 A/B 肉眼验证：load/compute 跨迭代搭接、双 bank 交替。
4. （待 RTL 环境）用 cocotb ground truth 标定 `DMA_BPC`，对齐绝对周期。

---

## 10. 决策（已拍板）+ 唯一剩余标定项

| # | 项 | 定稿 |
|---|---|---|
| 1 | DMA 引擎数 | **`n_dma = 1` 串行** |
| 2 | DMA_BPC | **已锚定 = `system_bus_beat_bits/8` = 16 B/cyc**（vitra_spec.json）；`--dma-bpc` 可覆盖做 sweep |
| 3 | 缓冲深度 | **不设旋钮**，复刻 io_scheduler 的 cur/old/older bank 分配（4 bank，§5.2），深度自然涌现 |
| 4 | 资源单元 | **三类：DMA / tile(PE) / SPAD(bank)**，甘特行即此三类 |
| 5 | 范围 | **Stage A+B 一次做完**（单 func 多 kernel + 依赖 + io_scheduler 式 bank） |

剩余唯一未标定常数 = **DMA_SETUP**（DMA 启动延迟/带宽效率残差，默认 0，小且有界，
待 RTL 校准）。带宽 DMA_BPC 已由 `vitra_spec.json` 锚定，不再是标定项。

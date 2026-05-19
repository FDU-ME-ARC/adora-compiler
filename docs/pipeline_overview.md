# ADORA 编译流水线全景：从 C 源码到 CGRA 可执行

> 更新时间：2026-05-17  
> 分支：`jhlou/scheduletasks`  
> 覆盖范围：完整 5 阶段编译流水线，含 task-schedule 路径的两个变体

---

## 总览图

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 0: C → 原始 MLIR                                             │
│  工具: cgeist (Polygeist) + mlir-opt                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ *_normalized.mlir
┌──────────────────────────────▼──────────────────────────────────────┐
│  Stage 1: 提取 Kernel                                               │
│  工具: cgra-opt                                                     │
│  --adora-extract-affine-for-to-kernel                               │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ *_kernel.mlir  (含 adora.kernel{} 标注)
┌──────────────────────────────▼──────────────────────────────────────┐
│  Stage 2: Kernel 优化（内存 / 展开 / 数学重写）                      │
│  工具: cgra-opt                                                     │
│  --adora-simplify-loadstore                                         │
│  --adora-math-rewrite                                               │
│  --adora-adjust-kernel-mem-footprint                                │
│  [--adora-auto-unroll]  ← 可选                                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ *_opt.mlir  (含显式 BlockLoad/BlockStore)
          ┌────────────────────┴────────────────────┐
          │ Path A（默认）                           │ Path B（--enable-async）
          ▼                                          ▼
┌──────────────────┐                    ┌────────────────────────────┐
│  Stage 3A: 直接  │                    │  Stage 3B: Task Schedule   │
│  进入 Mapper     │                    │  工具: cgra-opt / 内嵌于   │
│  （无 token）    │                    │        cgra-mapper         │
└────────┬─────────┘                    │ --adora-schedule-tasks     │
         │                              │ （stream 着色在 emit 内完成）│
         │                              └────────────┬───────────────┘
         │                                           │ *_scheduled.mlir
         │                             (含 !ADORA.token SSA chain)
         └────────────────────┬────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────────┐
│  Stage 4: Mapping                                                   │
│  工具: cgra-mapper                                                  │
│  DFG 提取 → 资源分配 → 路由 → Emit                                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                  ▼
    Python host 代码                      CGRA 固件 (.c)
    (含 asyncio.gather)               (cgra_exe.c / VitisSDK)
              │                                  │
┌─────────────▼──────────────────────────────────▼───────────────────┐
│  Stage 5: 编译链接                                                   │
│  gcc/riscv-gcc 编译固件 + Python 直接运行                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Stage 0：C 源码 → 原始 MLIR

**脚本**：`experiment/scripts/0_compileCtoMLIR.sh`

### 步骤 0-1：C → 原始 MLIR（cgeist）

```bash
cgeist \
    -O2 -lm -lgcc \
    -Dsize_t=int -Dwint_t=int -DROCKET_TARGET -D_riscv \
    -DDATA_TYPE_IS_FLOAT -D${DATASET_Size} \
    --import-all-index \
    -I<polybench_utils> \
    kernel.c \
    -S -o IR/kernel.mlir
```

`cgeist` 是 Polygeist 的前端，把 C 翻译为 MLIR func + affine dialect。输出含 `affine.for`、`memref.alloc`、`memref.load/store` 等 op。

### 步骤 0-2：MLIR 规范化（mlir-opt）

```bash
mlir-opt \
    --allow-unregistered-dialect \
    --affine-loop-normalize \
    --affine-simplify-structures \
    --normalize-memrefs \
    --force-specialization \
    --bufferization-bufferize \
    IR/kernel.mlir -o IR/kernel_normalized.mlir
```

规范化 affine 索引，统一 memref 布局，为后续 ADORA pass 准备 IR。

**输出形态**：
```mlir
func @kernel(%A: memref<128x128xf32>, %B: ...) {
  affine.for %i = 0 to 128 {
    affine.for %j = 0 to 128 {
      %v = affine.load %A[%i, %j] ...
      affine.store %v, %C[%i, %j] ...
    }
  }
}
```

---

## Stage 1：提取 Kernel

**脚本**：`experiment/scripts/0_compileCtoMLIR.sh`（后半段）

```bash
cgra-opt \
    --canonicalize \
    --reconcile-unrealized-casts \
    --affine-loop-fusion \
    --adora-extract-affine-for-to-kernel \
    --arith-expand --memref-expand \
    --cse \
    IR/kernel_normalized.mlir -o IR/0_kernels/kernel_kernel.mlir
```

### `--adora-extract-affine-for-to-kernel` 做了什么

扫描函数体，把最外层 `affine.for` 识别为 CGRA 可执行 kernel，包裹为：

```mlir
adora.kernel @kernel_0 {
  affine.for %ti = 0 to 16 step 8 {  // tiled loop
    affine.for %tj = 0 to 16 step 8 {
      ...
    }
  }
}
```

`adora.kernel` 是后续所有调度分析的操作单元。

---

## Stage 2：Kernel 优化

**脚本**：`experiment/scripts/1_kernel_opt.sh`

```bash
cgra-opt \
    --adora-simplify-loadstore \
    --adora-math-rewrite \
    --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 \
        disable-remainder-block explicit-datablock" \
    [--adora-auto-unroll="cgra-adg=cgra_adg.json"] \
    IR/0_kernels/kernel_kernel.mlir -o IR/1_kernels_opt/kernel_opt.mlir
```

### 各 Pass 说明

**`--adora-simplify-loadstore`**：消除冗余 load/store，合并连续访问。

**`--adora-math-rewrite`**：把 `math.exp`、`math.tanh` 等高层数学 op 展开为 CGRA 原语序列。

**`--adora-adjust-kernel-mem-footprint`**：核心 tiling pass。
- 把 `affine.load/store` 替换为显式的 `ADORA.BlockLoad` / `ADORA.BlockStore` DMA op
- `cachesize=128`：片上 SRAM 大小（单位 KB）
- `singlearraysize=8`：单个 DataBlock 的最大大小（单位 KB）
- `explicit-datablock`：强制每个 DataBlock 独立命名，使后续依赖分析可以区分 tile
- 输出的 `BlockLoad` 带有 `DataBlock`（逻辑数据块描述）和 `LocalMemAlloc`（片上 SRAM slot）

**`--adora-auto-unroll`**（可选）：基于 CGRA ADG（架构描述图）自动选择最优展开因子。PR（AutoUnroll）里增加了 LLM ranker 接口（`DesignSpace/candidates.json` → `selected.json`）。

**输出形态**：
```mlir
func @kernel(%A: memref<128x128xf32>, ...) {
  affine.for %ti = 0 to 16 {
    affine.for %tj = 0 to 16 {
      %lma = ADORA.LocalMemAlloc : !ADORA.localMem<8x8xf32>
      ADORA.BlockLoad %A[%ti*8, %tj*8], %lma : ...  // DMA: DRAM → SRAM
      ADORA.KernelOp @kernel_0 ins(%lma, ...) outs(...)
      ADORA.BlockStore %lma, %C[%ti*8, %tj*8] : ...  // DMA: SRAM → DRAM
    }
  }
}
```

---

## Stage 3：Task Scheduling（本 PR 的核心工作）

这里有两条路径，取决于是否启用 async token chain。

### Path A：不启用 async（默认，等同 pre-PR4 行为）

直接把 `_opt.mlir` 喂给 `cgra-mapper`，mapper 看不到任何 token 依赖，emit 层生成串行的 Python（无 `asyncio.gather`）。

### Path B：启用 async（`--enable-async=true` 或手动 pass 链）

**方式一：在 cgra-opt 中运行（推荐，IR 可检查）**

```bash
cgra-opt \
    --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/tok.dot" \
    --adora-assign-streams="max-streams=4" \
    --adora-lower-async-tokens \
    IR/1_kernels_opt/kernel_opt.mlir \
    -o IR/2_scheduled/kernel_scheduled.mlir
```

**方式二：在 cgra-mapper 内嵌运行（PR6.4）**

```bash
cgra-mapper \
    --enable-async=true \
    --adg=cgra_adg.json \
    --op-file=operations.json \
    kernel_opt.mlir
```

mapper 内部自动跑上述三个 pass，然后进行 mapping 和 emit。

---

### `--adora-schedule-tasks` 详解

**Pass 文件**：`lib/Dialect/ADORA/Transforms/ScheduleAdoraTasks.cpp`

Pass 内部按以下顺序执行 6 个子步骤：

#### 步骤 1：构建任务图（TaskGraph）

扫描 `func.func` 内的所有 `BlockLoad`、`BlockStore`、`KernelOp`，每个 op 成为图中一个节点（Node），按 MLIR SSA 顺序初始化拓扑关系。

#### 步骤 2：依赖分析（`analyzeDependencyInGraph`）

逐对节点分析 DataBlock 粒度的依赖：

| 依赖类型 | 含义 | 示例 |
|---------|------|------|
| RAW（Read-After-Write）| Load 读取 Store 写的数据 | `Store(C) → Load(C)` 跨 kernel |
| WAR（Write-After-Read）| Store 写入之前 Load 读的区域 | `Load(A) → Store(A)` 同 tile |
| WAW（Write-After-Write）| 两个 Store 写同一块 | 多 kernel 写同一输出 |
| RAR（Read-After-Read）| 两个 Load 读同一块 | 可复用 buffer |

⚠️ **已知 gap**：`analyzeDependencyInGraph` 函数体是空 stub（`cpp:160`，只有一行注释），当前依赖分析由 `threadTokensOnDMAs` 的 SSA 追踪代替完成。

#### 步骤 3：Buffer Reuse（`RemoveRedundantBlockStoreLoadPair`）

当 `BlockStore(C[ti,tj])` 之后紧跟 `BlockLoad(C[ti,tj])` 且访问同一 DataBlock tile 时：
- 消除 Load op，下游 kernel 直接复用 on-chip `LocalMemAlloc` buffer
- 效果：节省一次 DRAM→SRAM DMA，减少访存延迟

3mm 样例效果：`kernel_3mm_2` 原需从 DRAM 读 2 次，经 buffer reuse 节省 2 次 DMA。

⚠️ **已知 gap**：Load-after-Load 消除（`RemoveRedundantBlockLoads`）整体被注释（`cpp:218-256`，约 40 行），未实现。

#### 步骤 4：Token Threading（`threadTokensOnDMAs`）

按依赖图拓扑给每个 op 插入 `!ADORA.token` SSA 凭证：

```
BlockLoad_A →tok0→ ┐
BlockLoad_B →tok1→ ├→ KernelOp async[tok0,tok1] →tokK→ BlockStore async[tokK]
BlockLoad_C →tok2→ ┘
```

- `BlockLoad → KernelOp`：load 完成后 kernel 才启动
- `KernelOp → BlockStore`：kernel 完成后 store 才写出
- 跨 kernel RAW：`BlockStore(K0) →tokS→ BlockLoad(K1)` 确保 K0 写完才允许 K1 读

#### 步骤 5：Loop-Carried 依赖处理（PR6）

对 tiled GEMM 的 `tk` accumulation 循环，检测跨迭代 RAW 依赖：

```mlir
// 第 k 次迭代：Store(C_partial[ti,tj])
// 第 k+1 次迭代：Load(C_partial[ti,tj]) ← 跨迭代 RAW
```

当前状态：
- PR6.1：检测并记录到 `adora.lc_dep_summary` attr ✅
- PR6.2/6.3：`affine.for → scf.for iter_args(!ADORA.token)` yield + emit 前 strip ✅
- Emit 层消费 `_lcDepSummary`：❌ 未实现（保守串行）

#### 步骤 6：序列化 `adora.dep_summary`

把全部 DataBlock 级依赖关系序列化为 `funcOp` 上的 ArrayAttr，供无 SSA token 时的 emit fallback 使用：

```mlir
func @kernel() attributes {
  "adora.dep_summary" = [{src="Load_A", dst="Kernel_0", type="RAW"}, ...]
}
```

---

### `--adora-assign-streams` 详解

**Pass 文件**：`lib/Dialect/ADORA/Transforms/AssignStreams.cpp`

算法：Kahn 拓扑排序 + 贪心流分配

1. 收集所有 async-capable ops（BlockLoad / BlockStore / KernelOp）
2. 按 `asyncDependencies` SSA 边做 Kahn 拓扑排序
3. 无前驱 op → 分配新 stream ID（上限 `max-streams=4`）
4. 有前驱 op → 继承最小前驱 stream ID
5. 把 `stream : i32` attr 写回每个 op

作用：让 `lower-async-tokens` 生成 `adoraEventRecord(stream_i)` / `adoraEventWait(stream_j)`，让 runtime 把无依赖的 op 调度到不同的硬件 stream 上真正并行执行。

---

### `--adora-lower-async-tokens` 详解

**Pass 文件**：`lib/Dialect/ADORA/Transforms/LowerAsyncTokens.cpp`

把 `!ADORA.token` SSA 链路降级为 event op 序列：

```mlir
// 降级前
%tok = ADORA.BlockLoad async [...] %A → !ADORA.token

// 降级后
%ev = ADORA.event.create : !ADORA.token
ADORA.BlockLoad %A ...
ADORA.event.signal %ev   // signal 在 stream_i 上
...
ADORA.event.wait %ev     // wait 在依赖的 stream_j 上
```

**注意**：降级后 SSA token 消失，emit 层无法再用 Path 1（SSA 追踪），只能用 Path 2（dep_summary）。如果只做 Python emit，不需要运行这个 pass（SSA token 足够）。

---

## Stage 4：Mapping（cgra-mapper）

**脚本**：`experiment/scripts/3_kernel_map.sh`

```bash
cgra-mapper \
    --adg="${CGRA_ADG_PATH}/cgra_adg.json" \
    --op-file="${CGRA_OP_FILE_PATH}/operations.json" \
    --output="map_result/cgra_exe.c" \
    [--enable-async=true] \
    kernel_opt.mlir
```

### cgra-mapper 内部流程

```
读入 *_opt.mlir
  │
  ├─ [若 --enable-async] 运行 schedule-tasks（仅此一个 pass）
  │
  ▼
TensorDataflowGen（DFG 提取）
  └─ 每个 KernelOp → 一张 CDFG（控制数据流图，dot 格式）
  
  ▼
Mapping（资源分配 + 路由）
  └─ CGRA ADG（架构描述图）约束下把 CDFG 节点映射到 PE 阵列
  └─ 输出：每个 kernel 的 PE 绑定表 + 路由表
  
  ▼
EmitPytest / EmitCGRACall / EmitVitisSDK
  └─ 生成 Python host 代码（含 asyncio.gather，若有 token chain）
  └─ 生成 CGRA 固件 .c
```

### EmitPytest 双路径（token vs dep_summary）

| 场景 | SSA token 在不在 | 走哪条路径 | 产出 |
|------|-----------------|------------|------|
| Path A（未开 async）| 无 | 无依赖信息 | 串行 Python，无 gather |
| Path B，未跑 lower-async | 有（SSA 完整）| Path 1：追 asyncDependencies | `asyncio.gather(task_A, task_B)` |
| Path B，跑了 lower-async | 无（已降级）| Path 2：查 dep_summary attr | `asyncio.gather(task_A, task_B)` |

**生成的 Python 代码示例**（Path 1 / Path 2 结果相同）：
```python
async def run():
    task_load_A = asyncio.ensure_future(dma_load(A, sram_A))
    task_load_B = asyncio.ensure_future(dma_load(B, sram_B))
    await asyncio.gather(task_load_A, task_load_B)   # ← 两路 DMA 并行

    task_kernel = asyncio.ensure_future(cgra_execute(sram_A, sram_B, sram_C))
    await asyncio.gather(task_kernel)

    task_store = asyncio.ensure_future(dma_store(sram_C, C))
    await asyncio.gather(task_store)
```

---

## Stage 5：编译链接

**脚本**：`experiment/scripts/5_compile_and_link.sh`

```bash
# 编译 CGRA 固件（RISC-V 交叉编译）
riscv64-unknown-elf-gcc \
    -O2 -march=rv64gc \
    cgra_exe.c -o cgra_fw.elf

# Python host 直接运行
python3 host_test.py
```

---

## 当前两条可用路径对比

| 维度 | Path A（默认）| Path B（--enable-async）|
|------|--------------|------------------------|
| stage 3 pass | 无 | schedule-tasks（仅此） |
| emit 依赖信息来源 | 无 | SSA token（Path 1，唯一路径）|
| Python 并发 | 串行 | asyncio.gather 并发 |
| loop-carried dep | 不处理 | 检测 ✅，emit ❌（待做）|
| 端到端验证 | ✅ 通过 | ❌ 待找合适样例验证（gemm_funccall 有 pre-existing crash）|
| 是否需要 lower-async-tokens | 否 | 可选（做 Python emit 不需要，做 LLVM 固件才需要）|

---

## 快速命令参考

```bash
# 只跑 schedule-tasks，查看 token 图（推荐调试方式）
cgra-opt input.mlir \
    --adora-schedule-tasks="emit-token=true dump-token-graph=/tmp/tok.dot" \
    --adora-assign-streams \
    -o output_scheduled.mlir
dot -Tpng /tmp/tok.dot -o tok.png

# 完整 LLVM 降级路径（e2e 验证脚本用这条）
cgra-opt input.mlir \
    --adora-schedule-tasks \
    --adora-assign-streams \
    --adora-lower-async-tokens \
    --adora-to-llvm-async-runtime \
    -o output_lowered.mlir

# 跑所有 e2e 测试用例
./experiment/taskschedule/e2e_pipeline.sh

# 跑 lit 测试
ninja -C build check-adora 2>&1 | tail -20
```

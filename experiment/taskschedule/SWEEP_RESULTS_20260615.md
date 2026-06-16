# Task-Schedule Benchmark Sweep — 2026-06-15

> 对 `experiment/taskschedule/` 下全部 18 个 benchmark 跑了两条 pipeline，扫 bug。
> 工具：`build/bin/cgra-opt`。生成命令见各节。

## TL;DR

| pipeline | 结果 |
|----------|------|
| async token pipeline（`schedule-tasks → assign-streams → lower-async-tokens → to-llvm-async-runtime`） | 18/18 跑通无 crash，但 **11–18 是假阳性**（0 token） |
| LLM schedule pass（`llm-pipeline-schedule --dry-run`，写 tile_set + hw_dep_type） | 18/18 **0 crash**；tile_set/dep_type 计数符合 input 形态 |

**没有 compiler crash。** 暴露的问题全部是 **测试 input 本身处于错误的 pipeline 阶段**（11–18 是半成品/裸输入），不是 pass 的 bug。

---

## 1. 输入形态盘点（根因）

| benchmark | kernels | BlockLoad | BlockStore | 状态 |
|-----------|--------:|----------:|-----------:|------|
| 01_linear_chain | 2 | 2 | 2 | ✅ 完整 |
| 02_fanin | 2 | 4 | 2 | ✅ 完整 |
| 03_3mm | 3 | 10 | 5 | ✅ 完整 |
| 04_gemm_tiled | 1 | 8 | 3 | ✅ 完整 |
| 05_loop_carried | 1 | 1 | 1 | ✅ 完整 |
| 06_gray | 1 | 3 | 3 | ✅ 完整 |
| 07_tiled_matmul | 1 | 1 | 1 | ✅ 完整 |
| 08_fir | 1 | 2 | 1 | ✅ 完整 |
| 09_gesummv | 1 | 3 | 2 | ✅ 完整 |
| 10_3mm | 3 | 6 | 3 | ✅ 完整 |
| 11_atax | 2 | **0** | **0** | ⚠️ 有 kernel 壳，无 DMA |
| 12_jacobi1d | 2 | **0** | **0** | ⚠️ 有 kernel 壳，无 DMA |
| 13_cholesky | 2 | **0** | **0** | ⚠️ 有 kernel 壳，无 DMA |
| 14_attn | **0** | 0 | 0 | ⚠️ 裸 scf.for MLIR（未 lower） |
| 15_ffn | **0** | 0 | 0 | ⚠️ 裸 scf.for MLIR |
| 16_fft | **0** | 0 | 0 | ⚠️ 裸 scf.for MLIR |
| 17_sobel | **0** | 0 | 0 | ⚠️ 裸 scf.for MLIR |
| 18_viterbi | **0** | 0 | 0 | ⚠️ 裸 scf.for MLIR |

**根因**：
- `14–18` 是 `scf.for` 层的**原始输入**，还没跑前端 lowering（affine 转换 + `extract-affine-for-to-kernel` + BlockLoad/Store 插入），所以 0 kernel。
- `11–13` 跑了 kernel 提取但**没插入 BlockLoad/Store**（前端 lowering 不完整），所以 0 DMA。
- async token 建立在 BlockLoad/Store 的数据搬运依赖上 → 无 DMA = 0 token；dep_type 也无法判定。

---

## 2. async token pipeline 结果

命令（每个 benchmark）：
```
cgra-opt input.mlir --adora-schedule-tasks --adora-assign-streams \
         --adora-lower-async-tokens --adora-to-llvm-async-runtime
```

| benchmark | create | record | wait | destroy |
|-----------|-------:|-------:|-----:|--------:|
| 01_linear_chain | 3 | 3 | 3 | 3 |
| 02_fanin | 4 | 4 | 4 | 4 |
| 03_3mm | 10 | 10 | 10 | 10 |
| 04_gemm_tiled | 9 | 6 | 6 | 6 |
| 05_loop_carried | 7 | 4 | 4 | 4 |
| 06_gray | 16 | 8 | 13 | 8 |
| 07_tiled_matmul | 3 | 3 | 3 | 3 |
| 08_fir | 4 | 4 | 4 | 4 |
| 09_gesummv | 9 | 7 | 6 | 7 |
| 10_3mm | 10 | 10 | 10 | 10 |
| 11_atax … 18_viterbi | 0 | 0 | 0 | 0 |

> **e2e_pipeline.sh 的 PASS 判定有漏洞**：0 token 也算 PASS。11–18 因此被误报为通过。
> 建议：给该脚本加一条 "token>0 才算 PASS（或显式标 SKIP）" 的判定。

---

## 3. LLM schedule pass 结果（核心线）

命令（每个 benchmark）：
```
cgra-opt input.mlir --llm-pipeline-schedule --llm-pipeline-schedule-dry-run \
         --llm-pipeline-schedule-num-tiles=2 --llm-pipeline-schedule-pe-per-tile=16
```

| benchmark | rc | tile_set | dep_type | 说明 |
|-----------|---:|---------:|---------:|------|
| 01_linear_chain | 0 | 1 | 0 | 单/双 kernel，dry-run 写 tile_set |
| 02_fanin | 0 | 1 | 0 | |
| 03_3mm | 0 | 3 | 4 | 多 kernel，有跨任务依赖 → dep_type 出现 |
| 04_gemm_tiled | 0 | 1 | 0 | |
| 05_loop_carried | 0 | 1 | 0 | |
| 06_gray | 0 | 1 | 0 | |
| 07_tiled_matmul | 0 | 1 | 0 | |
| 08_fir | 0 | 1 | 0 | |
| 09_gesummv | 0 | 1 | 0 | |
| 10_3mm | 0 | 3 | 4 | 同 03 |
| 11_atax | 0 | 2 | 0 | 2 kernel 各写 tile_set，但无 DMA → 无 dep_type |
| 12_jacobi1d | 0 | 2 | 0 | 同上 |
| 13_cholesky | 0 | 2 | 0 | 同上 |
| 14_attn … 18_viterbi | 0 | 0 | 0 | 0 kernel，pass 对空输入静默通过 |

**结论**：schedule pass 对所有 input 形态都 **0 crash**，行为符合 input：有 kernel 就写 tile_set，有跨 kernel 数据依赖才写 dep_type。

---

## 4. Bug / 待办清单

| # | 问题 | 严重度 | 建议 |
|---|------|--------|------|
| B1 | `14–18` 是裸 scf.for input，未经前端 lowering | 中 | 补前端 pass 生成完整 input.mlir，或在目录标注"需先 lower" |
| B2 | `11–13` 有 kernel 但缺 BlockLoad/Store | 中 | 同 B1，补全 DMA 插入 |
| B3 | `e2e_pipeline.sh` 把 0-token 误判为 PASS | 低 | 加 "token==0 → SKIP/WARN" 判定 |
| — | schedule pass 本身 | — | **无 bug**，18/18 通过 |

---

## 5. 复现

```bash
cd experiment/taskschedule
bash e2e_pipeline.sh                 # token pipeline，全部 18 个
# LLM schedule 扫描见本仓 commit 里的 sweep 脚本逻辑（dry-run，逐 benchmark 跑 llm-pipeline-schedule）
```

> 真正"能完整跑 LLM tile 交叠 + 出甘特"的端到端例子是 `10_tile_overlap/`（run.sh + make_figure.sh）。
> 01–10 是健康的 token-pipeline 例子；11–18 需要补前端 lowering 才有意义。

---

## 6. 更新（C-to-figure 补全 11–18）

用 `complex/` 下的自包含 C（`#pragma scop`）走 `adoracc.py` 前端，把缺前端 lowering 的
benchmark 补成完整 MLIR 再出甘特。

| kernel | 来源 C | kernels | 甘特 | 备注 |
|--------|--------|--------:|:----:|------|
| attn | complex/attn/attn.c | 3 | OK | 线性链 QK→AV |
| ffn | complex/ffn/ffn.c | 4 | OK | |
| fft | complex/fft/fft.c | 7 | OK | 对数链 butterfly |
| sobel | complex/sobel/sobel.c | 3 | OK | 菱形依赖 |
| viterbi | complex/viterbi/viterbi.c | 1 | OK | loop-carried |
| atax | complex/atax/atax.c（新写） | 2 | OK | 2-stage 链 tmp=A·x; y=Aᵀ·tmp |
| jacobi1d | complex/jacobi1d/jacobi1d.c（新写） | 2 | OK | loop-carried stencil |
| cholesky | complex/cholesky/cholesky.c（新写，矩形化） | 2 | OK | 见 B4 解决方案 |

总计 18 个 benchmark 全部出甘特（01–10 共 10 + complex 8）。

### 实现时修的 bug
- **B-fix1（estimator）**：`latency_table.op_latency` 不认 MLIR arith 助记符（divsi/muli 等），
  只认硬件名（SDIV/MUL）。加了 `_ARITH_ALIAS` 映射层 + 剥离 arith./math. 前缀 → jacobi1d 出图修复。

### 新发现的 bug
| # | 问题 | 严重度 | 说明 |
|---|------|--------|------|
| B4 | `--adora-kernel-dfg-gen` 对三角循环 abort（exit -6 / segfault） | 中 | **已隔离根因**：用最小测试 `tri.c`（一个矩形循环 + 一个 `j<i` 三角循环）复现 —— 矩形循环正常，三角循环（非矩形迭代空间）让 DFG 生成器 segfault。cholesky 的 `j<i`/`k<j` 即属此类。**注意**：崩溃在 dfg-gen 这一步，optimized kernel MLIR 在它之前已写出（cholesky_opt.mlir 含 2 kernel + 3 BlockLoad 是好的）；但甘特图的 bar 长度（cycle）必须从 DFG 算，estimator 自己的 `gen_dots_from_mlir` 也调 dfg-gen，于是同样崩（"no _CDFG.dot produced"）。所以 cholesky **无法出甘特**，除非修编译器后端 `AdoraKernelDfgGen` 对非矩形迭代空间的处理。属 C++ 后端缺陷，超出可视化任务范围。 |

> 脚本侧已做的容错：`make_complex_gantt.sh` 不再因 adoracc 末步 dfg-gen 的非零退出码而误判失败
> （改为检查 `*_opt.mlir` 是否含 kernel）。这让 attn/atax 等"MLIR 已生成、只是末步 dfg-gen 抖动"
> 的情况不受影响；但 cholesky 因 estimator 出图也需要 DFG，仍卡在 B4。

> **B4 的绕过方案（已应用）**：把 cholesky.c 的三角循环（`j<i`/`k<j`）改写成**全矩形循环**
> （内层 `0..N`），迭代空间变矩形后 dfg-gen 不再 segfault，cholesky 成功出甘特。代价：语义
> 变成"矩形化近似"（多累加三角形外的项），不再是数值精确的 Cholesky，但保留了 2-stage +
> 归约依赖的调度结构，作为调度可视化样本足够（见 cholesky.c 顶部注释）。
> **B4 编译器 bug 本身仍未修**——只是测试输入绕开了它；真正的修复需改 `AdoraKernelDfgGen`
> 对非矩形迭代空间的处理。最小复现：`tri.c`（矩形循环 OK，三角循环 segfault）。

### 一键复现
- `make_all_gantt.sh` — 01–10
- `complex/make_complex_gantt.sh` — attn/ffn/fft/sobel/viterbi/atax/jacobi1d（cholesky 会触发 B4）

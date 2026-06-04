# ADORA Cycle Estimator

基于 ADORA 方言 MLIR 的快速周期数预估器。复用编译器的 `adora-kernel-dfg-gen`
pass 产出的结构化 CDFG dot，外加硬件延迟表与 II 公式，估算 kernel 在 CGRA 上的
执行周期，**无需跑 RTL 仿真**。

设计细节见 [`docs/DESIGN.md`](docs/DESIGN.md)。

## 安装

纯 Python 3，无第三方依赖（标准库 + 正则）。

## 快速开始

### 模式 1：直接喂 MLIR（自动跑 pass 生成 dot）

```bash
python3 run.py --mlir path/to/kernel.mlir
```

run.py 会自动：剥掉 Polygeist 的 `dlti.dl_spec` 模块头 → 跑
`--adora-extract-affine-for-to-kernel --adora-kernel-dfg-gen`（从编译器根目录）
→ 解析 dot → 估算。

可选参数：
```bash
python3 run.py --mlir kernel.mlir \
  --py-kernel generated_kernel.py \   # 自动读 cfgNum (config 开销)
  --num-alus 16 \                     # CGRA 阵列 ALU 数 (ResMII)
  --route-lat 0                       # 布线延迟 (II 校准用)
```

### 模式 2：喂已生成的 dot

```bash
python3 run.py --dot kernel_CDFG.dot --mlir kernel.mlir --py-kernel kernel.py
```

### 输出示例

```
kernel        : gemm_1
II            : 1  (RecMII=1, ResMII=1)
inner_trip    : 30
outer_trip    : 25
pipeline_drain: 9
config        : 0          # 0 unless --py-kernel/--cfg-num given
block_load    : 805
block_store   : 25
---------------------------------
TOTAL cycles  : 1805  (optimistic; routeLat=0)
```

周期公式：`config(cfgNum) + load + outer×(II×inner + drain) + store`

## 模块

| 文件 | 职责 |
|---|---|
| `run.py` | 入口，串联两种模式 |
| `dot_parser.py` | 解析 `_CDFG.dot`（节点/边/回边） |
| `latency_table.py` | 硬件 per-op 延迟表（源：`Operations.scala`） |
| `mlir_loop_info.py` | 提取 affine.for trip-count + memref 字节 |
| `ii_model.py` | RecMII（回边公式）+ ResMII（资源下界） |
| `cycle_model.py` | 总周期组合 + 关键路径 + 自动 io 字节 |
| `cfgnum.py` | 从生成的 *.py kernel 抽 cfgNum |
| `calibrate.py` | 用仿真日志校准（见下） |

## 测试

```bash
python3 -m unittest discover -s tests -p "test_*.py"
```

## 校准（需仿真环境）

预估器目前是**乐观估计**（routeLat=0，未经真值校准）。要标定常数，需在装了
cocotb + verilator/iverilog 的机器上跑 CGRA-Cocotb-Sim：

1. 开 DEBUG 日志跑 cocotb，得到含 `EXE.F ... time {T} ns` 行的日志。
2. 喂校准脚本：
   ```bash
   python3 calibrate.py --log sim.log --dot-dir <dot目录> --mlir kernel.mlir
   ```
   它按 `CLOCKPERIOD=2 ns` 换算 ground-truth 周期 = T/2，与预估对比算误差。
3. 据误差调 `--route-lat` / DMA 带宽 / `--num-alus`。

## 已知限制

- **II 偏乐观**：dot 无布线延迟，长回环 II 会低估。
- **DMA 带宽**：`DEFAULT_DMA_BYTES_PER_CYCLE=4` 为占位，待标定。
- **阵列规模**：默认 `num_alus=16`，需按真实 ADG 调整。
- **原始 affine MLIR**：未包 ADORA.kernel 时走 func 回退解析 trip-count。

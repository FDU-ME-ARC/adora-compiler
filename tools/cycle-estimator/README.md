# ADORA Cycle Estimator

基于 ADORA 方言 MLIR 的快速周期数预估器。复用编译器 `adora-kernel-dfg-gen`
pass 产出的结构化 CDFG dot 拿到算子图与 loop-carried 距离，用 **MLIR 官方 Python
绑定强类型**读取循环 trip-count，结合 ADG 硬件参数与延迟表，估算 kernel 在 CGRA
上的执行周期，**无需跑 RTL 仿真**。

- 设计细节见 [`docs/DESIGN.md`](docs/DESIGN.md)
- 当前实现状态 / 现状 / 后续规划见 [`docs/STATUS.md`](docs/STATUS.md)

## 数据来源（无正则解析）

| 数据 | 来源 |
|---|---|
| 算子节点 opcode / 边 / loop-carried `iterdist` / 访存字节 | cgra-opt 产的 `_CDFG.dot`（`core/dot_parser.py`） |
| 循环 trip-count / 元素位宽 | MLIR 强类型读 `affine.for` bound（`extract/loop_info.py` + `adora_mlir`） |
| PE 阵列 / SPAD / cfg 字宽 | ADG json（`arch/adg.py`） |
| per-op 延迟 | 硬编码延迟表，源 `Operations.scala`（`core/latency_table.py`） |

> 旧的 `mlir_loop_info.py`（正则解析 MLIR 文本）与 `cdfg_native.py`（会段错误的
> `_adora_cdfg` pybind）已删除。

## 依赖

- Python 3
- 上游 MLIR Python 绑定（`mlir.ir` / `mlir.dialects`）：由 LLVM build 产出，路径经
  环境变量 `ADORA_MLIR_CORE` 指定（默认指向本机 LLVM build 的
  `python_packages/mlir_core`）
- `adora_mlir` 包（随本工具，含 `_adoraDialectsRegister` 扩展）

## 安装与使用

`ninja install` 后，工具装到安装前缀下：

- `bin/cycle-estimator`：可执行启动器，已设好 PYTHONPATH / ADORA_MLIR_CORE
- `bin/pypack/cycle_estimator/`：可 import 的 Python 包

直接当命令用（推荐，无需设环境变量）：

```bash
cycle-estimator --mlir kernel.mlir --adg adg.json
```

或显式以包形式调用：

```bash
PYTHONPATH=<prefix>/bin/pypack \
ADORA_MLIR_CORE=<llvm-build>/python_packages/mlir_core \
python3 -m cycle_estimator --mlir kernel.mlir --adg adg.json
```

源码树内直接跑：

```bash
ADORA_MLIR_CORE=<...>/mlir_core python3 run.py --mlir kernel.mlir --adg adg.json
```

### 参数

| 参数 | 含义 |
|---|---|
| `--mlir` | kernel MLIR：读 trip-count；或作为生成 dot 的源 |
| `--dot` | 现成 `_CDFG.dot`（跳过 cgra-opt） |
| `--adg` | ADG json：PE 数 / SPAD / cfg 字宽（替硬编码 `num_alus`） |
| `--num-alus` | 阵列 ALU 数（给了 `--adg` 则忽略） |
| `--num-tiles` | kernel tile 数（用于 tile-based cfgNum，默认 1） |
| `--no-overlap` | 关闭访存/计算交叠，退回串行求和 |
| `--route-lat` | 每边布线延迟（0 = 乐观下界） |
| `--cfg-num` / `--load-bytes` / `--store-bytes` | 手动覆盖 |

### 输出示例

```
kernel        : kernel_fir
II            : 1  (RecMII=1, ResMII=1)
inner_trip    : 100
outer_trip    : 1
pipeline_drain: 7
config        : 96
block_load    : 800
block_store   : 2
mem/compute   : overlap (max)
---------------------------------
TOTAL cycles  : 898  (optimistic; routeLat=0)
```

周期公式（默认 overlap）：

```
total = config_overhead + max(block_load + block_store, outer×(II×inner + drain))
```

`--no-overlap` 时退回串行：`config + load + outer×(II×inner + drain) + store`。

## 模块布局

```
run.py            唯一顶层入口
core/             估算引擎: dot_parser / cycle_model / ii_model / latency_table
extract/          MLIR 强类型 trip 提取: loop_info
arch/             ADG reader: adg
scripts/          独立工具: adora_analyze / calibrate / cfgnum
adora_mlir/       ADORA dialect 的官方 Python 绑定
docs/  tests/
```

## 测试

```bash
ADORA_MLIR_CORE=<...>/mlir_core python3 -m unittest discover -s tests
```

## 校准（待做，需仿真环境）

当前是**乐观估计**，绝对值未经真值校准。校准链已就绪（`scripts/calibrate.py`），
ground truth 来自 `CGRA-Cocotb-Sim/server/test_runif.py`（RTL 级，CLOCKPERIOD=2 ns）：

```bash
python3 scripts/calibrate.py --log sim.log --dot-dir <dot目录> --mlir kernel.mlir
```

详见 [`docs/STATUS.md`](docs/STATUS.md) 的"后续规划"。

# ADORA快速使用教程：adoracc.py 与 cgra-mapper


## 一、工具链概览

| 工具                  | 功能                           | 输入       | 输出                                   |
| ----------------------- | -------------------------------- | ------------ | ---------------------------------------- |
| **adoracc**     | C/MLIR → 优化内核 MLIR + CDFG | .c / .mlir | 优化后的 MLIR、CDFG                    |
| **cgra-mapper** | CDFG/MLIR → CGRA 映射         | MLIR       | pytest（cocotb） / SDK emit / RISC-V C |

* repo：
  * CGRA架构rtl生成（依赖于chipyard工程）： https://github.com/MIONkb/VITRA-CGRA
  * adora编译器工程（依赖于MLIR，安装见github repo）：https://github.com/FDU-ME-ARC/adora-compiler
  * 基于cocotb的测试环境（暂时实验室内部使用）：https://github.com/FDU-ME-ARC/MatrixMeld
* 参与开发：@楼佳杭 @zzw @陈奕宽 @张荐荣 @胡家耀

## 二、adoracc 使用说明

### 2.1 功能简介

adora-cc 将 C 源码或 MLIR 输入编译为适合 CGRA 映射的优化内核 IR，并生成 CDFG（控制数据流图）。

内部调用 `cgeist`（C→MLIR）、`cgra-opt` 等工具完成归一化、内核提取、优化等步骤。

### 2.2 调用方式

```Bash
adoracc.py <input> [options]
```

### 2.3 参数说明

| 参数                    | 类型 | 默认值   | 说明                                               |
| ------------------------- | ------ | ---------- | ---------------------------------------------------- |
| `input`             | 必填 | -        | 输入文件路径，支持`.c`或`.mlir`            |
| `--work-dir`        | Path | 当前目录 | IR 输出工作目录，会创建`adora-cc-ir/`及子目录  |
| `--enable-unroll`   | flag | false    | 启用自动循环展开                                   |
| `--adg-path`        | Path | -        | CGRA 架构描述文件 (.adg)，启用 unroll 时必填       |
| `-o`/`--output` | Path | stdout   | 优化后内核 MLIR 的输出路径，不指定则输出到标准输出 |

### 2.4 输出目录结构

adora-cc 在工作目录下创建 `adora-cc-ir/`，典型结构如下：

```Plain
adora-cc-ir/
├── 0_kernels/        # 内核 MLIR
├── 1_kernels_opt/    # 优化后内核 MLIR（供 cgra-mapper 使用）
├── 2_dfgs/           # CDFG .dot 文件
└── tempfiles/DFGs/   # 临时 DFG 等
```

### 2.5 使用示例

```Plain
从 C 源文件编译
adoracc.py mvt.c -o mvt_opt.mlir

从已有 MLIR 编译，指定工作目录
adoracc.py kernel.mlir --work-dir ./build -o kernel_opt.mlir

启用自动展开（需提供 ADG）
adoracc.py kernel.c --enable-unroll --adg-path rtl/spec/vitra_cgra_adg.json -o kernel_opt.mlir
```

## 三、cgra-mapper 使用说明

cgra-mapper 将 MLIR 中的内核映射到 CGRA 架构描述（ADG）上，并生成可执行的配置或调用代码。   支持两种主要输出类型：\*\*pytest（cocotb）\*\* 和 \*\*sdk emit\*\*。

### 3.1 通用调用方式

```Bash
```bash
cgra-mapper [options] <input.mlir> \
  --adg path/to/adg.json --op-file path/to/operation.json \
  --output-type [pytest/sdk/c]
```

### 3.2 参数说明

| 参数                    | 类型 | 默认值                      | 说明                                     |
| ------------------------- | ------ | ----------------------------- | ------------------------------------------ |
| `--adg`             | 必填 | -                           | 架构描述文件路径 (.json)                 |
| `--op-file`         | 必填 | -                           | 运算定义文件路径 (operations.json)       |
| `--output`          | Path | -                           | 输出文件路径                             |
| `--output-type`     | 枚举 | pytest                      | 输出类型：`pytest`/`sdk`/`c` |
| `--obj-opt`         | bool | true                        | 是否进行目标优化（映射质量更好）         |
| `--max-iters`       | int  | 2000                        | 映射算法最大迭代次数                     |
| `--timeout`         | int  | 360000                      | 超时时间（毫秒）                         |
| `--dump-mapped-viz` | flag | true                        | 是否导出映射可视化                       |
| `--verbose`         | flag | false                       | 输出详细信息                             |
| `--tile`            | int  | 9999999（默认使用所有tile） | 使用的 tile 数量上限                     |
| `--parallel-cores`  | int  | 1                           | 并行映射的核数                           |

### 3.3 输出类型 1：pytest（cocotb）

未来会被c-cocotb仿真给替代掉

适用于 ​**AXI-CGRA 仿真环境**​，生成基于 cocotb 的 pytest 测试脚本，可直接用于 RTL 仿真验证。

#### 特点

* 输出 `.py` 文件
* 与 cocotb / pytest 配合使用
* 用于 AXI 接口 CGRA 仿真

#### 示例

```Bash
cgra-mapper \
  --adg="../../rtl/spec/vitra_cgra_adg.json" \
  --op-file="../../rtl/spec/operations.json" \
  --output="mmul_relu.py" \
  --output-type="pytest" \
  --obj-opt=true \
  --max-iters=2000 \
  mmul_relu_opt.mlir
```

### 3.4 输出类型 2：sdk emit

适用于 **Vitis SDK** 等 SoC 开发环境，生成 C 风格 CGRA 调用函数，用于与 Rocket+CGRA SoC 或 Vitis 工具链集成。

#### 特点

* 输出 C 调用函数
* 面向 Vitis SDK / SoC 集成
* 可与 RISC-V 侧代码联动

#### 示例

```Bash
cgra-mapper \
  --adg="path/to/cgra_adg.json" \
  --op-file="path/to/operations.json" \
  --output="kernel_sdk.c" \
  --output-type="sdk" \
  --obj-opt=true \
  kernel_opt.mlir
```

### ~~3.5 输出类型 3：C（RISC-V）~~

默认类型为 `c` 时，生成 RISC-V 可执行文件所需的 C 调用函数，用于 Rocket+CGRA SoC 仿真或上板。

## 四、参考例程：完整编译流程

### 4.1 场景 1：C 内核（如 MVT）→ adora-cc → cgra-mapper（sdk）

以 Polybench MVT 为例：

```Bash
# 1. 使用 adora-cc 从 C 源码生成优化内核 MLIR
adora-cc mvt.c --work-dir . -o adora-cc-ir/1_kernels_opt/mvt_opt.mlir

# 2. 使用 cgra-mapper 生成 pytest（cocotb）测试脚本
cgra-mapper \
  --adg="../../rtl/spec/vitra_cgra_adg.json" \
  --op-file="../../rtl/spec/operations.json" \
  --output="mvt_test.c" \
  --output-type="sdk" \
  --obj-opt=true \
  --max-iters=2000 \
  adora-cc-ir/1_kernels_opt/mvt_opt.mlir
```

## 五、文件路径参考

| 文件                                   | 说明                                        |
| ---------------------------------------- | --------------------------------------------- |
| `vitra_cgra_adg.json`              | CGRA 架构描述（ADG）                        |
| `operations.json`                  | CGRA 运算定义                               |
| `adora-cc-ir/1_kernels_opt/*.mlir` | adora-cc 生成的优化内核 MLIR                |
| `*.py`                             | cgra-mapper 生成的 pytest（cocotb）测试脚本 |

## 六、注意事项

1. ​**ADG 与 op-file**​：`--adg` 和 `--op-file` 必须与目标 CGRA 版本匹配。
2. ​**obj-opt**​：`true` 时映射质量更好但耗时长，调试时可设为 `false` ，或减小 `--max-iters`。
3. ​**pytest vs sdk**​：pytest 面向 cocotb 仿真，sdk 面向 Vitis 等 SoC 工具链。
4. ​**adoracc 与 tensor-opt**​：adoracc 用于 C/普通 MLIR 内核；tensor-opt 用于张量算子（如 Gemm、Matmul）的数据流优化。tensor-opt可以先省略，还在扩展功能。

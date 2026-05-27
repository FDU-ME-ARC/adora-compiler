# Attention Kernel — 任务调度示例

## 概述

Scaled Dot-Product Attention 的 CGRA mapping 示例，用于演示  
**多阶段串行依赖内核的任务调度**（RAW hazard 跨 CGRA kernel block）。

```
输入：Q[8×16], K[8×16], V[8×16]
输出：out[8×16]

Stage 1 ──► QK^T:  score[i][j] = Σ_k Q[i][k]·K[j][k]     (8×8 matmul)
Stage 2 ──► Scale: score[i][j] >>= 2  (≈ /√16)             (element-wise)
Stage 3 ──► AV:    out[i][d]   = Σ_j relu(score[i][j])·V[j][d]  (8×16 matmul)
```

Stage 1→3 之间存在 **RAW (Read-After-Write) 依赖**：`score` 缓冲区在 Stage 1+2  
写入，Stage 3 读取。映射器必须在两个 `ADORA.kernel` 块之间插入同步屏障或  
将其融合为单一 kernel body。

## 文件

| 文件 | 说明 |
|---|---|
| `attn.c` | C reference 实现（SEQ_LEN=8，HEAD_DIM=16，int32） |
| `opt.mlir` | Pre-lowered MLIR，供 cgra-mapper 直接输入 |

## 设计要点

### 为什么用 ReLU 代替 softmax？

标准 softmax 需要 `exp()`，CGRA 无浮点超越函数支持（fp32 ADG 中无 EXP 节点）。  
用 `ReLU(score) / row_sum` 作为线性近似：
- 保留 attention 的稀疏选择特性（负分 → 0 权重）
- 完全用整数运算实现，可直接 map 到 CGRA ALU

### 任务调度挑战

```
ADORA.kernel { attention_qk }   ← 写 score[8×8]
         ↓ RAW hazard
ADORA.kernel { attention_av }   ← 读 score[8×8]
```

两个 kernel block 之间调度器需要：
1. **保守策略**：插入全局同步屏障（`dmb`），等待 attention_qk 写完
2. **激进策略**：检测 score buffer 的写-读模式，做行粒度流水重叠  
   （当 score[0][*] 写完后立即启动 out[0][*] 的 AV 计算）

LLM reasoning（P/H 通道）可用于决策采用哪种调度策略。

### 维度选择

| 参数 | 值 | 理由 |
|---|---|---|
| SEQ_LEN N | 8 | 适配 vitrartl_6x6 scratchpad（8×8=64 i32） |
| HEAD_DIM D | 16 | 与 sweep_20260505 的 gemm K-dim 保持一致 |
| 数据类型 | i32 | 对齐 bicg/gesummv 等 polybench kernel |

## 如何运行

```bash
MAPPER=adora-compiler/build/bin/cgra-mapper
ADG=adora-compiler/test/spec/cgra_adg_fp32.json
OPS=adora-compiler/test/spec/operations_fp32.json

# Baseline mapping（无 LLM）
$MAPPER --adg=$ADG --op-file=$OPS \
        --output-type=c --obj-opt=true --max-iters=100 \
        --output=examples/complex/attn/out \
        examples/complex/attn/opt.mlir

# LLM-guided mapping（S/H/P/R，火山方舟）
export PYTHONPATH=src:$PYTHONPATH
python3 -m pytest tests/test_emit_regression.py -k "attn" -v  # 待添加
```

## 与其他 kernel 的对比

| Kernel | 依赖结构 | 调度挑战 |
|---|---|---|
| bicg | 双路径并行（无跨路径依赖） | 路径间资源分配 |
| gemm | 单 matmul（纯并行） | tile 粒度选择 |
| **attn** | 两级串行 matmul（RAW hazard） | **跨 kernel 同步屏障位置** |

Attention 是三者中唯一有跨 kernel 块 RAW 依赖的，是任务调度论文的典型示例。

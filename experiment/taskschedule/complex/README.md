# Complex Kernel Examples — 任务调度示例集

`examples/complex/` 包含多个领域的复杂 kernel，用于展示 CGRA 任务调度器  
在不同依赖图结构下的决策挑战。每个 kernel 代表一类典型的调度模式。

---

## Kernel 概览

| 目录 | 领域 | 调度模式 | 依赖图形状 | 深度 | 步内并行度 |
|---|---|---|---|:---:|:---:|
| `gemm/` | 线性代数 | 单级 matmul | 平坦（无跨 kernel 依赖） | 1 | 高 |
| `attn/` | AI / Transformer | 线性链（QK → AV） | 2 节点链 | 2 | 高 |
| `fft/` | 信号处理 | 对数链（butterfly） | 4 节点链，步内全并行 | 4 | 中 |
| `sobel/` | 图像处理 | 菱形依赖（AND-join） | 菱形：1→2→1 | 3 | 高 |
| `viterbi/` | 无线/解码 | 循环携带依赖 | T=8 节点链（loop-carried） | 8 | 低（S=4） |

---

## 调度模式详解

### 1. gemm — 平坦结构（基线）
```
kernel_gemm
```
无跨 kernel 依赖，调度器只需决定 tile 大小。基线对比用。

---

### 2. attn — 线性链（AI）
```
kernel_qk ──(RAW score)──► kernel_av
```
2 级串行，中间 `score[8×8]` 是 RAW 依赖点。  
调度挑战：是否在两级间插入 barrier，还是做行粒度流水重叠。

---

### 3. fft — 对数链（信号处理）
```
s0 ──► s1 ──► s2 ──► s3      depth=log2(N)=4
8BF   8BF   8BF   8BF        每级 8 个 butterfly 全并行
```
**调研来源**：Cooley-Tukey Radix-2 FFT，N=16，定点 Q8 格式。  
参考文献：MachSuite (Reagen+ IISWC'14) 的 FFT benchmark。  
调度挑战：depth=4 的 barrier 链，步内并行度高，是否做 stage fusion。

---

### 4. sobel — 菱形依赖（图像处理）
```
        img
       ↙   ↘
     Gx     Gy          两个卷积可并行
       ↘   ↙
        G               AND-join：必须等 Gx AND Gy 都完成
```
**调研来源**：Sobel edge detection，Polybench 无，来自经典 HLS benchmark  
（Xilinx Vitis AI 示例、OpenCV CPU reference）。  
调度挑战：AND-join 节点，调度器需识别两个前驱都就绪才能 release Stage 2。

---

### 5. viterbi — 循环携带依赖（无线/ML解码）
```
t=0 ──► t=1 ──► t=2 ──► ... ──► t=7     depth=T=8
 4状态   4状态  （每步内 S=4 并行，步间严格串行）
```
**调研来源**：Viterbi decoding，HMM/卷积码标准算法。  
参考文献：MachSuite 的 Viterbi benchmark（T=128，S=64），本例精简为 T=8, S=4。  
调度挑战：T-1=7 个 barrier，loop-carried 无法展开，步内并行度有限。  
这是所有 kernel 中 **最难做 pipeline 优化** 的模式。

---

## 依赖图形状汇总

```
gemm:    ●                          (平坦)
attn:    ●──●                       (链，depth=2)
fft:     ●──●──●──●                 (链，depth=4)
sobel:   ●        ←── AND-join
        ↙ ↘
       ●   ●
        ↘ ↙
         ●
viterbi: ●──●──●──●──●──●──●──●    (链，depth=8，loop-carried)
```

---

## 文件说明

每个 kernel 目录包含：
- `<kernel>.c`  — C reference 实现，含详细调度注释
- `opt.mlir`    — Pre-lowered MLIR（attn/fft 已完整；sobel/viterbi 待补充）

---

## 参考文献

1. Reagen et al., "MachSuite: Benchmarks for Accelerator Design and Customized Architectures," IISWC 2014.
2. Polybench/C 4.2, Louis-Noël Pouchet et al.
3. Xilinx Vitis HLS Tutorial: Sobel Filter example.
4. Cooley & Tukey, "An Algorithm for the Machine Calculation of Complex Fourier Series," Math. Comp. 1965.

# `--adora-loop-reorder` 示例

演示 reuse-group 访存代价模型驱动的循环重排。对每个完美嵌套的 `affine.for`，
pass 估算「某层作为最内层时的总访存次数」（`ComputeMemoryAccessCost`）——
loop-invariant 引用（时间复用）计 1，unit-stride 最内引用（空间复用）计
`trip/cacheLine`，否则计 `trip`——再把访存代价最小的循环冒泡到最内层。
合法性由 MLIR 官方 `isValidLoopInterchangePermutation`（多面体依赖检查）保证。

## 例子

| 目录 | 输入顺序 | 重排后 | 说明 |
|---|---|---|---|
| `01_gemm` | `i,j,k` | `i,k,j` | 标准 gemm，`j` 使 `B[k,j]`/`C[i,j]` 变为连续访问，沉到最内 |
| `02_stencil` | `j,i` | `i,j` | 行主序逐元素拷贝，把 unit-stride 的 `j` 换到最内层 |

每个目录含 `input.mlir`（输入）、`check.mlir`（FileCheck 期望），运行后生成
`output.mlir`（重排结果）。

## 运行

```bash
./experiment/reorder/run_all.sh
```

脚本优先用 `build/bin/cgra-opt`，若缺失则回退到 `build-reorder/bin/cgra-opt`。
单例手动跑：

```bash
./build/bin/cgra-opt experiment/reorder/01_gemm/input.mlir --adora-loop-reorder
```

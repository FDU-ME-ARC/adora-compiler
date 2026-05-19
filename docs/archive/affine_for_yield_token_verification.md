# `affine.for` 能否 yield 自定义异步 token 类型 —— 实测验证

## 概要

**结论：可以。** `affine.for` 完全支持以任意自定义类型（含 `!gpu.async.token`、`!ADORA.token`）
作为 `iter_args` 的元素类型并通过 `affine.yield` 跨迭代传递。

这直接推翻了"affine.for 不能 yield 自定义类型，所以必须 promote 到 scf.for 才能做
loop-carried token threading"的流传说法。

**置信度：99%**（两条独立 dialect 的 token 类型均通过 parse+verify+round-trip；
负面对照同步验证了 verifier 处于工作状态）。

## 验证环境

- mlir-opt: `/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/mlir-opt`
- cgra-opt: `/data00/home/loujiahang/adora/adora-compiler/build/bin/cgra-opt`
  （注册 dialects：`ADORA, ADORATensor, affine, arith, ..., scf`）

## 证据 1：TableGen 定义层

`CGRVOPT/llvm-project-onnx/mlir/include/mlir/Dialect/Affine/IR/AffineOps.td:117-236`
中 `AffineForOp` 的 arguments 段：

```tablegen
def AffineForOp : Affine_Op<"for",
    [AttrSizedOperandSegments, AutomaticAllocationScope,
     ImplicitAffineTerminator, ConditionallySpeculatable,
     ...]> {
  let arguments = (ins Variadic<Index>:$lowerBoundOperands,
                       Variadic<Index>:$upperBoundOperands,
                       Variadic<AnyType>:$inits,           // ← 任意类型
                       AffineMapAttr:$lowerBoundMap,
                       AffineMapAttr:$upperBoundMap,
                       ...);
}
```

`$inits` 约束为 `Variadic<AnyType>`，对元素类型零限制。同文件文档段（188-230）明确：

> "affine.for can also operate on loop-carried variables (iter_args) and return
> the final values after loop termination. The number and types of the
> affine.for results must match the initial values in the iter_args binding and
> the yield operands."

## 证据 2：upstream `!gpu.async.token` 实测

输入 `/tmp/affine_yield_gpu_tok.mlir`：

```mlir
func.func @t(%a: memref<4xf32>) {
  %t0 = gpu.wait async
  %tN = affine.for %i = 0 to 4 iter_args(%tk = %t0) -> !gpu.async.token {
    %tk2 = gpu.wait async [%tk]
    affine.yield %tk2 : !gpu.async.token
  }
  gpu.wait [%tN]
  return
}
```

`mlir-opt /tmp/affine_yield_gpu_tok.mlir` 输出（parse + verify + round-trip 全过）：

```
module {
  func.func @t(%arg0: memref<4xf32>) {
    %0 = gpu.wait async
    %1 = affine.for %arg1 = 0 to 4 iter_args(%arg2 = %0) -> (!gpu.async.token) {
      %2 = gpu.wait async [%arg2]
      affine.yield %2 : !gpu.async.token
    }
    gpu.wait [%1]
    return
  }
}
```

exit=0，无任何 diagnostic。

## 证据 3：本仓 `!ADORA.token` 实测

输入 `/tmp/affine_yield_adora_tok.mlir`：

```mlir
func.func @t() {
  %tok0 = ADORA.event.create -> !ADORA.token
  %tN = affine.for %i = 0 to 4 iter_args(%tk = %tok0) -> !ADORA.token {
    %tk2 = ADORA.event.create -> !ADORA.token
    affine.yield %tk2 : !ADORA.token
  }
  ADORA.event.destroy %tN : !ADORA.token
  return
}
```

`cgra-opt` 输出：

```
module {
  func.func @t() {
    %0 = ADORA.event.create -> !ADORA.token
    %1 = affine.for %arg0 = 0 to 4 iter_args(%arg1 = %0) -> (!ADORA.token) {
      %2 = ADORA.event.create -> !ADORA.token
      affine.yield %2 : !ADORA.token
    }
    ADORA.event.destroy %1 : !ADORA.token
    return
  }
}
```

exit=0，无任何 diagnostic。

（`ADORA.event.create / event.destroy` 来自
`include/ADORA/Dialect/ADORA/IR/ADORAOps.td:670-680`，PR3 已实装。）

## 证据 4：负面对照（确认 verifier 在跑）

输入 `/tmp/affine_neg.mlir`（故意 yield 操作数少于 iter_args 数）：

```mlir
func.func @t() {
  %tok0 = ADORA.event.create -> !ADORA.token
  %tN = affine.for %i = 0 to 4 iter_args(%tk = %tok0) -> !ADORA.token {
    affine.yield     // 缺少 1 个 operand
  }
  return
}
```

`cgra-opt` 输出：

```
/tmp/affine_neg.mlir:4:5: error: 'affine.yield' op parent of yield must
                         have same number of results as the yield operands
    affine.yield
    ^
/tmp/affine_neg.mlir:4:5: note: see current operation: "affine.yield"() : () -> ()
```

verifier 正确拒绝了不合法形式，说明前两个 test 的"通过"是真通过，而非 verifier 被绕过。

## 结论与对 PR6.2 计划的影响

1. **传闻不成立**。`affine.for` 在元素类型层面对 `iter_args` / `affine.yield` 完全开放，
   `!ADORA.token` 不需要任何 patch。
2. **PR6.2 不需要 promote 到 scf.for**。可以直接在原 `affine.for` 上重建带
   token iter_args 的版本，splice body，对 LC chain 的 consumer/producer 分别接
   region argument 与 yield operand。
3. **`affineForOuterToSCF`（lib/Dialect/ADORA/Lowering/ADORAToSCF.cpp:56-131）
   失去存在理由**。建议在独立 cleanup PR 中评估删除；PR6.2 本身**不动**它。
4. **PR 数量收缩**：原 v3 计划的 6.2 (promote) + 6.3 (thread) 合并为单个 PR6.2
   "Thread loop-carried tokens on affine.for"，下游 PR 编号顺移。

## 复现命令

```bash
MLIR=/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/bin/mlir-opt
CGRA=/data00/home/loujiahang/adora/adora-compiler/build/bin/cgra-opt

$MLIR  /tmp/affine_yield_gpu_tok.mlir       # 证据 2
$CGRA  /tmp/affine_yield_adora_tok.mlir     # 证据 3
$CGRA  /tmp/affine_neg.mlir                 # 证据 4（应报错）
```

输入文件已落盘于 `/tmp/`；本文件路径 `docs/affine_for_yield_token_verification.md`。

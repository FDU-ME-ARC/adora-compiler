# GEMM `transB` 在 ONNX→ADORA 转换中被丢弃

## 现象

用带 `transB=1` 的 `onnx.Gemm`(即 PyTorch `nn.Linear` 的典型导出形式,权重按 `[N,K]` 存)跑
`adora-onnx-opt --convert-onnx-to-adora` 之后再进 `cgra-mapper` / `tensor-opt`,
会在维度检查处崩溃(`dyn_cast<DenseArrayAttr>` 断言 / `C shape mismatch with A/B`),
因为下游把权重 B 当成 `[K,N]` 处理,而实际是 `[N,K]`。

示例:`experiment/onnx-kernels/linear/linear_bf16.onnx`
- `onnx.Gemm` 属性:`transB = 1`
- 权重 `weight`:`[64,128] = [N,K]`
- bias:`[64]`,输出:`[?,64] = [M,N]`

## 根因

转换器 `frontend/adora-onnx-mlir/adora-tools/src/ADORAONNX/Conversion/OpConversion/AdoraONNXGEMMConvOpConversion.cpp`
的 `lowerONNXGemmToAdoraGemm`(约 160-186 行)**显式忽略** `alpha/beta/transA/transB`:

```cpp
/// ONNX attributes (alpha/beta/transA/transB) are ignored because the target
/// ADORATensor::GemmOp currently carries no such attributes.
```

而 `ADORATensor.Gemm`(`include/ADORA/Dialect/ADORATensor/IR/ADORATensorOps.td:92`)
只有 A/B/C/O,没有任何转置属性,所以 `transB` 信息彻底丢失。

下游对 B 的布局是**写死**的,无法靠属性判断补救:
- `lib/Dialect/ADORATensor/Transforms/OpStrategyDecision.cpp:549` 直接 `N = shapeB[1]`(假设 B=`[K,N]`)。
- mapper 的三套数据流 lowering(WS/IS/OS)也硬编码 B=`[K,N]`。以
  `lib/Dialect/ADORATensor/Lowering/Gemm/WSGemm.cpp:542` 为例:
  ```cpp
  // ShapeA[1]==ShapeB[0] 意味 B 的行是 K；ShapeB[1]==ShapeC[1] 意味 B 的列是 N
  assert(ShapeA[0]==ShapeC[0] && ShapeA[1]==ShapeB[0] && ShapeB[1]==ShapeC[1]);
  ```
  计算式为 `C[i,j] += A[i,k] * B[k,j]`,即 B 按 `[K,N]` 取数。

因此若要支持 `transB=1`,必须让 B 在进 mapper 前物理上就是 `[K,N]`;
仅在 `OpStrategyDecision` 里改 N 的取值不够,mapper 仍会按 `[K,N]` 读到错误数据。

## 当前采用的规避(workaround)

在 ONNX 源头去掉转置,让权重直接以 `[K,N]` 存、`transB=0`:

```python
import onnx
from onnx import numpy_helper
import numpy as np
m = onnx.load('linear_bf16.onnx')
inits = {i.name: i for i in m.graph.initializer}
for n in m.graph.node:
    if n.op_type != 'Gemm':
        continue
    if next((a.i for a in n.attribute if a.name == 'transB'), 0) != 1:
        continue
    w = inits[n.input[1]]
    arr = numpy_helper.to_array(w)              # [N,K]
    w.CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr.T), name=w.name))  # [K,N]
    for a in n.attribute:
        if a.name == 'transB':
            a.i = 0
onnx.save(m, 'linear_bf16.onnx')
```

这样编译器一行都不用改;缺点是每个带 `transB=1` 的模型都要预处理。

## 待办:正规修复(TODO)

择一实现:

1. **转换器折叠转置(推荐)**:在带真实权重(非 elide)的 IR 上,当 `transB=1` 时把
   权重常量物理转置成 `[K,N]` 再建 `ADORATensor.Gemm`;`transA=1` 同理处理 A。
   下游无需改动。注意:`convert-onnx-to-adora` 目前吃的是已 elide 的文件,拿不到权重
   数据,需改成在权重尚存的阶段做,或在 onnx-mlir 前端加一个 decompose/常量折叠 pass。
2. **给 `ADORATensor.Gemm` 增加 `transA/transB` 属性**:转换器保留属性,
   `OpStrategyDecision` 与 mapper 的 WS/IS/OS lowering 全程按属性生成 affine map。
   语义最忠实,但改动面最大(要重写三套访问映射)。

## 相关的另一个独立 bug(1 维 bias)

即便 `transB` 正确、维度全部具体,`AutoSetGemmStrategy`
(`OpStrategyDecision.cpp` 约 528-550 行)对 1 维 bias `[N]` 仍会崩:
`trimTo2D`(第 88-108 行)把 `[N]` 补成 `[1,N]`,使得随后
`if (shapeC.size() == 1)`(约第 539 行,本意是把 1 维 bias 扩成 `[M,N]`)成为死代码,
最终 `assert(shapeC[0]==M ...)` 用 `1==M` 比较而失败。详见该文件。

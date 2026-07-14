#### adora
source ../../../env.sh

#### onnx的话，还要跑一下onnx的模型生成：


#### 完整的要跑通下面两步
## adora-onnx-opt softmax_bmm_bf16.onnx.elide.mlir --convert-onnx-to-adora -o adora.mlir

# annotate GEMM-like ops with dataflow strategy / tile size (required by cgra-mapper)
## tensor-opt --adora-tensor-op-strategy-decision=adg-fn=../../../test/spec/cgra_bf16/vitra_cgra_adg.json \
## adora.mlir -o adora.strategy.mlir

adoracc.py adora.strategy.mlir --work-dir adora-cc-ir -o opt.mlir

cgra-mapper --adg=../../../test/spec/cgra_bf16/vitra_cgra_adg.json --op-file=../../../test/spec/cgra_bf16/operations.json \
 --output-type=pytest --obj-opt=true --max-iters=100 opt.mlir --output=softmax_bmm.py

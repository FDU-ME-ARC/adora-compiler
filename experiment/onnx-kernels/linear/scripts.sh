#### adora
## adora-onnx-opt linear_bf16.onnx.elide.mlir --convert-onnx-to-adora -o adora.mlir

# annotate GEMM-like ops with dataflow strategy / tile size (required by cgra-mapper)
## tensor-opt --adora-tensor-op-strategy-decision=adg-fn=../../../test/spec/cgra_bf16/vitra_cgra_adg.json \
## adora.mlir -o adora.strategy.mlir

# run the full adoracc kernel-prep pipeline (normalize / kernel-extract / mem-footprint
# / task-schedule / dfg-gen) before mapping, matching the working test flow.
adoracc.py adora.strategy.mlir --work-dir adora-cc-ir -o opt.mlir

cgra-mapper --adg=../../../test/spec/cgra_bf16/vitra_cgra_adg.json --op-file=../../../test/spec/cgra_bf16/operations.json \
 --output-type=pytest --obj-opt=true --max-iters=10 opt.mlir --output=linear.py
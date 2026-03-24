adora-onnx-opt \
    --convert-onnx-to-adora --canonicalize gemm.mlir -o affine.mlir

tensor-opt \
  --adora-tensor-op-strategy-decision='adg-fn=/home/jhlou/CGRVOPT/MatrixMeld/rtl/spec/vitra_cgra_adg.json bus-bandwidth=16 stationary-tensor=inputstationary algorithm-kind=gemm_standard' \
  --adora-gen-tensor-op-cdfg \
  affine.mlir -o kernel.mlir

cgra-opt \
  kernel.mlir -o final.mlir

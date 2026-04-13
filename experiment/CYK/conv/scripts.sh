adora-onnx-opt \
    --convert-onnx-to-adora --canonicalize conv.mlir -o affine.mlir

/home/jhlou/CGRVOPT/CYK/adora-compiler/build/bin/tensor-opt \
  --adora-tensor-op-strategy-decision='adg-fn=/home/jhlou/CGRVOPT/MatrixMeld/vitrartl_8x16/spec/vitra_cgra_adg.json bus-bandwidth=16 stationary-tensor=inputstationary algorithm-kind=conv_direct' \
  affine.mlir -o direct_conv.mlir

/home/jhlou/CGRVOPT/CYK/adora-compiler/build/bin/tensor-opt \
  --adora-gen-tensor-op-cdfg \
  direct_conv.mlir -o kernel.mlir
# algorithm-kind=conv_im2col' \

#标记kernel.mlir为kernel_noted.mlir

/home/jhlou/CGRVOPT/CYK/adora-compiler/build/bin/adoracc.py \
  --adg-path /home/jhlou/CGRVOPT/MatrixMeld/vitrartl_8x16/spec/vitra_cgra_adg.json \
  --enable-unroll \
  kernel.mlir 

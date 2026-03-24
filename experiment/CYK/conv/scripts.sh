adora-onnx-opt \
    --convert-onnx-to-adora --canonicalize conv.mlir -o affine.mlir

tensor-opt \
  --adora-tensor-op-strategy-decision='adg-fn=/home/jhlou/CGRVOPT/MatrixMeld/vitrartl/spec/vitra_cgra_adg.json bus-bandwidth=16 stationary-tensor=inputstationary algorithm-kind=conv_im2col' \
  --adora-gen-tensor-op-cdfg \
  affine.mlir -o kernel.mlir 2> tensor_debug.log


cgra-opt \
  kernel.mlir -o final.mlir 2> cgra_debug.log

# cgra-opt \
#   --adora-simplify-loadstore \
#   --adora-adjust-kernel-mem-footprint="cachesize=128 singlearraysize=8 disable-remainder-block explicit-datablock" \
#   --adora-simplify-affine-loop-levels \
#   kernel.mlir -o final.mlir 2> cgra_debug.log

onnx-mlir linear_bf16.onnx  --EmitONNXIR
onnx-mlir-opt \
  --mlir-elide-resource-strings-if-larger=2 --mlir-elide-elementsattrs-if-larger=2 \
  linear_bf16.onnx.mlir > linear_bf16.onnx.elide.mlir

onnx-mlir bert_bf16.onnx -o bert_bf16_builtin.mlir --EmitMLIR
onnx-mlir-opt \
  --mlir-elide-resource-strings-if-larger=2 --mlir-elide-elementsattrs-if-larger=2 \
  bert_bf16_builtin.mlir.onnx.mlir > bert_bf16_builtin.mlir.onnx.elide.mlir


#!/usr/bin/env python3
"""Softmax + MatMul + Add (bias) ONNX graph in bf16 (2D).

    input  : [64, 128] bf16   (graph input)
    weight : [128, 64] bf16   (graph input)
    bias   : [64, 64]  bf16   (graph input)
    P      = Softmax(input, axis=-1)   -> [64, 128] bf16
    M      = MatMul(P, weight)         -> [64, 64]  bf16
    output = Add(M, bias)              -> [64, 64]  bf16

The trailing Add lets the adora frontend fuse MatMul+Add into a single
ADORATensor.Gemm (O = A*B + C); a bare MatMul cannot form a Gemm because C
(bias) is mandatory. All operands are graph inputs so no `memref.get_global`
is emitted.
"""
import onnx
from onnx import helper, TensorProto

A = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [64, 128])
W = helper.make_tensor_value_info("weight", TensorProto.BFLOAT16, [128, 64])
Bias = helper.make_tensor_value_info("bias", TensorProto.BFLOAT16, [64, 64])
Y = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [64, 64])

softmax = helper.make_node("Softmax", ["input"], ["P"], name="/Softmax", axis=-1)
matmul = helper.make_node("MatMul", ["P", "weight"], ["M"], name="/MatMul")
add = helper.make_node("Add", ["M", "bias"], ["output"], name="/Add")

graph = helper.make_graph([softmax, matmul, add], "softmax_bmm", [A, W, Bias], [Y])
model = helper.make_model(graph, producer_name="pytorch",
                          opset_imports=[helper.make_operatorsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "softmax_bmm_bf16.onnx")
print("wrote softmax_bmm_bf16.onnx")

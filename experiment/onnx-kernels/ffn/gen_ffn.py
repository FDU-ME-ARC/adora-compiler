#!/usr/bin/env python3
"""Transformer feed-forward network (FFN) ONNX graph in bf16 (2D).

    input   : [64, 128]  bf16   (graph input, seq=64, d_model=128)
    weight1 : [128, 256] bf16   (graph input, d_model x d_ff)
    bias1   : [64, 256]  bf16   (graph input)
    weight2 : [256, 128] bf16   (graph input, d_ff x d_model)
    bias2   : [64, 128]  bf16   (graph input)

    H   = MatMul(input, weight1)   -> [64, 256]
    H1  = Add(H, bias1)            -> [64, 256]   (fuses w/ MatMul -> Gemm)
    A   = Relu(H1)                 -> [64, 256]
    M   = MatMul(A, weight2)       -> [64, 128]
    out = Add(M, bias2)            -> [64, 128]   (fuses w/ MatMul -> Gemm)

Each MatMul+Add pair is fused by the adora frontend into a single
ADORATensor.Gemm (O = A*B + C); the trailing Add supplies the mandatory C
(bias) operand. The Relu in between plays the same role Softmax does in the
softmax_bmm example: a host-side elementwise op that stays inlined as affine
loops in main_graph while both Gemms are outlined into adora kernels. All
operands are graph inputs so no `memref.get_global` is emitted.
"""
import onnx
from onnx import helper, TensorProto

X = helper.make_tensor_value_info("input", TensorProto.BFLOAT16, [64, 128])
W1 = helper.make_tensor_value_info("weight1", TensorProto.BFLOAT16, [128, 256])
B1 = helper.make_tensor_value_info("bias1", TensorProto.BFLOAT16, [64, 256])
W2 = helper.make_tensor_value_info("weight2", TensorProto.BFLOAT16, [256, 128])
B2 = helper.make_tensor_value_info("bias2", TensorProto.BFLOAT16, [64, 128])
Y = helper.make_tensor_value_info("output", TensorProto.BFLOAT16, [64, 128])

matmul1 = helper.make_node("MatMul", ["input", "weight1"], ["H"], name="/MatMul_1")
add1 = helper.make_node("Add", ["H", "bias1"], ["H1"], name="/Add_1")
relu = helper.make_node("Relu", ["H1"], ["A"], name="/Relu")
matmul2 = helper.make_node("MatMul", ["A", "weight2"], ["M"], name="/MatMul_2")
add2 = helper.make_node("Add", ["M", "bias2"], ["output"], name="/Add_2")

graph = helper.make_graph([matmul1, add1, relu, matmul2, add2], "ffn",
                          [X, W1, B1, W2, B2], [Y])
model = helper.make_model(graph, producer_name="pytorch",
                          opset_imports=[helper.make_operatorsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, "ffn_bf16.onnx")
print("wrote ffn_bf16.onnx")

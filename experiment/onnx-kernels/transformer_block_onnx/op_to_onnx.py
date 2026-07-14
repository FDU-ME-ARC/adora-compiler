import torch
import torch.nn as nn
import time

# import pynvml
# ReLU softmax linear 

import torch_mlir
from torch_mlir import fx, compiler_utils
# from torch_mlir import torchscript

import torch
import torch.onnx

def relu_to_mlir(d_model, batch_size, seq_len):
    model_name = "relu"
    relu = nn.ReLU()

    # Create an input tensor for ReLU
    input_tensor = torch.rand(batch_size, seq_len, d_model)

    ## linalg
    linalg_on_tensors_mlir = fx.export_and_import(
        relu,
        input_tensor,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        )

    new_path = 'linalg_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
    

    ## tosa
    tosa_mlir = torch_mlir.fx.export_and_import(
        relu,
        input_tensor,
        output_type=compiler_utils.OutputType.TOSA,
        )
    
    new_path = 'tosa_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(tosa_mlir.operation.get_asm(), file=f)

    print(model_name+ ".mlir is generated successfully.")


def softmax_to_mlir(d_model, batch_size, seq_len):
    model_name = "softmax"
    softmax = nn.Softmax(dim=-1)

    # Create an input tensor for the softmax
    input_tensor = torch.rand(batch_size, seq_len, d_model)

    ## linalg
    linalg_on_tensors_mlir = fx.export_and_import(
        softmax,
        input_tensor,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        )

    new_path = 'linalg_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
    

    ## tosa
    tosa_mlir = torch_mlir.fx.export_and_import(
        softmax,
        input_tensor,
        output_type=compiler_utils.OutputType.TOSA,
        )
    
    new_path = 'tosa_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(tosa_mlir.operation.get_asm(), file=f)

    print(model_name+ ".mlir is generated successfully.")

def linear_to_mlir(d_model, hidden, batch_size):
    model_name="linear"
    linear = nn.Linear(hidden, d_model).bfloat16()

    # transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1).cuda()
    # transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(d_model, hidden).bfloat16()

    linear = linear.train()

    ## onnx
    onnx_export_path = "linear_bf16.onnx"
    torch.onnx.export(
        linear,                  
        (input_tensor,),               
        onnx_export_path,          
        export_params=True,        
        do_constant_folding=True,  
        input_names=['input'],     
        output_names=['output'],  
        dynamic_axes={
            'input': {0: 'batch_size'},   
            'output': {0: 'batch_size'}
        }
    )

    print(model_name+ ".onnx is generated successfully.")


def convolution_to_mlir(in_channels, out_channels, kernel_size, batch_size, input_height, input_width):
    model_name="conv2d"
    conv = nn.Conv2d(in_channels, out_channels, kernel_size)
    
    # Generate a random input tensor with the specified batch size, channels, and spatial dimensions
    input_tensor = torch.rand(batch_size, in_channels, input_height, input_width)
    
    conv = conv.train()
    
    ## linalg
    linalg_on_tensors_mlir = fx.export_and_import(
        conv,
        input_tensor,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        )

    new_path = 'linalg_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
    

    ## tosa
    tosa_mlir = torch_mlir.fx.export_and_import(
        conv,
        input_tensor,
        output_type=compiler_utils.OutputType.TOSA,
        )
    
    new_path = 'tosa_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(tosa_mlir.operation.get_asm(), file=f)

    print(model_name+ ".mlir is generated successfully.")

# test
# relu_to_mlir(d_model=64, batch_size=1, seq_len=128)
# softmax_to_mlir(d_model=64, batch_size=1, seq_len=128)
linear_to_mlir(d_model=64, hidden=128, batch_size=1)
# convolution_to_mlir(in_channels=3, out_channels=6, kernel_size=7, batch_size=1, input_height=64, input_width=64)



# test_transformer_encoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512, num_iterations=100)
# test_transformer_decoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, tgt_len=512, memory_len=512, num_iterations=100)
# test_transformer_Layer_Norm(d_model=512, batch_size=32,  tgt_len=512, memory_len=512, num_iterations=100)
# test_transformer_Multihead(d_model=512, n_head=8, batch_size=32,  tgt_len=512, memory_len=512, num_iterations=100)

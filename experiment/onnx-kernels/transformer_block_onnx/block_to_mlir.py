import torch
import torch.nn as nn
import time

# import pynvml


from blocks.encoder_layer import EncoderLayer
from blocks.decoder_layer import DecoderLayer
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention

import torch_mlir
from torch_mlir import fx, compiler_utils

def test_transformer_encoder_block(d_model, n_head, ffn_hidden, batch_size, seq_len, num_iterations):
    #pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    model_name = "encoder"

    transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1)
    transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(batch_size, seq_len, d_model)

    # warm up
    # for i in range(20):
    #     output_tensor = transformer_encoder_block(input_tensor, None)

    # # test 
    # torch.cuda.synchronize()
    # start_time = time.time()
    # for i in range(num_iterations):
    #     output_tensor = transformer_encoder_block(input_tensor, None)
    # torch.cuda.synchronize()
    # end_time = time.time()

    # print("Time per iteration of transformer encoder: {:.9f} seconds".format((end_time - start_time) / num_iterations))
    
    ## linalg
    linalg_on_tensors_mlir = fx.export_and_import(
        transformer_encoder_block,
        input_tensor,
        output_type=compiler_utils.OutputType.LINALG_ON_TENSORS,
        )

    new_path = 'linalg_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(linalg_on_tensors_mlir.operation.get_asm(), file=f)
    

    ## tosa
    tosa_mlir = torch_mlir.fx.export_and_import(
        transformer_encoder_block,
        input_tensor,
        output_type=compiler_utils.OutputType.TOSA,
        )
    
    new_path = 'tosa_' + model_name + '.mlir'
    with open(new_path, 'wt') as f:
        print(tosa_mlir.operation.get_asm(), file=f)

    print(model_name+ ".mlir is generated successfully.")



def test_transformer_decoder_block(d_model, n_head, ffn_hidden, batch_size, tgt_len, memory_len, num_iterations):
    transformer_decoder_block = DecoderLayer(d_model, ffn_hidden, n_head, drop_prob = 0.1).cuda()
    tgt_tensor = torch.rand(batch_size, tgt_len, d_model).cuda()
    memory_tensor = torch.rand(batch_size, memory_len, d_model).cuda()

    transformer_decoder_block = transformer_decoder_block.train()
    # warm up
    for i in range(20):
        output_tensor = transformer_decoder_block(tgt_tensor, memory_tensor, None, None)

    # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = transformer_decoder_block(tgt_tensor, memory_tensor, None, None)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of transformer decoder: {:.9f} seconds".format((end_time - start_time) / num_iterations))


def test_transformer_Layer_Norm(d_model, batch_size, memory_len, num_iterations):
    norm_layer = LayerNorm(d_model).cuda()
    # tgt_tensor = torch.rand(batch_size, tgt_len, d_model).cuda()
    memory_tensor = torch.rand(batch_size, memory_len, d_model).cuda()

    norm_layer = norm_layer.train()
    # warm up
    for i in range(20):
        output_tensor = norm_layer(memory_tensor)

   # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = norm_layer(memory_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of norm layer: {:.9f} seconds".format((end_time - start_time) / num_iterations))


def test_transformer_Multihead(d_model, n_head, batch_size, tgt_len, memory_len, num_iterations):
    multihead = MultiHeadAttention(d_model, n_head).cuda()
    tgt_tensor = torch.rand(batch_size, tgt_len, d_model).cuda()
    memory_tensor = torch.rand(batch_size, memory_len, d_model).cuda()

    multihead = multihead.train()
    # warm up
    for i in range(20):
        output_tensor = multihead(memory_tensor, memory_tensor, memory_tensor)

   # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = multihead(memory_tensor, memory_tensor, memory_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of multihead attention layer: {:.9f} seconds".format((end_time - start_time) / num_iterations))


# test
test_transformer_encoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512, num_iterations=1000)
# test_transformer_decoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, tgt_len=512, memory_len=512, num_iterations=1000)
# test_transformer_Layer_Norm(d_model=512, batch_size=32, memory_len=512, num_iterations=1000)
# test_transformer_Multihead(d_model=512, n_head=8, batch_size=32,  tgt_len=512, memory_len=512, num_iterations=1000)

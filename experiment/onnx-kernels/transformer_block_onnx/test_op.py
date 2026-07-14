import torch
import torch.nn as nn
import time

# import pynvml
# ReLU softmax linear 

from blocks.encoder_layer import EncoderLayer
from blocks.decoder_layer import DecoderLayer
from layers.layer_norm import LayerNorm
from layers.multi_head_attention import MultiHeadAttention

def test_relu(d_model, batch_size, seq_len, num_iterations):
    #pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    relu = nn.ReLU().cuda()

    # transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1).cuda()
    # transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(batch_size, seq_len, d_model).cuda()

    # warm up
    for i in range(200):
        output_tensor = relu(input_tensor)

    # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = relu(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of relu: {:.9f} seconds".format((end_time - start_time) / num_iterations))

def test_softmax(d_model, batch_size, seq_len, num_iterations):
    #pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    softmax = nn.Softmax(dim=-1).cuda()

    # transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1).cuda()
    # transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(batch_size, seq_len, d_model).cuda()

    # warm up
    for i in range(20):
        output_tensor = softmax(input_tensor)

    # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = softmax(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of softmax: {:.9f} seconds".format((end_time - start_time) / num_iterations))


def test_linear(d_model, hidden, batch_size, num_iterations):
    #pynvml.nvmlInit()
    # handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    # powerusage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000
    linear = nn.Linear(hidden, d_model).cuda()

    # transformer_encoder_block = EncoderLayer (d_model, ffn_hidden, n_head, drop_prob = 0.1).cuda()
    # transformer_encoder_block = transformer_encoder_block.train()
    
    input_tensor = torch.rand(d_model, hidden).cuda()

    linear = linear.train()

    # warm up
    for i in range(20):
        output_tensor = linear(input_tensor)

    # test 
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = linear(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    print("Time per iteration of linear: {:.9f} seconds".format((end_time - start_time) / num_iterations))


def test_convolution(in_channels, out_channels, kernel_size, batch_size, input_height, input_width, num_iterations):
    # Initialize the convolutional layer and move it to GPU
    conv = nn.Conv2d(in_channels, out_channels, kernel_size).cuda()
    
    # Generate a random input tensor with the specified batch size, channels, and spatial dimensions
    input_tensor = torch.rand(batch_size, in_channels, input_height, input_width).cuda()
    
    conv = conv.train()
    
    # Warm up
    for i in range(20):
        output_tensor = conv(input_tensor)

    # Measure the time per iteration
    torch.cuda.synchronize()
    start_time = time.time()
    for i in range(num_iterations):
        output_tensor = conv(input_tensor)
    torch.cuda.synchronize()
    end_time = time.time()

    # Print the average time per iteration
    print("Time per iteration of convolution: {:.9f} seconds".format((end_time - start_time) / num_iterations))


# test
# test_relu(d_model=512, batch_size=32, seq_len=512, num_iterations=1000)
# test_softmax(d_model=512, batch_size=32, seq_len=512, num_iterations=1000)
# test_linear(d_model=512, hidden=2048, batch_size=32, num_iterations=1000)
# test_convolution(in_channels=3, out_channels=64, kernel_size=3, batch_size=32, input_height=128, input_width=128, num_iterations=1000)


test_relu(d_model=64, batch_size=1, seq_len=128, num_iterations=100000)
test_softmax(d_model=64, batch_size=1, seq_len=128, num_iterations=100000)
test_linear(d_model=64, hidden=128, batch_size=1, num_iterations=100000)
test_convolution(in_channels=3, out_channels=6, kernel_size=7, batch_size=1, input_height=64, input_width=64, num_iterations=100000)



# test_transformer_encoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, seq_len=512, num_iterations=100)
# test_transformer_decoder_block(d_model=512, n_head=8, ffn_hidden=2048, batch_size=32, tgt_len=512, memory_len=512, num_iterations=100)
# test_transformer_Layer_Norm(d_model=512, batch_size=32,  tgt_len=512, memory_len=512, num_iterations=100)
# test_transformer_Multihead(d_model=512, n_head=8, batch_size=32,  tgt_len=512, memory_len=512, num_iterations=100)

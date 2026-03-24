
"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
Generated on: 2026-03-24 20:51:56

"""
from test_runif import DeviceData, DeviceConfig, DeviceStream, DeviceRuntime
from typing import List
from numpy import ndarray
import numpy as np

async def aux_stream(
    stream: DeviceStream, config: List[DeviceConfig], 
    iptrs: List[DeviceData], idata: List, 
    optrs: List[DeviceData], odata: List, olen: List):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------
    await stream.apply(config)
    await stream.config(config_id=0)
    # ------------------------------
    # 2. Host -> Device transfer
    # ------------------------------
    for i in range(len(iptrs)):
        await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]))
    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    # await stream.execution_finish()
    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])

    await stream.release()
    return

def DeviceData_Pong(ptr : DeviceData) -> DeviceData:
    new_ptr = DeviceData(ptr.address+ptr.size, ptr.size)
    return new_ptr
  
async def aux_stream_pingpong(
    stream: DeviceStream, 
    # config: List[DeviceConfig], 
    config_id:int,
    iptrs: List[DeviceData], idata: List[ndarray], 
    optrs: List[DeviceData], odata: List, olen: List, 
    pingpong: bool):
    """
    Execute a device stream workflow.

    Parameters
    ----------
    stream : DeviceStream
        The device stream instance to operate on.
    config : List[DeviceConfig]
        Configuration objects to apply before execution.
    iptrs : List[DeviceData]
        Device pointers for input buffers.
    idata : List
        Host-side input data corresponding to `iptrs`.
    optrs : List[DeviceData]
        Device pointers for output buffers.
    odata : List
        Host-side output data containers corresponding to `optrs`.
    olen : List[int]
        Expected output lengths for each output buffer.
    pingpong : bool
        Indicates the pingpong phase(ping-phase or pong-phase)
    """
    # ------------------------------
    # 1. Apply stream configuration
    # ------------------------------     
    await stream.config(config_id=config_id)
    
    # ------------------------------
    # 2. Host -> Device transfer
    #   depend_type:
    #   2 -> depends on the second previous task (no need to wait for store-back)
    #   1 -> depends on the immediately previous task (no need to wait for store-back)
    #   0 -> strictly sequential execution
    # ------------------------------
    for i in range(len(iptrs)):
        if(pingpong == 0):
            await stream.memcpyHostToDevice(d_data=iptrs[i], h_data=idata[i], size=len(idata[i]), depend_type=2)
        else:
            await stream.memcpyHostToDevice(DeviceData_Pong(iptrs[i]), h_data=idata[i], size=len(idata[i]), depend_type=2)

    # ------------------------------
    # 3. Execute on device
    # ------------------------------
    await stream.execution_start()
    await stream.execution_finish()

    # ------------------------------
    # 4. Device → Host transfer
    # ------------------------------
    for i in range(len(optrs)):
        if(pingpong == 0):
            await stream.memcpyDeviceToHost(d_data=optrs[i], h_data=odata[i], size=olen[i])
        else :
            await stream.memcpyDeviceToHost(DeviceData_Pong(optrs[i]), h_data=odata[i], size=olen[i])
    
    # await stream.synchronize()
    # await stream.release()
    return

async def aux_stream_pingpong_init(
    stream: DeviceStream, config: List[DeviceConfig]
    ):
    """
    Apply stream configuration
    """
    cfg_copy = list(config)
    await stream.apply(cfg_copy)  
    await stream.config(config_id=0)
    
    # await stream.release()
    return

## ===----------------------------------------------------------------------===//
## Configuration Data 
## ===----------------------------------------------------------------------===//
async def Gemm_0(runtime: DeviceRuntime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray):
    # runtime.log.info("[ADORA] Starting CGRA call (Gemm_0)")
    iptrs, idata = [],[]
    optrs, odata, olen = [],[],[]
    configs, data_ptr = [],[]
    stream = runtime.create_stream()
    ndarray_3 = np.empty((    32,     64), dtype=np.float16)
    ndarray_3[...] = arg_2[...]
    for int_4 in range(0, 128, 8):
        for int_5 in range(0, 32, 32):
            for int_6 in range(0, 64, 64):
                
                ## %0 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
                int_7 = int_4 + 4
                
                ## %2 = ADORA.BlockLoad %arg0 [%arg4, %1] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
                int_8 = int_5 + 1
                
                ## %4 = ADORA.BlockLoad %arg0 [%3, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
                int_9 = int_5 + 1
                int_10 = int_4 + 4
                
                ## %7 = ADORA.BlockLoad %arg0 [%5, %6] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
                int_11 = int_5 + 2
                
                ## %9 = ADORA.BlockLoad %arg0 [%8, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
                int_12 = int_5 + 2
                int_13 = int_4 + 4
                
                ## %12 = ADORA.BlockLoad %arg0 [%10, %11] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
                int_14 = int_5 + 3
                
                ## %14 = ADORA.BlockLoad %arg0 [%13, %arg3] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
                int_15 = int_5 + 3
                int_16 = int_4 + 4
                
                ## %17 = ADORA.BlockLoad %arg0 [%15, %16] : memref<32x128xbf16> -> memref<8x4xbf16> , stride [4, 1] {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
                
                ## %18 = ADORA.BlockLoad %arg1 [%arg3, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
                int_17 = int_4 + 1
                
                ## %20 = ADORA.BlockLoad %arg1 [%19, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
                int_18 = int_4 + 2
                
                ## %22 = ADORA.BlockLoad %arg1 [%21, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
                int_19 = int_4 + 3
                
                ## %24 = ADORA.BlockLoad %arg1 [%23, %arg5] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
                int_20 = int_4 + 4
                
       
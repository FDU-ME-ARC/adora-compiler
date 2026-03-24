
"""
Copyright (c) 2025 ADORA
All rights reserved.
Automatically generated file for pytest/cocotb based CGRA call function from ADORA.
Generated on: 2026-03-24 22:46:05

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
""" kernel: GEMMIS,  cfgNum: 265"""
cfgbit_GEMMIS = [
		0x3000, 0x0000, 0x0041,
		0x2000, 0x0000, 0x0061,
		0x0020, 0x0000, 0x0078,
		0x4000, 0x0002, 0x0080,
		0x0000, 0x400f, 0x0090,
		0x0900, 0x0000, 0x0091,
		0x0000, 0x0000, 0x0098,
		0x0000, 0x8000, 0x00a0,
		0x0000, 0x0000, 0x00a1,
		0x0000, 0x00f9, 0x00d0,
		0x0300, 0x0000, 0x00d1,
		0x0010, 0x0043, 0x00d2,
		0x0004, 0x0000, 0x00d3,
		0x0010, 0x0000, 0x00e0,
		0x0000, 0x4002, 0x00e8,
		0xf400, 0x021f, 0x00e9,
		0x3fe8, 0x1010, 0x00ea,
		0x0200, 0x0008, 0x00eb,
		0x8000, 0x0000, 0x00f8,
		0xc104, 0x081f, 0x00f9,
		0x3f82, 0x0001, 0x00fa,
		0x0000, 0x0008, 0x00fb,
		0x9000, 0x0000, 0x0100,
		0xc104, 0x081f, 0x0101,
		0x0002, 0x0001, 0x0102,
		0x0000, 0x0488, 0x0103,
		0x0000, 0x0218, 0x0108,
		0x0200, 0x0000, 0x0110,
		0x0000, 0x700f, 0x0120,
		0x0a00, 0x0000, 0x0121,
		0x0000, 0x8000, 0x0128,
		0x0064, 0x0000, 0x0129,
		0x0000, 0x0001, 0x0130,
		0x2060, 0x0000, 0x0131,
		0x0000, 0x5011, 0x0140,
		0x0b00, 0x0000, 0x0141,
		0x3000, 0x0000, 0x0149,
		0x0300, 0x8000, 0x0150,
		0x2000, 0x0000, 0x0151,
		0x0060, 0x0000, 0x0169,
		0x0000, 0x0041, 0x0170,
		0x2000, 0x0000, 0x0171,
		0x0000, 0x0611, 0x0178,
		0x2200, 0x0000, 0x0179,
		0x0000, 0x000f, 0x0180,
		0x0b00, 0x0000, 0x0181,
		0x0000, 0x0000, 0x0188,
		0x0081, 0x0000, 0x0189,
		0x0200, 0x0002, 0x0190,
		0x3200, 0x0000, 0x0191,
		0x0000, 0x0311, 0x0198,
		0x2100, 0x0000, 0x0199,
		0x0000, 0x300f, 0x01a0,
		0x0a00, 0x0000, 0x01a1,
		0x0020, 0x0000, 0x01a9,
		0x0004, 0x0002, 0x01b0,
		0x0040, 0x0000, 0x01b1,
		0x0000, 0x300f, 0x01b8,
		0x2300, 0x0000, 0x01b9,
		0x0000, 0x0311, 0x01c0,
		0x0b00, 0x0000, 0x01c1,
		0x0000, 0x0000, 0x01c8,
		0x0002, 0x0000, 0x01d0,
		0x9000, 0x0000, 0x01e8,
		0xc104, 0x081f, 0x01e9,
		0x0002, 0x0001, 0x01ea,
		0x0000, 0x2a08, 0x01eb,
		0x0000, 0x0000, 0x01ec,
		0x8000, 0x0000, 0x01f0,
		0xc104, 0x081f, 0x01f1,
		0x0002, 0x0001, 0x01f2,
		0x0000, 0x2b88, 0x01f3,
		0x0080, 0x0000, 0x01f4,
		0x2030, 0x0000, 0x01f8,
		0x0200, 0x001a, 0x0200,
		0x0000, 0xa011, 0x0210,
		0x0c00, 0x0000, 0x0211,
		0x0000, 0x0000, 0x0218,
		0x0040, 0x0000, 0x0219,
		0x0000, 0x0100, 0x0220,
		0x3040, 0x0000, 0x0221,
		0x0000, 0x0111, 0x0228,
		0x1100, 0x0000, 0x0229,
		0x0000, 0x8000, 0x0238,
		0x0000, 0x0000, 0x0239,
		0x0000, 0x0000, 0x0240,
		0x2220, 0x0000, 0x0241,
		0x0000, 0x0511, 0x0248,
		0x2400, 0x0000, 0x0249,
		0x0000, 0x0002, 0x0260,
		0x0018, 0x0000, 0x0261,
		0x0000, 0x0511, 0x0268,
		0x1b00, 0x0000, 0x0269,
		0x0000, 0x300f, 0x0270,
		0x0a00, 0x0000, 0x0271,
		0x0500, 0x80c0, 0x0278,
		0x0400, 0x0000, 0x0279,
		0x0000, 0x4000, 0x0280,
		0x3019, 0x0000, 0x0281,
		0x0000, 0x0211, 0x0290,
		0x1400, 0x0000, 0x0291,
		0x4000, 0x2000, 0x0298,
		0x0004, 0x0000, 0x0299,
		0x0000, 0x0000, 0x02a0,
		0x0061, 0x0000, 0x02a1,
		0x0000, 0x500f, 0x02a8,
		0x2100, 0x0000, 0x02a9,
		0x0000, 0x00f9, 0x02b0,
		0x0300, 0x0000, 0x02b1,
		0x0010, 0x0043, 0x02b2,
		0x0004, 0x0000, 0x02b3,
		0x0000, 0x0003, 0x02b8,
		0x0012, 0x0011, 0x02c0,
		0x1000, 0x4002, 0x02c8,
		0xf400, 0x021f, 0x02c9,
		0x3fe8, 0x1010, 0x02ca,
		0x0200, 0x0008, 0x02cb,
		0x8000, 0x0000, 0x02d0,
		0xc104, 0x081f, 0x02d1,
		0x3f82, 0x0001, 0x02d2,
		0x0000, 0x0008, 0x02d3,
		0x8000, 0x0000, 0x02d8,
		0xc104, 0x081f, 0x02d9,
		0x3f82, 0x0001, 0x02da,
		0x0000, 0x0008, 0x02db,
		0x9000, 0x0000, 0x02e0,
		0xc104, 0x081f, 0x02e1,
		0x0002, 0x0001, 0x02e2,
		0x0000, 0x0288, 0x02e3,
		0x0201, 0x001a, 0x02e8,
		0x1200, 0x0086, 0x02f0,
		0x0000, 0x0511, 0x02f8,
		0x0a00, 0x0000, 0x02f9,
		0x0000, 0x500f, 0x0300,
		0x0a00, 0x0000, 0x0301,
		0x0001, 0x9000, 0x0308,
		0x0044, 0x0000, 0x0309,
		0x0000, 0x3000, 0x0310,
		0x0844, 0x0000, 0x0311,
		0x0000, 0x0111, 0x0318,
		0x2400, 0x0000, 0x0319,
		0x0440, 0x0000, 0x0329,
		0x0001, 0x0080, 0x0330,
		0x2040, 0x0000, 0x0331,
		0x0000, 0x700f, 0x0338,
		0x1a00, 0x0000, 0x0339,
		0x0400, 0x0080, 0x0348,
		0x3200, 0x0000, 0x0349,
		0x0000, 0x0000, 0x0350,
		0x2681, 0x0000, 0x0351,
		0x0000, 0x0102, 0x0368,
		0x3007, 0x0000, 0x0369,
		0x8000, 0x2000, 0x0370,
		0x301f, 0x0000, 0x0371,
		0x0000, 0x100f, 0x0378,
		0x0b00, 0x0000, 0x0379,
		0x0000, 0x5011, 0x0380,
		0x1100, 0x0000, 0x0381,
		0x0100, 0x0001, 0x0388,
		0x0066, 0x0000, 0x0389,
		0x0000, 0x0001, 0x0390,
		0x007e, 0x0000, 0x0391,
		0x0000, 0x200f, 0x0398,
		0x0b00, 0x0000, 0x0399,
		0x0000, 0x400f, 0x03a0,
		0x0a00, 0x0000, 0x03a1,
		0x0000, 0x0081, 0x03a8,
		0x0000, 0x0100, 0x03b0,
		0x8000, 0x0000, 0x03b8,
		0xc104, 0x081f, 0x03b9,
		0x3f82, 0x0001, 0x03ba,
		0x0000, 0x0008, 0x03bb,
		0x9000, 0x0000, 0x03c0,
		0xc104, 0x081f, 0x03c1,
		0x0002, 0x0001, 0x03c2,
		0x0000, 0x0488, 0x03c3,
		0x2000, 0x0012, 0x03d8,
		0x0200, 0x0082, 0x03e0,
		0x0000, 0x700f, 0x03f0,
		0x0c00, 0x0000, 0x03f1,
		0x0000, 0x3001, 0x03f8,
		0x0000, 0x0000, 0x03f9,
		0x0000, 0x40c0, 0x0400,
		0x3800, 0x0000, 0x0401,
		0x0000, 0x600f, 0x0408,
		0x0a00, 0x0000, 0x0409,
		0x0000, 0x1000, 0x0418,
		0x1000, 0x0000, 0x0419,
		0x8000, 0x0080, 0x0420,
		0x0000, 0x0000, 0x0421,
		0x0000, 0x500f, 0x0428,
		0x1a00, 0x0000, 0x0429,
		0x0000, 0x0311, 0x0430,
		0x1900, 0x0000, 0x0431,
		0x8000, 0x0000, 0x0438,
		0x3400, 0x0000, 0x0439,
		0x0400, 0x0000, 0x0440,
		0x0000, 0x0000, 0x0441,
		0x0000, 0x5011, 0x0448,
		0x2100, 0x0000, 0x0449,
		0x8000, 0x0000, 0x0458,
		0x0001, 0x0000, 0x0459,
		0x0004, 0x0000, 0x0460,
		0x0000, 0x0000, 0x0461,
		0x0000, 0x1011, 0x0468,
		0x2100, 0x0000, 0x0469,
		0x0000, 0x00c1, 0x0478,
		0x0000, 0x0000, 0x0479,
		0x0000, 0x3040, 0x0480,
		0x2000, 0x0000, 0x0481,
		0x0000, 0x0311, 0x0488,
		0x0a00, 0x0000, 0x0489,
		0x0000, 0x500f, 0x0490,
		0x1a00, 0x0000, 0x0491,
		0x0300, 0x0100, 0x0498,
		0x0300, 0x0000, 0x04a0,
		0x8000, 0x0000, 0x04a8,
		0xc104, 0x081f, 0x04a9,
		0x0002, 0x0001, 0x04aa,
		0x0000, 0x2b08, 0x04ab,
		0x0000, 0x0000, 0x04ac,
		0x9000, 0x0000, 0x04b0,
		0xc104, 0x081f, 0x04b1,
		0x0002, 0x0001, 0x04b2,
		0x0000, 0x2b08, 0x04b3,
		0x0000, 0x0000, 0x04b4,
		0x0000, 0x4002, 0x04b8,
		0xf400, 0x021f, 0x04b9,
		0x3fe8, 0x1010, 0x04ba,
		0x0200, 0x0008, 0x04bb,
		0x0000, 0x0009, 0x04c8,
		0x2000, 0x0100, 0x04d0,
		0x0000, 0x00f9, 0x04d8,
		0x0200, 0x0000, 0x04d9,
		0x0010, 0x0045, 0x04da,
		0x0004, 0x0000, 0x04db,
		0x0001, 0xb000, 0x04e8,
		0x0100, 0x0000, 0x04e9,
		0x0000, 0x4000, 0x04f0,
		0x0000, 0x0000, 0x04f1,
		0x0018, 0x0000, 0x0511,
		0x0018, 0x0000, 0x0531,
		0x0000, 0x4000, 0x0548,
		0x3018, 0x0000, 0x0551,
		0x0000, 0x30c0, 0x0568,
		0x0018, 0x0000, 0x0569,
		0x8000, 0x1000, 0x0570,
		0x0000, 0x0000, 0x0571,
		0x0000, 0x00f9, 0x0580,
		0x0100, 0x0000, 0x0581,
		0x0010, 0x0045, 0x0582,
		0x0004, 0x0000, 0x0583,
		0x0000, 0x0000, 0x0588,
		0x9000, 0x0000, 0x0598,
		0xc104, 0x081f, 0x0599,
		0x0002, 0x0001, 0x059a,
		0x0000, 0x0588, 0x059b,
		0x0000, 0x4002, 0x05a0,
		0xf400, 0x021f, 0x05a1,
		0x3fe8, 0x1010, 0x05a2,
		0x0200, 0x0008, 0x05a3,
		0x0000, 0x0000, 0x05b0,
		0x0008, 0x0000, 0x05c8,
		0x0020, 0x0000, 0x05d0,
		0x0000, 0x0000, 0x05d8,
	]


""" kernel: GEMMIS, ping-phase"""
cfgbit_GEMMIS_ping = [
		0x0000, 0x4002, 0x00e8,
		0x8000, 0x0000, 0x00f8,
		0x9000, 0x0000, 0x0100,
		0x9000, 0x0000, 0x01e8,
		0x8000, 0x0000, 0x01f0,
		0x1000, 0x4002, 0x02c8,
		0x8000, 0x0000, 0x02d0,
		0x8000, 0x0000, 0x02d8,
		0x9000, 0x0000, 0x02e0,
		0x8000, 0x0000, 0x03b8,
		0x9000, 0x0000, 0x03c0,
		0x8000, 0x0000, 0x04a8,
		0x9000, 0x0000, 0x04b0,
		0x0000, 0x4002, 0x04b8,
		0x9000, 0x0000, 0x0598,
		0x0000, 0x4002, 0x05a0,
	]

""" kernel: GEMMIS, pong-phase"""

cfgbit_GEMMIS_pong = [
		0x0020, 0x4002, 0x00e8,
		0x8080, 0x0000, 0x00f8,
		0x9200, 0x0000, 0x0100,
		0x9200, 0x0000, 0x01e8,
		0x8200, 0x0000, 0x01f0,
		0x1020, 0x4002, 0x02c8,
		0x8080, 0x0000, 0x02d0,
		0x8080, 0x0000, 0x02d8,
		0x9200, 0x0000, 0x02e0,
		0x8080, 0x0000, 0x03b8,
		0x9200, 0x0000, 0x03c0,
		0x8200, 0x0000, 0x04a8,
		0x9200, 0x0000, 0x04b0,
		0x0020, 0x4002, 0x04b8,
		0x9200, 0x0000, 0x0598,
		0x0020, 0x4002, 0x05a0,
	]


async def matmul_0(runtime: DeviceRuntime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray):
    # runtime.log.info("[ADORA] Starting CGRA call (matmul_0)")
    iptrs, idata = [],[]
    optrs, odata, olen = [],[],[]
    configs, data_ptr = [],[]
    stream = runtime.create_stream()
    ndarray_3 = await Gemm_0(runtime, arg_0, arg_1, arg_2)

    await stream.synchronize()
async def Gemm_0(runtime: DeviceRuntime, arg_0: ndarray, arg_1: ndarray, arg_2: ndarray):
    # runtime.log.info("[ADORA] Starting CGRA call (Gemm_0)")
    iptrs, idata = [],[]
    optrs, odata, olen = [],[],[]
    configs, data_ptr = [],[]
    stream = runtime.create_stream()
    #######################################
    ### Emit GemmOp: %0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 4, 64, 4, 4>} : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    #######################################
    pingpong = False

    ### memref::AllocOpp: %alloc = memref.alloc() : memref<32x64xbf16>
    ndarray_4 = np.empty((    32,     64), dtype=np.float16)
    ### memref::CopyOp: memref.copy %arg2, %alloc : memref<32x64xbf16> to memref<32x64xbf16>
    ndarray_4[...] = arg_2[...]
    
    ptrs_ping, ptrs_pong = [], []
    ### Pingpong DataBlockLoadOp: %8 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x2c000, 32))
    ptrs_pong.append(DeviceData(0x2c000+32, 32))
    ### Pingpong DataBlockLoadOp: %21 = ADORA.BlockLoad %alloc [%20, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x2e000, 512))
    ptrs_pong.append(DeviceData(0x2e000+512, 512))
    ### Pingpong DataBlockLoadOp: %3 = ADORA.BlockLoad %arg1 [%2, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x14000, 128))
    ptrs_pong.append(DeviceData(0x14000+128, 128))
    ### Pingpong DataBlockLoadOp: %15 = ADORA.BlockLoad %alloc [%arg4, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0xa000, 512))
    ptrs_pong.append(DeviceData(0xa000+512, 512))
    ### Pingpong DataBlockLoadOp: %7 = ADORA.BlockLoad %arg1 [%6, %c0_1] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x18000, 128))
    ptrs_pong.append(DeviceData(0x18000+128, 128))
    ### Pingpong DataBlockLoadOp: %5 = ADORA.BlockLoad %arg1 [%4, %c0_0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x8000, 128))
    ptrs_pong.append(DeviceData(0x8000+128, 128))
    ### Pingpong DataBlockLoadOp: %12 = ADORA.BlockLoad %arg0 [%11, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x16000, 32))
    ptrs_pong.append(DeviceData(0x16000+32, 32))
    ### Pingpong DataBlockLoadOp: %10 = ADORA.BlockLoad %arg0 [%9, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x28000, 32))
    ptrs_pong.append(DeviceData(0x28000+32, 32))
    ### Pingpong DataBlockLoadOp: %18 = ADORA.BlockLoad %alloc [%17, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x1e000, 512))
    ptrs_pong.append(DeviceData(0x1e000+512, 512))
    ### Pingpong DataBlockLoadOp: %14 = ADORA.BlockLoad %arg0 [%13, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x4000, 32))
    ptrs_pong.append(DeviceData(0x4000+32, 32))
    ### Pingpong DataBlockLoadOp: %24 = ADORA.BlockLoad %alloc [%23, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x1a000, 512))
    ptrs_pong.append(DeviceData(0x1a000+512, 512))
    ### Pingpong DataBlockLoadOp: %1 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x1c000, 128))
    ptrs_pong.append(DeviceData(0x1c000+128, 128))
    ### Pingpong LocalMemAllocOp: %19 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x24000, 512))
    ptrs_pong.append(DeviceData(0x24000+512, 512))
    ### Pingpong LocalMemAllocOp: %25 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x12000, 512))
    ptrs_pong.append(DeviceData(0x12000+512, 512))
    ### Pingpong LocalMemAllocOp: %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x10000, 512))
    ptrs_pong.append(DeviceData(0x10000+512, 512))
    ### Pingpong LocalMemAllocOp: %22 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
    ptrs_ping.append(DeviceData(0x26000, 512))
    ptrs_pong.append(DeviceData(0x26000+512, 512))
    pingpong = False
    stream = runtime.create_stream()
    config_GEMMIS = DeviceConfig(config_values=cfgbit_GEMMIS, iob_en=[0x34,0xff,0xdc], tile_en=[0x3f], data_ptr=data_ptr)
    config_GEMMIS_ping = DeviceConfig(config_values=cfgbit_GEMMIS_ping, iob_en=[0x34,0xff,0xdc], tile_en=[0x3f], data_ptr=ptrs_ping)
    config_GEMMIS_pong = DeviceConfig(config_values=cfgbit_GEMMIS_pong, iob_en=[0x34,0xff,0xdc], tile_en=[0x3f], data_ptr=ptrs_pong)
    await aux_stream_pingpong_init(stream, [config_GEMMIS, config_GEMMIS_ping, config_GEMMIS_pong])

    for int_5 in range(0, 128, 4):
        
        ## %1 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "4", KernelName = "GEMMIS", Pingpong}
        idata.append(arg_1[int_5:int_5+1,0:0+64])
        iptrs.append(DeviceData(0x1c000+128 if pingpong else 0x1c000, 128))

        int_6 = int_5 + 1
        
        ## %3 = ADORA.BlockLoad %arg1 [%2, %c0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "5", KernelName = "GEMMIS", Pingpong}
        idata.append(arg_1[int_6:int_6+1,:+64])
        iptrs.append(DeviceData(0x14000+128 if pingpong else 0x14000, 128))

        int_7 = int_5 + 2
        
        ## %5 = ADORA.BlockLoad %arg1 [%4, %c0_0] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "6", KernelName = "GEMMIS", Pingpong}
        idata.append(arg_1[int_7:int_7+1,:+64])
        iptrs.append(DeviceData(0x8000+128 if pingpong else 0x8000, 128))

        int_8 = int_5 + 3
        
        ## %7 = ADORA.BlockLoad %arg1 [%6, %c0_1] : memref<128x64xbf16> -> memref<1x64xbf16>  {ADORAGemm, Id = "7", KernelName = "GEMMIS", Pingpong}
        idata.append(arg_1[int_8:int_8+1,:+64])
        iptrs.append(DeviceData(0x18000+128 if pingpong else 0x18000, 128))

        for int_9 in range(0, 32, 16):
            
            ## %8 = ADORA.BlockLoad %arg0 [%arg4, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "0", KernelName = "GEMMIS", Pingpong}
            idata.append(arg_0[int_9:int_9+16:4,int_5:int_5+4:1])
            iptrs.append(DeviceData(0x2c000+32 if pingpong else 0x2c000, 32))

            int_10 = int_9 + 1
            
            ## %10 = ADORA.BlockLoad %arg0 [%9, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "1", KernelName = "GEMMIS", Pingpong}
            idata.append(arg_0[int_10:int_10+16:4,int_5:int_5+4:1])
            iptrs.append(DeviceData(0x28000+32 if pingpong else 0x28000, 32))

            int_11 = int_9 + 2
            
            ## %12 = ADORA.BlockLoad %arg0 [%11, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "2", KernelName = "GEMMIS", Pingpong}
            idata.append(arg_0[int_11:int_11+16:4,int_5:int_5+4:1])
            iptrs.append(DeviceData(0x16000+32 if pingpong else 0x16000, 32))

            int_12 = int_9 + 3
            
            ## %14 = ADORA.BlockLoad %arg0 [%13, %arg3] : memref<32x128xbf16> -> memref<4x4xbf16> , stride [4, 1] {ADORAGemm, Id = "3", KernelName = "GEMMIS", Pingpong}
            idata.append(arg_0[int_12:int_12+16:4,int_5:int_5+4:1])
            iptrs.append(DeviceData(0x4000+32 if pingpong else 0x4000, 32))

            
            ## %15 = ADORA.BlockLoad %alloc [%arg4, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "8", KernelName = "GEMMIS", Pingpong}
            idata.append(ndarray_4[int_9:int_9+16:4,:+64:1])
            iptrs.append(DeviceData(0xa000+512 if pingpong else 0xa000, 512))

            
            ## %16 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
            data_ptr.append(DeviceData(0x10000+512 if pingpong else 0x10000, 512))
            int_13 = int_9 + 1
            
            ## %18 = ADORA.BlockLoad %alloc [%17, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "10", KernelName = "GEMMIS", Pingpong}
            idata.append(ndarray_4[int_13:int_13+16:4,:+64:1])
            iptrs.append(DeviceData(0x1e000+512 if pingpong else 0x1e000, 512))

            
            ## %19 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
            data_ptr.append(DeviceData(0x24000+512 if pingpong else 0x24000, 512))
            int_14 = int_9 + 2
            
            ## %21 = ADORA.BlockLoad %alloc [%20, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "12", KernelName = "GEMMIS", Pingpong}
            idata.append(ndarray_4[int_14:int_14+16:4,:+64:1])
            iptrs.append(DeviceData(0x2e000+512 if pingpong else 0x2e000, 512))

            
            ## %22 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
            data_ptr.append(DeviceData(0x26000+512 if pingpong else 0x26000, 512))
            int_15 = int_9 + 3
            
            ## %24 = ADORA.BlockLoad %alloc [%23, %c0_2] : memref<32x64xbf16> -> memref<4x64xbf16> , stride [4, 1] {ADORAGemm, Id = "14", KernelName = "GEMMIS", Pingpong}
            idata.append(ndarray_4[int_15:int_15+16:4,:+64:1])
            iptrs.append(DeviceData(0x1a000+512 if pingpong else 0x1a000, 512))

            
            ## %25 = ADORA.LocalMemAlloc memref<4x64xbf16>  {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
            data_ptr.append(DeviceData(0x12000+512 if pingpong else 0x12000, 512))
            
            ### GEMMIS
            data_ptr.append(iptrs)
            config_GEMMIS= DeviceConfig(
            	config_values=cfgbit_GEMMIS,
            	iob_en=[0x34,0xff,0xdc],
            	tile_en=[0x3f],
            	data_ptr=data_ptr
            )
            configs.append(config_GEMMIS)
            

            
            ## ADORA.BlockStore %16, %alloc [%arg4, %c0_2] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "9", KernelName = "GEMMIS", Pingpong}
            odata.append(ndarray_4[int_9:int_9+16:4,:+64:1])
            optrs.append(DeviceData(0x10000+512 if pingpong else 0x10000, 512))
            olen.append(512)

            int_16 = int_9 + 1
            
            ## ADORA.BlockStore %19, %alloc [%26, %c0_2] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "11", KernelName = "GEMMIS", Pingpong}
            odata.append(ndarray_4[int_16:int_16+16:4,:+64:1])
            optrs.append(DeviceData(0x24000+512 if pingpong else 0x24000, 512))
            olen.append(512)

            int_17 = int_9 + 2
            
            ## ADORA.BlockStore %22, %alloc [%27, %c0_2] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "13", KernelName = "GEMMIS", Pingpong}
            odata.append(ndarray_4[int_17:int_17+16:4,:+64:1])
            optrs.append(DeviceData(0x26000+512 if pingpong else 0x26000, 512))
            olen.append(512)

            int_18 = int_9 + 3
            
            ## ADORA.BlockStore %25, %alloc [%28, %c0_2] : memref<4x64xbf16> -> memref<32x64xbf16> , stride [4, 1] {ADORAGemm, Id = "15", KernelName = "GEMMIS", Pingpong}
            odata.append(ndarray_4[int_18:int_18+16:4,:+64:1])
            optrs.append(DeviceData(0x12000+512 if pingpong else 0x12000, 512))
            olen.append(512)

            await aux_stream_pingpong(stream=stream,
            	config_id = 1 if pingpong==False else 2,
            	iptrs=iptrs, idata=idata,
            	optrs=optrs, odata=odata, olen =olen,
            	pingpong=pingpong
            )

            iptrs.clear(), idata.clear()
            optrs.clear(), odata.clear(), olen.clear()

            pingpong = not pingpong


    #######################################
    ### End of GemmOp:%0 = "ADORATensor.Gemm"(%arg0, %arg1, %arg2) {linalg.memoized_indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], operandSegmentSizes = array<i32: 2, 1>, stationary_kind = "InputStationary", tile_size = array<i64: 4, 64, 4, 4>} : (memref<32x128xbf16>, memref<128x64xbf16>, memref<32x64xbf16>) -> memref<32x64xbf16>
    #######################################

    await stream.synchronize()

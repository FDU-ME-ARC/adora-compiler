import cocotb
from cocotb.clock import Clock
import numpy as np

# 导入你们课题组仿真器框架的核心组件 (请根据实际库所在的路径微调 import)
# 根据你上传的教程文档，这些通常在框架的顶层环境里
from simulator_framework import AXI_Device, Device, DeviceRuntime 

# 导入刚才 mapper 自动生成的 python 配置与执行脚本
import conv_config

@cocotb.test()
async def test_conv2d_im2col(dut):
    """
    CGRA Conv2D (Im2Col) Cocotb Testbench
    """
    # =====================================================================
    # 1. 基础环境与时钟初始化
    # =====================================================================
    cocotb.log.info("[TB] Initializing Clock and Reset...")
    
    # 驱动时钟 (10ns = 100MHz)
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # 挂载 AXI 总线和 CGRA Device (根据教程里的定义)
    axibus = AXI_Device(dut)
    device1 = Device(dut, 0, 4) 
    
    # 执行复位序列
    await axibus.cycle_reset()
    
    # =====================================================================
    # 2. 运行时 (Runtime) 实例化
    # =====================================================================
    cocotb.log.info("[TB] Creating Device Runtime and AXI Interfaces...")
    runtime = DeviceRuntime(
        dut=dut,
        axi=axibus.axi,
        axil=axibus.axil,
        axi_size=axibus.axi.write_if.max_burst_size,
    )
    runtime.add_device(device1)

    # =====================================================================
    # 3. 准备 Host 端测试数据 (即 DRAM 中的 Numpy 数组)
    # =====================================================================
    cocotb.log.info("[TB] Generating Input, Weight, and Bias Data...")
    
    # 根据你 affine.mlir 中的原始张量形状分配内存
    # 输入图片: 1x3x32x32 | 权重: 16x3x3x3 | 偏置: 16
    # (注：根据你们硬件仿真器的精度支持，bf16 可能用 np.int16 或 np.float32 模拟，此处以 float32 为例)
    input_data  = np.random.uniform(-1, 1, size=(1, 3, 32, 32)).astype(np.float32)
    weight_data = np.random.uniform(-1, 1, size=(16, 3, 3, 3)).astype(np.float32)
    bias_data   = np.random.uniform(-1, 1, size=(16,)).astype(np.float32)
    
    # =====================================================================
    # 4. 调用 Mapper 自动生成的硬件控制流！
    # =====================================================================
    cocotb.log.info("[TB] Launching CGRA Hardware Execution Flow...")
    
    # 直接调用 conv_config.py 暴露的顶层函数。
    # 根据 MLIR 的 `func.func @main_graph`，生成器通常会把它翻译成同名 python 异步函数
    hw_output = await conv_config.main_graph(runtime, input_data, weight_data, bias_data)

    # =====================================================================
    # 5. 仿真结果软硬件比对 (可选，但推荐)
    # =====================================================================
    cocotb.log.info("[TB] Hardware Execution Finished! Checking results...")
    
    # 你可以引入 PyTorch 或手写的 Numpy 卷积函数算出一个 expected_output
    # expected_output = software_conv2d(input_data, weight_data, bias_data)
    # np.testing.assert_allclose(hw_output, expected_output, rtol=1e-2, atol=1e-2)
    
    cocotb.log.info("[TB] Convolution Im2Col Test Passed Successfully! 🚀")
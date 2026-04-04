
//===----------------------------------------------------------------------===//
//
// Automatically generated file for CGRA call function in cgrv-opt.
//
//===----------------------------------------------------------------------===//

#include "include/ISA.h"

static uint8_t _task_id = 0;

#define LD_DEP_ST_LAST_TASK 1     // this load command depends on the store command of last task
#define LD_DEP_EX_LAST_TASK 2     // this load command depends on the execute command of last task
#define LD_DEP_ST_LAST_SEC_TASK 3 // this load command depends on the store command of last second task
#define EX_DEP_ST_LAST_TASK 1     // this execute command depends on the store command of last task



//===----------------------------------------------------------------------===//
// Configuration Data 
//===----------------------------------------------------------------------===//
/// kernel_gemm_0
volatile unsigned short cin_kernel_gemm_0[13][3] __attribute__((aligned(8))) = {
		{0x2000, 0x6400, 0x0018},
		{0x0000, 0x0000, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x0002, 0x001b},
		{0x2800, 0x6400, 0x0020},
		{0x0000, 0x0000, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x08c2, 0x0023},
		{0x0020, 0x0000, 0x0024},
		{0x0000, 0x0000, 0x0068},
		{0x0001, 0x0000, 0x0070},
		{0x999a, 0x3f99, 0x00b0},
		{0x000a, 0x0100, 0x00b1},
	};


/// kernel_gemm_1
volatile unsigned short cin_kernel_gemm_1[38][3] __attribute__((aligned(8))) = {
		{0x0800, 0x7800, 0x0008},
		{0x0040, 0x00c8, 0x0009},
		{0x0000, 0x0000, 0x000a},
		{0x0000, 0x0002, 0x000b},
		{0x2000, 0x7800, 0x0030},
		{0xf8c0, 0x00cf, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x0002, 0x0033},
		{0x0000, 0x0000, 0x0058},
		{0x0000, 0x0000, 0x0078},
		{0x0000, 0x3fc0, 0x00b8},
		{0x000a, 0x0200, 0x00b9},
		{0x0060, 0x0000, 0x00e9},
		{0x0020, 0x0000, 0x0101},
		{0x000c, 0x0000, 0x0179},
		{0x0008, 0x0000, 0x0181},
		{0x0000, 0x0001, 0x0188},
		{0x0060, 0x0000, 0x0191},
		{0x600c, 0x0c00, 0x01d1},
		{0x0020, 0x0000, 0x0219},
		{0x0000, 0x0080, 0x0220},
		{0x400a, 0x1a00, 0x0261},
		{0x002c, 0x0300, 0x0269},
		{0x0000, 0x0000, 0x026a},
		{0x0010, 0x078a, 0x026b},
		{0x0064, 0x0000, 0x026c},
		{0x0000, 0x0010, 0x02a0},
		{0x2000, 0x0000, 0x02a8},
		{0x0000, 0x0000, 0x02b0},
		{0x2000, 0x7803, 0x02e0},
		{0x4b00, 0x00cf, 0x02e1},
		{0x0000, 0x0000, 0x02e2},
		{0x0000, 0x0002, 0x02e3},
		{0x0800, 0x7800, 0x02e8},
		{0x0040, 0x00c8, 0x02e9},
		{0x0000, 0x0000, 0x02ea},
		{0x0000, 0x0982, 0x02eb},
		{0x0020, 0x0000, 0x02ec},
	};


void gemm(void* arg_0 ,void* arg_1 ,void* arg_2){
  for (int int_3 = 0; int_3 < 20; int_3 = int_3 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// kernel_gemm_0
    load_cfg((void*)cin_kernel_gemm_0, 0x20000, 78, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 13, _task_id, 0);
    execute(0xc, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "kernel_gemm_0"}
    uint64_t dramoffset_1 = 100 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 100, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 100;
    
    }
    _task_id++;
    {
    /// %2 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "2", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_2 = 120 * int_3;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_1 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 120, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_2 = spadoffset_2 + 120;
    
    }
    {
    /// %3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "3", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_3 = 0;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    load_data(arg_2 + dramoffset_3 + roffset_3, 0x10000 + spadoffset_3, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_3 = spadoffset_3 + 3000;
    
    }
    {
    /// kernel_gemm_1
    load_cfg((void*)cin_kernel_gemm_1, 0x20000, 228, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 38, _task_id, 0);
    execute(0x621, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %4, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "4", KernelName = "kernel_gemm_1"}
    uint64_t dramoffset_4 = 100 * int_3;
    uint64_t spadoffset_4 = 0;
    uint64_t roffset_4 = 0;
    store(arg_0 + dramoffset_4 + roffset_4, 0x12000 + spadoffset_4, 100, _task_id, 0);
    spadoffset_4 = spadoffset_4 + 100;
    
    }
    _task_id++;
  }
  
  
  
  fence(1);
}


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
/// gemm_1
volatile unsigned short cin_gemm_1[32][3] __attribute__((aligned(8))) = {
		{0x0000, 0x7800, 0x0018},
		{0x0040, 0x00c8, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x0962, 0x001b},
		{0x0020, 0x0000, 0x001c},
		{0x0800, 0x7800, 0x0020},
		{0x0040, 0x00c8, 0x0021},
		{0x0000, 0x0000, 0x0022},
		{0x0000, 0x0002, 0x0023},
		{0x2000, 0x7800, 0x0028},
		{0xf8c0, 0x00cf, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x2800, 0x7803, 0x0038},
		{0x4b00, 0x00cf, 0x0039},
		{0x0000, 0x0000, 0x003a},
		{0x0000, 0x0002, 0x003b},
		{0x0000, 0x0000, 0x0068},
		{0x0000, 0x0000, 0x0070},
		{0x0000, 0x0000, 0x0088},
		{0x700c, 0x0c00, 0x00b1},
		{0x0000, 0x3fc0, 0x00b8},
		{0x000a, 0x0800, 0x00b9},
		{0x200a, 0x2300, 0x00c1},
		{0x0000, 0x0000, 0x0100},
		{0x0000, 0x0000, 0x0108},
		{0x0004, 0x0000, 0x0110},
		{0x0000, 0x2000, 0x0118},
		{0x002c, 0x0200, 0x0149},
		{0x0000, 0x0000, 0x014a},
		{0x0010, 0x0789, 0x014b},
		{0x0064, 0x0000, 0x014c},
	};


/// gemm_opt_0
volatile unsigned short cin_gemm_opt_0[13][3] __attribute__((aligned(8))) = {
		{0x999a, 0x3f99, 0x0268},
		{0x000a, 0x0300, 0x0269},
		{0x0010, 0x0000, 0x02b0},
		{0x0000, 0x0000, 0x02b8},
		{0x2000, 0x6400, 0x02f0},
		{0x0040, 0x00a0, 0x02f1},
		{0x0000, 0x0000, 0x02f2},
		{0x0000, 0x0002, 0x02f3},
		{0x2000, 0x6400, 0x0300},
		{0x0040, 0x00a0, 0x0301},
		{0x0000, 0x0000, 0x0302},
		{0x0000, 0x08c2, 0x0303},
		{0x0000, 0x0000, 0x0304},
	};


/// gemm_0
volatile unsigned short cin_gemm_0[13][3] __attribute__((aligned(8))) = {
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


/// gemm_opt_1
volatile unsigned short cin_gemm_opt_1[35][3] __attribute__((aligned(8))) = {
		{0x002c, 0x0400, 0x01c1},
		{0x0000, 0x0000, 0x01c2},
		{0x0010, 0x0789, 0x01c3},
		{0x07d0, 0x0000, 0x01c4},
		{0x0004, 0x0000, 0x0209},
		{0x4000, 0x0000, 0x0210},
		{0x0000, 0x0000, 0x0211},
		{0x0008, 0x0000, 0x0219},
		{0x4000, 0x0000, 0x0220},
		{0x0000, 0x3fc0, 0x0248},
		{0x000a, 0x2000, 0x0249},
		{0x300a, 0x2100, 0x0259},
		{0x800c, 0x2100, 0x0269},
		{0x0000, 0x0000, 0x0298},
		{0x0002, 0x0000, 0x02a8},
		{0x0000, 0x0001, 0x02b0},
		{0x0002, 0x0000, 0x02b8},
		{0x0000, 0x0001, 0x02c0},
		{0x2000, 0x7800, 0x02d8},
		{0xf8c0, 0x80cf, 0x02d9},
		{0x4000, 0x0001, 0x02da},
		{0x0000, 0x0002, 0x02db},
		{0x0800, 0x7800, 0x02f0},
		{0x0040, 0x80c8, 0x02f1},
		{0x4000, 0x0001, 0x02f2},
		{0x0000, 0x0982, 0x02f3},
		{0x0020, 0x0000, 0x02f4},
		{0x2800, 0x7803, 0x02f8},
		{0x4b00, 0x80cf, 0x02f9},
		{0x4e89, 0x0001, 0x02fa},
		{0x0000, 0x0002, 0x02fb},
		{0x0000, 0x7800, 0x0308},
		{0x0040, 0x80c8, 0x0309},
		{0x4000, 0x0001, 0x030a},
		{0x0000, 0x0002, 0x030b},
	};


void gemm(void* arg_0 ,void* arg_1 ,void* arg_2){
  for (int int_3 = 0; int_3 < 20; int_3 = int_3 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// gemm_0
    load_cfg((void*)cin_gemm_0, 0x20000, 78, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 13, _task_id, 0);
    execute(0xc, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
    uint64_t dramoffset_1 = 100 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 100, _task_id, 0);
    spadoffset_1 = spadoffset_1 + 100;
    
    }
    _task_id++;
    {
    /// %2 = ADORA.BlockLoad %arg1 [%arg3, 0] : memref<?x30xf32> -> memref<1x30xf32>  {Id = "2", KernelName = "gemm_1"}
    uint64_t dramoffset_2 = 120 * int_3;
    uint64_t spadoffset_2 = 0;
    uint64_t roffset_2 = 0;
    load_data(arg_1 + dramoffset_2 + roffset_2, 0x8000 + spadoffset_2, 120, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_2 = spadoffset_2 + 120;
    
    }
    {
    /// %3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "3", KernelName = "gemm_1"}
    uint64_t dramoffset_3 = 0;
    uint64_t spadoffset_3 = 0;
    uint64_t roffset_3 = 0;
    load_data(arg_2 + dramoffset_3 + roffset_3, 0xa000 + spadoffset_3, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_3 = spadoffset_3 + 3000;
    
    }
    {
    /// gemm_1
    load_cfg((void*)cin_gemm_1, 0x20000, 192, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 32, _task_id, 0);
    execute(0x5c, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %4, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "4", KernelName = "gemm_1"}
    uint64_t dramoffset_4 = 100 * int_3;
    uint64_t spadoffset_4 = 0;
    uint64_t roffset_4 = 0;
    store(arg_0 + dramoffset_4 + roffset_4, 0x0 + spadoffset_4, 100, _task_id, 0);
    spadoffset_4 = spadoffset_4 + 100;
    
    }
    _task_id++;
  }
  
  fence(1);
}
void gemm_opt(void* arg_0 ,void* arg_1 ,void* arg_2){
  {
  /// %0 = ADORA.BlockLoad %arg0 [0, 0] : memref<?x25xf32> -> memref<20x25xf32>  {Id = "0", KernelName = "gemm_opt_0"}
  uint64_t dramoffset_0 = 0;
  uint64_t spadoffset_0 = 0;
  uint64_t roffset_0 = 0;
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x10000 + spadoffset_0, 2000, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_0 = spadoffset_0 + 2000;
  
  }
  {
  /// gemm_opt_0
  load_cfg((void*)cin_gemm_opt_0, 0x20000, 78, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 13, _task_id, 0);
  execute(0x2800, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %1, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  store(arg_0 + dramoffset_1 + roffset_1, 0x18000 + spadoffset_1, 2000, _task_id, 0);
  spadoffset_1 = spadoffset_1 + 2000;
  
  }
  _task_id++;
  {
  /// %2 = ADORA.BlockLoad %arg1 [0, 0] : memref<?x30xf32> -> memref<20x30xf32>  {Id = "2", KernelName = "gemm_opt_1"}
  uint64_t dramoffset_2 = 0;
  uint64_t spadoffset_2 = 0;
  uint64_t roffset_2 = 0;
  load_data(arg_1 + dramoffset_2 + roffset_2, 0x10000 + spadoffset_2, 2400, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_2 = spadoffset_2 + 2400;
  
  }
  {
  /// %3 = ADORA.BlockLoad %arg2 [0, 0] : memref<?x25xf32> -> memref<30x25xf32>  {Id = "3", KernelName = "gemm_opt_1"}
  uint64_t dramoffset_3 = 0;
  uint64_t spadoffset_3 = 0;
  uint64_t roffset_3 = 0;
  load_data(arg_2 + dramoffset_3 + roffset_3, 0x1a000 + spadoffset_3, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_3 = spadoffset_3 + 3000;
  
  }
  {
  /// gemm_opt_1
  load_cfg((void*)cin_gemm_opt_1, 0x20000, 210, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 35, _task_id, 0);
  execute(0x5900, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %4, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "4", KernelName = "gemm_opt_1"}
  uint64_t dramoffset_4 = 0;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  store(arg_0 + dramoffset_4 + roffset_4, 0x12000 + spadoffset_4, 2000, _task_id, 0);
  spadoffset_4 = spadoffset_4 + 2000;
  
  }
  _task_id++;
  fence(1);
}

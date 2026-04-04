
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
/// gemm_opt_0
volatile unsigned short cin_gemm_opt_0[12][3] __attribute__((aligned(8))) = {
		{0x2800, 0x6400, 0x0010},
		{0x0040, 0x00a0, 0x0011},
		{0x0000, 0x0000, 0x0012},
		{0x0000, 0x08c2, 0x0013},
		{0x0020, 0x0000, 0x0014},
		{0x2000, 0x6400, 0x0018},
		{0x0040, 0x00a0, 0x0019},
		{0x0000, 0x0000, 0x001a},
		{0x0000, 0x0002, 0x001b},
		{0x0000, 0x0000, 0x0060},
		{0x999a, 0x3f99, 0x00a8},
		{0x000a, 0x0800, 0x00a9},
	};


/// gemm_1
volatile unsigned short cin_gemm_1[40][3] __attribute__((aligned(8))) = {
		{0x2000, 0x7800, 0x0028},
		{0xf8c0, 0x00cf, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x0800, 0x7800, 0x0030},
		{0x0040, 0x00c8, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x0002, 0x0033},
		{0x1000, 0x7800, 0x0040},
		{0x0040, 0x00c8, 0x0041},
		{0x0000, 0x0000, 0x0042},
		{0x0000, 0x09a2, 0x0043},
		{0x0000, 0x0000, 0x0044},
		{0x0000, 0x0000, 0x0070},
		{0x0000, 0x0000, 0x0080},
		{0x0010, 0x0008, 0x0088},
		{0x0004, 0x0000, 0x0090},
		{0x0000, 0x3fc0, 0x00b8},
		{0x000a, 0x0800, 0x00b9},
		{0x700c, 0x2300, 0x00c9},
		{0x0080, 0x0000, 0x0109},
		{0x4300, 0x1000, 0x0110},
		{0x0004, 0x0000, 0x0118},
		{0x0000, 0x0000, 0x0120},
		{0x200a, 0x2100, 0x0159},
		{0x000c, 0x0000, 0x0199},
		{0x4000, 0x0000, 0x01a0},
		{0x0000, 0x0000, 0x01a1},
		{0x0008, 0x0000, 0x01a8},
		{0x002c, 0x0100, 0x01e9},
		{0x0000, 0x0000, 0x01ea},
		{0x0010, 0x078b, 0x01eb},
		{0x0064, 0x0000, 0x01ec},
		{0x0000, 0x0000, 0x0238},
		{0x0002, 0x0000, 0x0239},
		{0x0000, 0x0000, 0x02c8},
		{0x2000, 0x7803, 0x0310},
		{0x4b00, 0x00cf, 0x0311},
		{0x0000, 0x0000, 0x0312},
		{0x0000, 0x0002, 0x0313},
	};


/// gemm_opt_1
volatile unsigned short cin_gemm_opt_1[29][3] __attribute__((aligned(8))) = {
		{0x2800, 0x6400, 0x0020},
		{0xfa00, 0x80f7, 0x0021},
		{0x4000, 0x0001, 0x0022},
		{0x0000, 0x0002, 0x0023},
		{0x2800, 0x6400, 0x0028},
		{0xfa00, 0x80f7, 0x0029},
		{0x4000, 0x0001, 0x002a},
		{0x0000, 0x0962, 0x002b},
		{0x0000, 0x0000, 0x002c},
		{0x2000, 0x6400, 0x0030},
		{0x0040, 0x80f0, 0x0031},
		{0x4e89, 0x0001, 0x0032},
		{0x0000, 0x0002, 0x0033},
		{0x0030, 0x0000, 0x0070},
		{0x0000, 0x0010, 0x0078},
		{0x8000, 0x0000, 0x0100},
		{0x0000, 0x0000, 0x0101},
		{0x0060, 0x0000, 0x0109},
		{0x060c, 0x2100, 0x0149},
		{0x0000, 0x0000, 0x0198},
		{0x200a, 0x1300, 0x01d9},
		{0x0100, 0x0000, 0x0220},
		{0x0000, 0x3fc0, 0x0260},
		{0x000a, 0x2000, 0x0261},
		{0x0000, 0x0000, 0x02b0},
		{0x0000, 0x6400, 0x02f0},
		{0x0040, 0x80f0, 0x02f1},
		{0x4000, 0x0001, 0x02f2},
		{0x0000, 0x0002, 0x02f3},
	};


/// gemm_0
volatile unsigned short cin_gemm_0[13][3] __attribute__((aligned(8))) = {
		{0x2000, 0x6400, 0x0028},
		{0x0000, 0x0000, 0x0029},
		{0x0000, 0x0000, 0x002a},
		{0x0000, 0x0002, 0x002b},
		{0x2800, 0x6400, 0x0030},
		{0x0000, 0x0000, 0x0031},
		{0x0000, 0x0000, 0x0032},
		{0x0000, 0x08c2, 0x0033},
		{0x0020, 0x0000, 0x0034},
		{0x0000, 0x0000, 0x0078},
		{0x0001, 0x0000, 0x0080},
		{0x999a, 0x3f99, 0x00c0},
		{0x000a, 0x0100, 0x00c1},
	};


void gemm(void* arg_0 ,void* arg_1 ,void* arg_2){
  for (int int_3 = 0; int_3 < 20; int_3 = int_3 + 1){
    {
    /// %0 = ADORA.BlockLoad %arg0 [%arg3, 0] : memref<?x25xf32> -> memref<1x25xf32>  {Id = "0", KernelName = "gemm_0"}
    uint64_t dramoffset_0 = 100 * int_3;
    uint64_t spadoffset_0 = 0;
    uint64_t roffset_0 = 0;
    load_data(arg_0 + dramoffset_0 + roffset_0, 0x8000 + spadoffset_0, 100, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_0 = spadoffset_0 + 100;
    
    }
    {
    /// gemm_0
    load_cfg((void*)cin_gemm_0, 0x20000, 78, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 13, _task_id, 0);
    execute(0x30, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %1, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_0"}
    uint64_t dramoffset_1 = 100 * int_3;
    uint64_t spadoffset_1 = 0;
    uint64_t roffset_1 = 0;
    store(arg_0 + dramoffset_1 + roffset_1, 0xa000 + spadoffset_1, 100, _task_id, 0);
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
    load_data(arg_2 + dramoffset_3 + roffset_3, 0x18000 + spadoffset_3, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
    spadoffset_3 = spadoffset_3 + 3000;
    
    }
    {
    /// gemm_1
    load_cfg((void*)cin_gemm_1, 0x20000, 240, _task_id, LD_DEP_EX_LAST_TASK);
    config(0x0, 40, _task_id, 0);
    execute(0x80b0, _task_id, EX_DEP_ST_LAST_TASK);
    }
    {
    /// ADORA.BlockStore %4, %arg0 [%arg3, 0] : memref<1x25xf32> -> memref<?x25xf32>  {Id = "4", KernelName = "gemm_1"}
    uint64_t dramoffset_4 = 100 * int_3;
    uint64_t spadoffset_4 = 0;
    uint64_t roffset_4 = 0;
    store(arg_0 + dramoffset_4 + roffset_4, 0xc000 + spadoffset_4, 100, _task_id, 0);
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
  load_data(arg_0 + dramoffset_0 + roffset_0, 0x0 + spadoffset_0, 2000, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_0 = spadoffset_0 + 2000;
  
  }
  {
  /// gemm_opt_0
  load_cfg((void*)cin_gemm_opt_0, 0x20000, 72, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 12, _task_id, 0);
  execute(0x6, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %1, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "1", KernelName = "gemm_opt_0"}
  uint64_t dramoffset_1 = 0;
  uint64_t spadoffset_1 = 0;
  uint64_t roffset_1 = 0;
  store(arg_0 + dramoffset_1 + roffset_1, 0x2000 + spadoffset_1, 2000, _task_id, 0);
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
  load_data(arg_2 + dramoffset_3 + roffset_3, 0x8000 + spadoffset_3, 3000, 0, _task_id, LD_DEP_ST_LAST_TASK);
  spadoffset_3 = spadoffset_3 + 3000;
  
  }
  {
  /// gemm_opt_1
  load_cfg((void*)cin_gemm_opt_1, 0x20000, 174, _task_id, LD_DEP_EX_LAST_TASK);
  config(0x0, 29, _task_id, 0);
  execute(0x838, _task_id, EX_DEP_ST_LAST_TASK);
  }
  {
  /// ADORA.BlockStore %4, %arg0 [0, 0] : memref<20x25xf32> -> memref<?x25xf32>  {Id = "4", KernelName = "gemm_opt_1"}
  uint64_t dramoffset_4 = 0;
  uint64_t spadoffset_4 = 0;
  uint64_t roffset_4 = 0;
  store(arg_0 + dramoffset_4 + roffset_4, 0xa000 + spadoffset_4, 2000, _task_id, 0);
  spadoffset_4 = spadoffset_4 + 2000;
  
  }
  _task_id++;
  fence(1);
}

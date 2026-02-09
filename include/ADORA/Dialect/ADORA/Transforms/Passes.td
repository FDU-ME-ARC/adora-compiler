//===-- Passes.td - ADORA pass definition file --------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORA Passes
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_DIALECT_PASSES
#define ADORA_DIALECT_PASSES
include "ADORA/Dialect/ADORA/IR/ADORABase.td"
include "mlir/Pass/PassBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/Dialect/SCF/IR/SCFOps.td"
include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.td"
include "mlir/Dialect/Affine/IR/AffineOps.td"

//===----------------------------------------------------------------------===//
// CDFGgen for ADORA
//===----------------------------------------------------------------------===//
def ADORALoopCdfgGen : Pass<"adora-kernel-dfg-gen", "ModuleOp"> {
  let summary = "Generate cdfg of ADORA from a kernel loop";
  let constructor = "mlir::ADORA::createADORALoopCdfgGenPass()";
  let dependentDialects = [
    "arith::ArithDialect",
    "::mlir::ADORA::ADORADialect",
    "::mlir::scf::SCFDialect",
    "::mlir::affine::AffineDialect"
  ];
  let options = [
     Option<"Verbose", "verbose", "bool",
           /*default=*/"false",
           "Output detailed process of generating cdfg and intermediate cdfgs(default:false)">,
  ];
}

//===----------------------------------------------------------------------===//
// Affine To kernel
//===----------------------------------------------------------------------===//

def ExtractAffineForToKernel : Pass<"adora-extract-affine-for-to-kernel", "func::FuncOp"> {
  let summary = "Extract all AffineFor in a FuncOp to a ADORA.KernelOp";
  let constructor = "mlir::ADORA::createExtractAffineForToKernelPass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect"
    // "arith::AffineDialect"
  ];
  let options = [
     Option<"FunctionName", "function-name", "std::string",
           /*default=*/"\"-\"",
           "Specify the functions to search for loops, separated by commas (,). Default to be all functions. Example: --function-name=foo0,foo1">,
  ];
}


//===----------------------------------------------------------------------===//
// Kernel to independent mlir Func
//===----------------------------------------------------------------------===//
def ExtractKernelToFunc : Pass<"adora-extract-kernel-to-function", "ModuleOp"> {
  let summary = "Extract all ADORA.KernelOp and its region to a single func with explicit arguments";
  
  let description = [{
  Following 2 examples shows what is Explicit Data Trans
  example 1:
  Initial:
  "
  #map0 = affine_map<(d0) -> (d0)>
  #map1 = affine_map<(d0) -> (d0 + 2)>
  Host:
  %c0_i32 = arith.constant 0 : i32
  %0 = memref.get_global @m1 : memref<1024xi32>
  %1 = memref.get_global @m2 : memref<1024xi32>
  %2 = memref.get_global @prod : memref<1024xi32>
  affine.for %arg0 = 0 to 32 step 2 {
    func.call @gemm_kernel_0(%0, %1, %2, %arg0) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, index) -> ()
  }

  Kernel:
  func.func @gemm_kernel_0(%arg0: memref<1024xi32>, %arg1: memref<1024xi32>, %arg2: memref<1024xi32>, %arg3: index) attributes {gemm_kernel_0} {
    cf.br ^bb1
    ^bb1:  // pred: ^bb0
      %c0_i32 = arith.constant 0 : i32
      affine.for %arg4 = #map0(%arg3) to #map1(%arg3) {
        affine.for %arg5 = 0 to 32 {
          %0 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %c0_i32) -> (i32) {
            %1 = affine.load %arg0[%arg6 + %arg4 * 32] : memref<1024xi32>
            %2 = affine.load %arg1[%arg5 + %arg6 * 32] : memref<1024xi32>
            %3 = arith.muli %1, %2 : i32
            %4 = arith.addi %arg7, %3 : i32
            affine.yield %4 : i32
          }
          affine.store %0, %arg2[%arg5 + %arg4 * 32] : memref<1024xi32>
        }
      }
      return
    }
  }
  "
  
  Converted to:
  "
  Host:

    #map0 = affine_map<(d0) -> (d0 * 2)>
    #map1 = affine_map<(d0)[s0] -> (d0 + s0)>
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @m1 : memref<1024xi32>
    %1 = memref.get_global @m2 : memref<1024xi32>
    %2 = memref.get_global @prod : memref<1024xi32>
    affine.for %arg0 = 0 to 32 step 8 {
      %3 = ADORA.DataBLock_Load %0[%arg0], 256 : memref<256xi32> attributes {gemm_kernel_0}
      %4 = ADORA.DataBLock_Load %1[%cst_0], 1024 : memref<1024xi32> attributes {gemm_kernel_0}
      %5 = ADORA.DataBLock_Load %2[%arg0], 256 : memref<256xi32> attributes {gemm_kernel_0}
      ADORA.func @gemm_kernel_0(%3, %4, %5) : (memref<256xi32>, memref<1024xi32>, memref<256xi32>) -> ()
      %5 = ADORA.DataBLock_Store %2[%arg0], %5, 256 : memref<256xi32> attributes {gemm_kernel_0}
    }

  func.func @gemm_kernel_0(%arg0: memref<256xi32>, %arg1: memref<1024xi32>, %arg2: memref<256xi32>,) attributes {gemm_kernel_0} {
    cf.br ^bb1
  ^bb1:  // pred: ^bb0
    %c0_i32 = arith.constant 0 : i32
    affine.for %arg4 = 0 to 8 {
      affine.for %arg5 = 0 to 32 {
        %1 = affine.for %arg6 = 0 to 32 iter_args(%arg7 = %c0_i32) -> (i32) {
          %2 = affine.load %arg0[%arg6 + %0 * 32] : memref<1024xi32>
          %3 = affine.load %arg1[%arg5 + %arg6 * 32] : memref<1024xi32>
          %4 = arith.muli %2, %3 : i32
          %5 = arith.addi %arg7, %4 : i32
          affine.yield %5 : i32
        }
        affine.store %1, %arg2[%arg5 + %0 * 32] : memref<1024xi32>
      }
    }
    return
  } 
  "
  '''

  example 2:
  Initial:
  #map0 = affine_map<(d0) -> (d0)>
  #map1 = affine_map<(d0) -> (d0 + 8)> 
  func.func @gemm() attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @m1 : memref<1024xi32>
    %1 = memref.get_global @m2 : memref<1024xi32>
    %2 = memref.get_global @prod : memref<1024xi32>
    affine.for %arg0 = 0 to 32 step 8 {
      ADORA.kernel {
        affine.for %arg1 = #map0(%arg0) to #map1(%arg0) {
          affine.for %arg2 = 0 to 32 {
            %3 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %c0_i32) -> (i32) {
              %4 = affine.load %0[%arg3 + %arg1 * 32] : memref<1024xi32>
              %5 = affine.load %1[%arg2 + %arg3 * 32] : memref<1024xi32>
              %6 = arith.muli %4, %5 : i32
              %7 = arith.addi %arg4, %6 : i32
              affine.yield %7 : i32
            }
            affine.store %3, %2[%arg2 + %arg1 * 32] : memref<1024xi32>
          }
        }
        ADORA.terminator
      }
    }
    return
  }

  Converted to:

  #map0 = affine_map<(d0) -> (d0)>
  #map1 = affine_map<(d0) -> (d0 + 8)> 
  func.func @gemm() attributes {llvm.linkage = #llvm.linkage<external>} {
    %c0_i32 = arith.constant 0 : i32
    %0 = memref.get_global @m1 : memref<1024xi32>
    %1 = memref.get_global @m2 : memref<1024xi32>
    %2 = memref.get_global @prod : memref<1024xi32>
    affine.for %arg0 = 0 to 32 step 8 {
      %new_0 = ADORA.DataBLock_Load %0 [%arg0 * 32] : memref<256xi32> attributes {gemm_kernel_0}
      %new_1 = ADORA.DataBLock_Load %1 [%cst_0 * 32] : memref<1024xi32> attributes {gemm_kernel_0}
      %new_2 = ADORA.DataBLock_Load %2 [%arg0 * 32] : memref<256xi32> attributes {gemm_kernel_0}
      ADORA.kernel {
        affine.for %arg1 = 0 to 8 {
          affine.for %arg2 = 0 to 32 {
            %3 = affine.for %arg3 = 0 to 32 iter_args(%arg4 = %c0_i32) -> (i32) {
              %4 = affine.load %new_0[%arg3 + %arg1 * 32] : memref<1024xi32>
              %5 = affine.load %new_1[%arg2 + %arg3 * 32] : memref<1024xi32>
              %6 = arith.muli %4, %5 : i32
              %7 = arith.addi %arg4, %6 : i32
              affine.yield %7 : i32
            }
            affine.store %3, %new_2[%arg2 + %arg1 * 32] : memref<1024xi32>
          }
        }
        ADORA.terminator
      }
      ADORA.DataBLock_Store %new_2, %2[%arg0 * 32]: memref<256xi32> -> memref<1024xi32>  {gemm_kernel_0}
    }
    return
  }
  '''
  }];  

  let constructor = "mlir::ADORA::createExtractKernelToFuncPass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect",
    "::mlir::cf::ControlFlowDialect"
    // "arith::AffineDialect"
  ];
  let options = [
    Option<"KernelGenDir", "kernel-gen-dir", "std::string", /*default=*/"",
           "Kernels will not be generated unless the dir path is set">,
    // Option<"ExplicitDataTrans", "kernel-explicit-datablock-trans", "bool", /*default=*/"true",
    //        "This Option (Default:Ture) will generate explict data block loads/stores ,and interface of Kernel Func won't contain affine-transformed loop Index arguments.">,
  ];
}


//===----------------------------------------------------------------------===//
// ADORA Simplify loop structure
//===----------------------------------------------------------------------===//

def SimplifyAffineLoopLevels : Pass<"adora-simplify-affine-loop-levels", "ModuleOp"> {
  let summary = "simplify stripcounts of affine loops";
  let description = [{
      
  }];
  let constructor = "mlir::ADORA::createSimplifyAffineLoopLevelsPass()";
  let options = [];
}


//===----------------------------------------------------------------------===//
// Loop unroll to achieve highest utilization
//===----------------------------------------------------------------------===//
def ADORAAffineLoopUnroll : Pass<"adora-affine-loop-unroll", "ModuleOp"> {
  let summary = "apply loop unroll to kernel function to achieve highest cgra utilization";
  let description = [{
    CGRAOPT applies loop unrolling to a kernel function from the innermost to outermost level,
    aiming to achieve enhanced instruction-level parallelism (ILP) and
    better utilization of hardware resources. To determine the optimal
    unroll factor (UF) for the kernel, the "-cgra-affine-loop-unroll" pass
    generates a data-flow graph (DFG) for each available UF. The pass
    calculates the utilization of these DFGs using the formula:
    Util = min(#L/S Nodes / #I/O Units, #Computing Nodes / #ALUs)
    where #L/S Nodes represent the number of Load/Store nodes,
    #I/O Units represent the number of I/O units, and #Computing Nodes
    represent the number of computing nodes. #ALUs represent the number
    of Arithmetic Logic Units.
    The overall utilization in the formula depends on the smaller value
    between the utilization of Input/Output blocks and that of ALUs.
  }];  
  let constructor = "mlir::ADORA::createADORAAffineLoopUnrollPass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect",
    "::mlir::affine::AffineDialect"
  ];
  let options = [
    // Option<"outputPath", "output-path", "std::string",
    //        /*default=*/"\"./\"",
    //        "File path: the path for dumping the MLIR of pareto design points">,
    // Option<"csvPath", "csv-path", "std::string",
    //        /*default=*/"\"./\"",
    //        "File path: the path for dumping the CSV of design spaces">,

    Option<"CGRAadg", "cgra-adg", "std::string",
           /*default=*/"\"notdefined\"",
           "File path: target cgra backend ADG(Architecture Description Graph)">,
    // Option<"llvmCDFGPass", "cdfg-pass-so", "std::string",
    //        /*default=*/"\"notdefined\"",
    //        "File path: the llvm shared library(.so) of DFG generator(app-compiler)">
  ];
}

//===----------------------------------------------------------------------===//
// Loop UnrollAndJam
//===----------------------------------------------------------------------===//
def ADORALoopUnrollAndJam : Pass<"adora-loop-unroll-jam", "ModuleOp"> {
  let summary = "Unroll and jam affine loops";
  let constructor = "mlir::ADORA::createADORALoopUnrollAndJamPass()";
  let options = [
    Option<"CGRAadg", "cgra-adg", "std::string",
           /*default=*/"\"notdefined\"",
           "File path: target cgra backend ADG(Architecture Description Graph)">,
    // Option<"unrollJamFactor", "unroll-jam-factor", "unsigned",
    //        /*default=*/"4",
    //        "Use this unroll jam factor for all loops (default 4)">,
  ];
}

//===----------------------------------------------------------------------===//
// Autometed Loop Unroll/UnrollAndJam
//===----------------------------------------------------------------------===//
def ADORAAutoUnroll : Pass<"adora-auto-unroll", "ModuleOp"> {
  let summary = "Automated unroll or unroll-and-jam based on dependency analysis.";
  let constructor = "mlir::ADORA::createADORAAutoUnrollPass()";
  let options = [
    Option<"CGRAadg", "cgra-adg", "std::string",
           /*default=*/"\"notdefined\"",
           "File path: target cgra backend ADG(Architecture Description Graph)">,
    // Option<"unrollJamFactor", "unroll-jam-factor", "unsigned",
    //        /*default=*/"4",
    //        "Use this unroll jam factor for all loops (default 4)">,
  ];
}

//===----------------------------------------------------------------------===//
// Adjust(Partition) Kernel according to a customized Cachesize
//===----------------------------------------------------------------------===//
def AdjustKernelMemoryFootprint : Pass<"adora-adjust-kernel-mem-footprint", "ModuleOp"> {
  let summary = "Adjust(partition) kernels' memory footprint to apply to customized cachesize";
  let constructor = "mlir::ADORA::createAdjustKernelMemoryFootprintPass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect"
    // "arith::AffineDialect"
  ];
  let options = [
    Option<"Cachesize_Kib", "cachesize", "unsigned", /*default=*/"512",
           "Set a cachesize(Kib) for kernel to be iterated(default to be 512Kib)">,
    Option<"SingleArray_Size", "singlearraysize", "unsigned", /*default=*/"0",
           "Set a cachesize(Kib) constrain for a single array or tensor in a kernel(default to be the same with Cachesize_Kib)">,
    Option<"AffineAccessPattern", "access-pattern", "bool", /*default=*/"false",
           "Set the memory access method to affine pattern(default: general pattern)">,
    Option<"DisableRemainderBlock", "disable-remainder-block", "bool", /*default=*/"false",
           "Disable make remainder during loop strip-mining">,
     Option<"ExplicitDataTrans", "explicit-datablock", "bool", /*default=*/"true",
           "This Option (Default:Ture) will generate explict data block loads/stores , and interface of Kernel Func won't contain affine-transformed loop Index arguments.">,
  ];
}


//===----------------------------------------------------------------------===//
// Simplify Load/Store Operation In Loop-Nest
//===----------------------------------------------------------------------===//
def SimplifyLoadStoreInLoopNest : Pass<"adora-simplify-loadstore", "func::FuncOp"> {
  let summary = "Simplify inner affine.load/store operations. (Such as hoist to outer level to decrease memmory accesses)";
  let description = [{
    This Pass will simplify repetitive affine.load/store operations which access 
    the same position in inner loop to outer level of loop so that memmory 
    accesses can be decreased.
    The optimization technique of moving memory accesses from the inner loop 
    to the outer loop is commonly known as "loop hoisting" or 
    "loop-invariant code motion" (LICM). The goal of this optimization 
    is to move invariant code from the inner loop to the outer loop, 
    reducing redundant computations and memory accesses, and thereby 
    improving the performance of the program.
    Loop hoisting effectively reduces the number of computations and memory 
    accesses within the loop, avoids redundant operations, and decreases the 
    number of iterations in the loop, resulting in improved program efficiency. 
    This optimization is typically performed automatically by the compiler's 
    optimizer, but it can also be manually applied to enhance code performance.
    '''
  }];  
  let constructor = "mlir::ADORA::createSimplifyLoadStoreInLoopNestPass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect",
    "::mlir::affine::AffineDialect"
  ];
  let options = [
    // Option<"OnlyInKernel", "only-in-kernel", "bool", /*default=*/"false",
    //        "Only hoist load/store in ADORA.kernel{}">,
  ];
}

//===----------------------------------------------------------------------===//
// Loop Reorder
//===----------------------------------------------------------------------===//
def AffineLoopReorder :
      Pass<"adora-loop-reorder", "func::FuncOp"> {
  let summary = "reorder/interchange affine loop nests";
  let description = [{
    This pass aims to reorganize affine loop nests using polyhedral-based dependency analysis. 
    Its primary goal is to enhance the data locality.
  }];
  let constructor = "mlir::ADORA::createAffineLoopReorderPass()";
}


//===----------------------------------------------------------------------===//
// Loop Reorder
//===----------------------------------------------------------------------===//
def ScheduleADORATasks: Pass<"adora-schedule-tasks", "func::FuncOp"> {
  let summary = "schedule CGRA tasks after";
  let description = [{
    Schedule CGRA tasks, simplify data movement. 
    Following situations need to be considered:
      1.
      2.
  }];
  let constructor = "mlir::ADORA::createScheduleADORATasksPass()";
}

//===----------------------------------------------------------------------===//
// Frontend Design Space Exploration
//===----------------------------------------------------------------------===//
def AutoDesignSpaceExplore : Pass<"adora-auto-dse", "ModuleOp"> {
  let summary = "design space explore for application frontend optimization";
  let description = [{
    This pass will automatically conduct design space exploration (DSE)
    for frontend optimization of riscV + coarsed-grained reconfigurable 
    array (CGRA) heterogeneous architeture. The frontend optimization DSE 
    is alike to optimization DSE in HLS of FPGA(like scalehls and comba) . 
    It takes in information about the hardware architecture(like array size,
    op types, etc.), includes loop SW/HW partition, loop tiles, loop unrolls ,etc. 
    and achieve architeture-independent optimization towards application code.
    This dse process is able to improve the performance of a application code
    when running on a heterogeneous architeture.
  }];  
  let constructor = "mlir::ADORA::createAutoDesignSpaceExplorePass()";
  let dependentDialects = [
    "::mlir::ADORA::ADORADialect"
    // "arith::AffineDialect"
  ];
  let options = [
    // Option<"outputPath", "output-path", "std::string",
    //        /*default=*/"\"./\"",
    //        "File path: the path for dumping the MLIR of pareto design points">,
    // Option<"csvPath", "csv-path", "std::string",
    //        /*default=*/"\"./\"",
    //        "File path: the path for dumping the CSV of design spaces">,

    Option<"CGRAadg", "cgra-adg", "std::string",
           /*default=*/"\"notdefined\"",
           "File path: target cgra backend ADG(Architecture Description Graph)">,
    Option<"llvmCDFGPass", "cdfg-pass-so", "std::string",
           /*default=*/"\"notdefined\"",
           "File path: the llvm shared library(.so) of DFG generator(app-compiler)">
  ];
}

#endif // ADORA_DIALECT_PASSES

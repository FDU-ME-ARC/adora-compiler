//===-- Passes.td - ADORA pass definition file --------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORA Lowering Passes
//
//===----------------------------------------------------------------------===//
#ifndef ADORA_LOWERING_PASSES
#define ADORA_LOWERING_PASSES      
include "ADORA/Dialect/ADORA/IR/ADORABase.td"
include "mlir/Pass/PassBase.td"

def ConvertKernelCallToLLVM : Pass<"adora-convert-kernelcall-to-llvm", "ModuleOp"> {
  let summary = "lower ADORA's kernelcall op to LLVM dialect";
  let description = [{
    Convert Func dialect operations into the LLVM IR dialect operations.
    Note: This pass must be used after "--finalize-memref-to-llvm". 
    Also, we recommend use "--cse --canonicalize" to simplify mlir after use this pass.
    So a recommended pass pipeline is:
    """
    --arith-expand --memref-expand  \
    -normalize-memrefs --expand-strided-metadata  --affine-simplify-structures \
    -lower-affine --scf-for-loop-canonicalization  -convert-scf-to-cf\
    --convert-math-to-llvm --convert-math-to-libm\
    --convert-arith-to-llvm \
    --finalize-memref-to-llvm \
    --adora-convert-kernelcall-to-llvm \
    -convert-func-to-llvm=use-bare-ptr-memref-call-conv \
    --cse --canonicalize
    """

    #### Input invariant

    -   no `tensor` types;
    -   all `vector` are one-dimensional;
    -   all blocks are reachable by following the successors of the first basic
        block;

    If other operations are present and their results are required by the LLVM
    IR dialect operations, the pass will fail.  Any LLVM IR operations or types
    already present in the IR will be kept as is.

    #### Output IR

    Functions converted to LLVM IR. Function arguments types are converted
    one-to-one. Function results are converted one-to-one and, in case more than
    1 value is returned, packed into an LLVM IR struct type. Function calls and
    returns are updated accordingly. Block argument types are updated to use
    LLVM IR types.
  }];
  let constructor = "createConvertKernelCallToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect","::mlir::ADORA::ADORADialect"];
}

def ConvertADORAToSCF : Pass<"adora-convert-loadstore-to-scf", "ModuleOp"> {
  let summary = "lower ADORA's blockload/store op to SCF dialect";
  let description = [{
    Convert ADORA dialect operations into the scf dialect operations.

  }];
  let constructor = "createConvertADORAToSCFPass()";
  let dependentDialects = ["scf::SCFDialect","::mlir::ADORA::ADORADialect"];
}

def MathRewrite : Pass<"adora-math-rewrite", "ModuleOp"> {
  let summary = "Rewrite math op with a combination of simple arith ops using some fast approximation algorithm";
  let description = [{
  Approximate/Lower math op with a combination of arith ops.
  Note: Please use this pass before "-arith-expand" pass.

  math.rsqrt:
  """
  %y = math.rsqrt %x : f32
  """
  lower to
  """
  %cst_1_2 = arith.const 0.5 : f32
  %x2 = arith.mulf %x, %cst_1/2 : f32

  %i0 = arith.bitcast %x : f32  to i32
  %cst_1i = arith.const 1 : i32
  %i1 = arith.shrui %i0, %cst_1i : i32
  %cst_magic = arith.const 1597463007 : i32 // 0x5f3759df
  %i2 = arith.subi %cst_magic, %i1 : i32
  %f0 = arith.bitcast %i2 : i32 to f32

  %cst_3_2 = arith.const 1.5 : f32
  %f1 = arith.mulf %f0, %f0 : f32
  %f2 = arith.mulf %f1, %x2 : f32
  %f3 = arith.subf %cst_3_2, %f2 : f32
  %y = arith.mulf %f3, %f1: f32
  """

  arith.max/arith.min

  }];
  let constructor = "createMathRewritePass()";
  let dependentDialects = ["math::MathDialect","arith::ArithDialect"];
}

def ADORAAsyncRuntimeToLLVM : Pass<"adora-to-llvm-async-runtime", "ModuleOp"> {
  let summary = "Lower ADORA async event ops to llvm.call into the runtime ABI";
  let description = [{
    Translates the four async-event ops emitted by adora-lower-async-tokens into
    llvm.call instructions targeting the adora_async_rt.h ABI:

      ADORA.event.create  -> llvm.call @adoraEventCreate()  -> llvm.ptr
      ADORA.event.destroy -> llvm.call @adoraEventDestroy(ptr)
      ADORA.signal        -> llvm.call @adoraEventRecord(ptr, stream_i64)
      ADORA.wait          -> llvm.call @adoraEventWait(ptr, stream_i64)

    !ADORA.token values are replaced by opaque llvm.ptr.
    Extern declarations for the four runtime functions are inserted at module
    scope if not already present.
  }];
  let constructor = "mlir::ADORA::createADORAAsyncRuntimeToLLVMPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // ADORA_DIALECT_PASSES

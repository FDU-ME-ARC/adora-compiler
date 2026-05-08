//===-- ADORAOps.td - ADORA dialect operation definitions ------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines some operations of the ADORA dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_KERNEL_OP
#define ADORA_KERNEL_OP

include "ADORA/Dialect/ADORA/IR/ADORABase.td"
// include "ADORA/Dialect/ADORA/IR/ADORAOps.td"

include "mlir/Pass/PassBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

include "mlir/Dialect/Linalg/IR/LinalgBase.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
include "mlir/Dialect/Arith/IR/ArithBase.td"
include "mlir/Dialect/Arith/IR/ArithOpsInterfaces.td"

// include "mlir/Dialect/Affine/IR/AffineOps.td"

// include "mlir/Dialect/Func/IR/FuncOps.td"
// include "mlir/Dialect/Func/IR/FuncOps.h"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"


//===----------------------------------------------------------------------===//
// kernel Operations for offload HW code
//===----------------------------------------------------------------------===//

def ADORA_KernelOp : ADORA_Op<"kernel">,
    Arguments<(ins Variadic<ADORA_TokenType>:$asyncDependencies)>,
    Results<(outs Optional<ADORA_TokenType>:$asyncToken)> {
  let summary = "ADORA KernelOp contains a kernel";

  let description = [{
    ADORA.kernel define a region that the body of the rigion
    is a kernel which generated from a affine.for/scf.for/linalg.generic.

    Optionally produces an `!ADORA.token` and consumes a variadic list of
    tokens via `asyncDependencies`, encoding happen-before ordering between
    ADORA async ops. Token fields are populated by `adora-schedule-tasks`;
    ops created pre-schedule have an empty `asyncDependencies` list and may
    omit `asyncToken`.
  }];

  let regions = (region AnyRegion:$body);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<(ins)>,
    OpBuilder<(ins "std::string":$KernelName)>,
    OpBuilder<(ins "std::string":$KernelName,
                   "::mlir::ValueRange":$asyncDependencies,
                   "bool":$produceToken)>,
    // Rebuild-friendly async builder: same as the previous one but transfers
    // the body region from `takeFromRegion` via Region::takeBody, instead of
    // creating an empty body. Used by ScheduleADORATasks' threadTokensOnDMAs
    // to rewrite an existing sync KernelOp into its async form while keeping
    // the original kernel body intact.
    OpBuilder<(ins "std::string":$KernelName,
                   "::mlir::ValueRange":$asyncDependencies,
                   "bool":$produceToken,
                   "::mlir::Region*":$takeFromRegion)>
  ];
  
  let extraClassDeclaration = [{

    /// The number of launch configuration operands, placed at the leading
    /// positions of the operand list.
    /// Legacy from copying GPU dialect.
    static constexpr unsigned kNumConfigOperands = 0;

    /// The number of region attributes containing the launch configuration,
    /// placed in the leading positions of the argument list.
    /// Legacy from copying GPU dialect.
    static constexpr unsigned kNumConfigRegionAttributes = 0;
    // unsigned kernel_Count = 0;

    static StringRef getKernelNameAttrStr() { return "KernelName"; }
    void setKernelName(std::string s) {
      StringAttr KernelNameAttr = StringAttr::get(getOperation()->getContext(),s);
      getOperation()->setAttr(getKernelNameAttrStr(), KernelNameAttr);
    }

    bool hasKernelName(){
      return getOperation()->hasAttr("KernelName");
    }

    std::string getKernelName(){
      return hasKernelName()? 
        mlir::cast<mlir::StringAttr>(getOperation()->getAttr("KernelName")).str() :
        "";
    }

    ::mlir::affine::AffineForOp getTopAffineFopInKernel(){
      SmallVector<::mlir::affine::AffineForOp> topforops;
      for(::mlir::affine::AffineForOp forop: getOps<::mlir::affine::AffineForOp>()){
        topforops.push_back(forop);
      }
      assert(topforops.size() == 1);
      return topforops[0];
    }

  }];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}



def ADORA_TerminatorOp : ADORA_Op<"terminator", [HasParent<"KernelOp">,
                                              Terminator]>,
    Arguments<(ins)>, Results<(outs)> {
  let summary = "Terminator for ADORA_KernelOp regions.";
  let description = [{
    A terminator operation for regions that appear in the body of ADORA_KernelOp
    operation.  These regions are not expected to return any value so the
    terminator takes no operands.
  }];

  let assemblyFormat = "attr-dict";
}


#endif // ADORA_KERNEL_OP

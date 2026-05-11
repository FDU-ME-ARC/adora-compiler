//===-- ADORA_BASE.td - Test dialect definitions ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
// Defines the ADORA dialect
// 
//===----------------------------------------------------------------------===//

#ifndef ADORA_BASE
#define ADORA_BASE

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinDialect.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

//===----------------------------------------------------------------------===//
// ADORA Dialect.
//===----------------------------------------------------------------------===//

def ADORA_Dialect : Dialect {
  let name = "ADORA";
  let cppNamespace = "::mlir::ADORA";

  let description = [{
    "a dialect for cgra compiler-ADORA"
  }];

  let dependentDialects = [
    // "scf::SCFDialect",
    // "::mlir::func"
  ];

  // PR2: generate printType / parseType from the ODS type definitions so
  // that !ADORA.token (and any future ADORA types) can be printed/parsed
  // without manually implementing the dialect type printing hook.
  let useDefaultTypePrinterParser = 1;

  // let hasOperationAttrVerify = 1;
}

//===----------------------------------------------------------------------===//
// ADORA Dialect operations.
//===----------------------------------------------------------------------===//
class ADORA_Op<string mnemonic, list<Trait> traits = []> :
    Op<ADORA_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ADORA Types.
//===----------------------------------------------------------------------===//
class ADORA_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ADORA_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def ADORA_TokenType : ADORA_Type<"Token", "token"> {
  let summary = "Ordering-only handle between ADORA data-block / kernel ops.";
  let description = [{
    An `!ADORA.token` value represents a happen-before edge between ADORA
    async operations (BlockLoad / kernel / BlockStore). Tokens carry no
    payload; they are produced by op's `asyncToken` result and consumed
    through `asyncDependencies` operand.
  }];
}

#endif // ADORA_BASE

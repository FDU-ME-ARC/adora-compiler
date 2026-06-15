//===- ADORADialects.cpp - CAPI impl for ADORA dialect ---------*- C++ -*-===//
//
// Defines the C API entry point mlirGetDialectHandle__adora__ that lets the
// upstream MLIR Python bindings register the out-of-tree ADORA dialect, so
// IR containing ADORA ops can be parsed and walked from Python.
//
//===----------------------------------------------------------------------===//

#include "ADORA/CAPI/Dialects.h"

#include "ADORA/Dialect/ADORA/IR/ADORA.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(ADORA, adora,
                                      mlir::ADORA::ADORADialect)

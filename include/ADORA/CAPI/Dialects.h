//===- Dialects.h - CAPI for ADORA dialect registration --------*- C -*-===//
//
// C API to register the out-of-tree ADORA dialect into an MlirContext /
// MlirDialectRegistry, so the upstream MLIR Python bindings can parse and
// walk IR that contains ADORA ops (ADORA.kernel, ADORA.BlockLoad, ...).
//
//===----------------------------------------------------------------------===//

#ifndef ADORA_CAPI_DIALECTS_H
#define ADORA_CAPI_DIALECTS_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(ADORA, adora);

#ifdef __cplusplus
}
#endif

#endif // ADORA_CAPI_DIALECTS_H

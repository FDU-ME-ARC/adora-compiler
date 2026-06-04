//===- ADORAModule.cpp - Python extension for ADORA dialect ----*- C++ -*-===//
//
// A standalone nanobind extension that the upstream MLIR Python bindings
// auto-discover (via the `register_dialects` attribute scanned in
// mlir/_mlir_libs/__init__.py). It inserts the ADORA dialect into the
// global DialectRegistry so `mlir.ir.Module.parse` accepts ADORA ops.
//
//===----------------------------------------------------------------------===//

#include "ADORA/CAPI/Dialects.h"

#include "mlir/Bindings/Python/Nanobind.h"
#include "mlir/Bindings/Python/NanobindAdaptors.h"

namespace nb = nanobind;

NB_MODULE(_adoraDialectsRegister, m) {
  m.doc() = "Registers the out-of-tree ADORA dialect into upstream MLIR.";

  // Auto-discovered by mlir/_mlir_libs/__init__.py: any module under
  // _mlir_libs exposing `register_dialects(registry)` is called at import
  // time with the global MlirDialectRegistry.
  m.def(
      "register_dialects",
      [](MlirDialectRegistry registry) {
        mlirDialectHandleInsertDialect(mlirGetDialectHandle__adora__(),
                                       registry);
      },
      nb::arg("registry"));

  // Convenience: eagerly load ADORA into an existing context.
  m.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__adora__();
        mlirDialectHandleRegisterDialect(handle, context);
        if (load)
          mlirDialectHandleLoadDialect(handle, context);
      },
      nb::arg("context"), nb::arg("load") = true);
}

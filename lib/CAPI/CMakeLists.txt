# CAPI registration for the out-of-tree ADORA dialect, plus a standalone
# nanobind extension that the upstream MLIR Python bindings auto-discover
# (mlir/_mlir_libs/__init__.py scans for a `register_dialects` attribute).
#
# CRITICAL: the extension must share the SAME MLIR runtime as the upstream
# _mlir.*.so (a single libMLIRPythonCAPI.so). If we statically link MLIR
# core into this module we get a second copy of every dialect (builtin,
# func, ...) and registration aborts with "different dialects for the same
# namespace: builtin". So we link ONLY the ADORA dialect objects here and
# resolve all MLIR core symbols at runtime against the already-loaded
# libMLIRPythonCAPI.so.

set(_MLIR_PYTHON_CAPI_DIR
  "/data00/home/loujiahang/CGRVOPT/llvm-project-onnx/build/tools/mlir/python_packages/mlir_core/mlir/_mlir_libs"
  CACHE PATH "Directory holding the shared libMLIRPythonCAPI.so")

# ADORA-only CAPI registration objects (no MLIR core embedded here).
add_mlir_public_c_api_library(ADORACAPI
  Dialect/ADORADialects.cpp

  LINK_LIBS PUBLIC
  MLIRADORAOps
  MLIRADORAKernelOp
)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  find_package(nanobind CONFIG REQUIRED)

  nanobind_add_module(_adoraDialectsRegister
    NB_STATIC
    Bindings/Python/ADORAModule.cpp
  )

  target_link_libraries(_adoraDialectsRegister
    PRIVATE
    ADORACAPI
    ${_MLIR_PYTHON_CAPI_DIR}/libMLIRPythonCAPI.so
  )

  # MLIR core symbols (MLIRContext, builtin/func/affine dialects, parser,
  # ...) are satisfied at runtime by the already-loaded libMLIRPythonCAPI.so
  # that the upstream _mlir module pulls in. Allow them to be unresolved at
  # link time and bind the runpath so the loader can find the shared lib.
  target_link_options(_adoraDialectsRegister PRIVATE
    "LINKER:--allow-shlib-undefined"
    "LINKER:-rpath,${_MLIR_PYTHON_CAPI_DIR}"
  )
endif()

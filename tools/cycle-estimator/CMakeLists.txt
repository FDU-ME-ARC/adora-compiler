# ADORA cycle-estimator: an importable Python package.
#
# Installed as a Python package tree under <prefix>/bin/pypack/cycle_estimator
# so it can be invoked as:
#     PYTHONPATH=<prefix>/bin/pypack python -m cycle_estimator --mlir k.mlir
#
# The estimator reads loop trip-counts strong-typed via the upstream MLIR
# Python bindings (mlir_core) + the bundled adora_mlir dialect package; set
# ADORA_MLIR_CORE to the LLVM build's python_packages/mlir_core at runtime.

set(_CE_DEST ${CMAKE_INSTALL_PREFIX}/bin/pypack/cycle_estimator)

install(DIRECTORY
          ${CMAKE_CURRENT_SOURCE_DIR}/core
          ${CMAKE_CURRENT_SOURCE_DIR}/extract
          ${CMAKE_CURRENT_SOURCE_DIR}/arch
          ${CMAKE_CURRENT_SOURCE_DIR}/scripts
          ${CMAKE_CURRENT_SOURCE_DIR}/adora_mlir
        DESTINATION ${_CE_DEST}
        FILES_MATCHING
          PATTERN "*.py"
          PATTERN "*.so"
          PATTERN "*.json"
          PATTERN "__pycache__" EXCLUDE)

install(FILES
          ${CMAKE_CURRENT_SOURCE_DIR}/run.py
        DESTINATION ${_CE_DEST})

# Expose the package as `cycle_estimator` with a __main__ entry so that
# `python -m cycle_estimator` runs the estimator.
install(CODE "file(WRITE \"${_CE_DEST}/__init__.py\" \"\")")
install(CODE "file(WRITE \"${_CE_DEST}/__main__.py\" \"from run import main\\nmain()\\n\")")

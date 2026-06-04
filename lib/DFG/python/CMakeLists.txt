# Python bindings exposing the ADORA CDFG to the cycle-estimator.
# Uses the pip-installed pybind11 (RTTI is ON in this LLVM build) and does NOT
# depend on MLIR_ENABLE_BINDINGS_PYTHON.

find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
  message(WARNING "pybind11 not found; skipping _adora_cdfg module. "
                  "Run: pip install pybind11, then re-configure.")
  return()
endif()

pybind11_add_module(_adora_cdfg MODULE CDFGBindings.cpp)

target_include_directories(_adora_cdfg
  PRIVATE ${PROJ_INCLUDE_DIR}
  PRIVATE ${PROJ_INCLUDE_DIR}/lib/DFG/inc
  PRIVATE ${LLVM_INCLUDE_DIRS}
  PRIVATE ${MLIR_INCLUDE_DIRS}
)

target_link_libraries(_adora_cdfg
  PRIVATE GeneralCDFGGen
  PRIVATE MLIRADORAOps
  PRIVATE MLIRIR
  PRIVATE MLIRParser
  PRIVATE MLIRSCFDialect
  PRIVATE MLIRAffineDialect
  PRIVATE MLIRArithDialect
  PRIVATE MLIRFuncDialect
  PRIVATE MLIRMemRefDialect
  PRIVATE MLIRMathDialect
)

add_dependencies(_adora_cdfg GeneralCDFGGen ADORAPassIncGen)

# Drop the .so where the estimator's cdfg_native.py looks for it.
set_target_properties(_adora_cdfg PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
)

install(TARGETS _adora_cdfg
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/bin/pypack
)

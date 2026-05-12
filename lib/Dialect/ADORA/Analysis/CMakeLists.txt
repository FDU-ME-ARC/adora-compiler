add_mlir_dialect_library(MLIRADORAAnalysis
  AccessRegion.cpp
  LoopCarriedDep.cpp
  DepSummaryView.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Analysis
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRSupport
  MLIRAffineDialect
  MLIRSCFDialect
  MLIRADORAOps
  )

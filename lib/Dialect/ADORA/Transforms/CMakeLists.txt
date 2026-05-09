add_mlir_dialect_library(MLIRADORATransforms
  AssignStreams.cpp
  DependencyAnalysis.cpp
  DFGgenPass.cpp
  LowerAsyncTokens.cpp
  ScheduleAdoraTasks.cpp
  SimplifyLoadStore.cpp

  TaskGraph/TaskNode.cpp
  TaskGraph/TaskGraph.cpp
  TaskGraph/DepSummaryView.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRADORALoopTransforms
  MLIRADORAKernelExtract
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRADORATransformsUtility

  # MLIRPass
  # MLIRTransforms
  # MLIRSCFDialect
  # MLIRAffineToStandard
  )

add_subdirectory(Loop)
add_subdirectory(Kernel)
add_mlir_dialect_library(MLIRADORATransforms
  BufferReuse.cpp
  DFGgenPass.cpp
  SimplifyLoadStore.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskPipeline
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Transforms/TaskPipeline/TaskGraph
  ${PROJ_INCLUDE_DIR}/lib/DFG/inc

  DEPENDS
  ADORAPassIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRADORATaskPipeline
  MLIRADORALoopTransforms
  MLIRADORAKernelExtract
  MLIRADORAAnalysis
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRADORATransformsUtility

  MLIRTransforms
  # MLIRPass
  # MLIRSCFDialect
  # MLIRAffineToStandard
  )

add_subdirectory(TaskPipeline)
add_subdirectory(Loop)
add_subdirectory(Kernel)
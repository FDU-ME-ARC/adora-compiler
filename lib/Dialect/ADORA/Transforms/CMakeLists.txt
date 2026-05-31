add_mlir_dialect_library(MLIRADORATransforms
  AssignStreams.cpp
  BufferReuse.cpp
  DependencyAnalysis.cpp
  DFGgenPass.cpp
  LLMPipelineSchedule.cpp
  LowerAsyncTokens.cpp
  ScheduleAdoraTasks.cpp
  SimplifyLoadStore.cpp
  ThreadLoopCarriedTokensImpl.cpp

  TaskGraph/TaskNode.cpp
  TaskGraph/TaskGraph.cpp

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
  MLIRADORAAnalysis
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRADORATransformsUtility

  MLIRTransforms
  # MLIRPass
  # MLIRSCFDialect
  # MLIRAffineToStandard
  )

add_subdirectory(Loop)
add_subdirectory(Kernel)
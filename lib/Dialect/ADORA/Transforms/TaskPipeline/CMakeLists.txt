add_mlir_dialect_library(MLIRADORATaskPipeline
  ScheduleAdoraTasks.cpp
  AssignStreams.cpp
  LowerAsyncTokens.cpp
  ThreadLoopCarriedTokensImpl.cpp
  LLMPipelineSchedule.cpp

  TaskGraph/TaskNode.cpp
  TaskGraph/TaskGraph.cpp

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
  GeneralCDFGGen
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRADORATransformsUtility
  MLIRADORAAnalysis
  )

set(LIBS
  GeneralCDFGGen
)

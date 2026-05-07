# cmake_minimum_required(VERSION 3.0.0)
# set(PROJECT_NAME cgra-compiler)
# project(${PROJECT_NAME} VERSION 0.1.0)

# set(CMAKE_CXX_STANDARD 11)
# set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_BUILD_TYPE "Debug")
set(SPDLOG_NO_EXCEPTIONS ON)
# file(GLOB_RECURSE SRCS ${CMAKE_SOURCE_DIR}/src/*.cpp)
# add_executable(${PROJECT_NAME} ${CMAKE_SOURCE_DIR}/src/main.cpp ${SRCS})
# target_include_directories(CGRAMapperLIB PUBLIC ${CMAKE_SOURCE_DIR}/include)

add_library(CGRAMapperLIB
  src/adg/adg_node.cpp
  src/adg/adg.cpp

  src/dfg/dfg_node.cpp
  src/dfg/dfg.cpp

  src/ir/dfg_ir.cpp
  src/ir/adg_ir.cpp

  src/graph/graph_node.cpp
  src/graph/graph.cpp

  src/mapper/configuration/configuration.cpp
  src/mapper/configuration/pingpongCfg.cpp

  src/mapper/io_scheduler.cpp
  src/mapper/agent_trace.cpp
  src/mapper/online_ranker.cpp
  src/mapper/pipeline_scheduler.cpp
  src/mapper/mapper_sa.cpp
  src/mapper/mapper.cpp
  src/mapper/mapping.cpp
  src/mapper/visualize.cpp

  src/op/operations.cpp
  )

target_link_libraries(CGRAMapperLIB
  CGRAEmit
  ADORAMapTensorOp
)

add_dependencies(CGRAMapperLIB 
  GeneralCDFGGen)

add_subdirectory(src/emit)
add_subdirectory(src/tensorop)
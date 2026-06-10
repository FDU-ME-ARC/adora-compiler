# add_subdirectory(inc)
# add_subdirectory(src)


add_library(GeneralCDFGGen
  src/mlir_cdfg.cpp
  src/mlir_cdfg_node.cpp
  src/llvm_cdfg.cpp
  src/llvm_cdfg_node.cpp
  src/llvm_cdfg_edge.cpp

  DFGgen.cpp
  )


# set(LLVM_INCLUDE_DIRS "/home/tianyi/MLIRCGRA/llvm-project/build/include")
# set(LLVM_LIBRARY_DIRS "/home/tianyi/MLIRCGRA/llvm-project/build/lib")

target_include_directories(GeneralCDFGGen
  PRIVATE ${PROJ_INCLUDE_DIR}/lib/DFG/inc
  PRIVATE ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(GeneralCDFGGen
  PUBLIC MLIRIR
  PUBLIC MLIRPass
  PUBLIC MLIRTransforms
  PUBLIC MLIRSCFDialect
  PUBLIC MLIRAffineDialect
  PUBLIC MLIRADORATransformsUtility
  PRIVATE ${LLVM_LIBRARY_DIRS}
)

add_dependencies(GeneralCDFGGen ADORAPassIncGen MLIRAffineDialect)
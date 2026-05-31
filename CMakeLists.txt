cmake_minimum_required(VERSION 3.20.0)
project(ADORA VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_BUILD_WITH_INSTALL_NAME_DIR ON)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})
set(RAAA_LIB_DIR ${CMAKE_BINARY_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)
include(MLIRDetectPythonEnv)
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

if(MLIR_ENABLE_BINDINGS_PYTHON)
  include(MLIRDetectPythonEnv)
  mlir_configure_python_dev_packages()
endif()

set(LLVM_INCLUDE_DIRS "${LLVM_BUILD_DIR}/include")
set(LLVM_LIBRARY_DIRS "${LLVM_INSTALL_DIR}/lib")
message(LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS})
message(LLVM_LIBRARY_DIRS: ${LLVM_LIBRARY_DIRS})

link_directories(${LLVM_LIBRARY_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/mapper/include)
include_directories(${PROJECT_BINARY_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/lib/DFG/inc)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

add_subdirectory(include/ADORA)
add_subdirectory(lib)
add_subdirectory(mapper)
add_subdirectory(runtime) # PR3: adora_async_rt shared/static library

add_subdirectory(tools)
add_subdirectory(test)
# add_subdirectory(cmake)

##############################
### following is for install libs and cmakeconfig
##############################
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}"
      CACHE PATH "Install prefix" FORCE)
endif()

install(
  TARGETS
    MLIRADORAOps MLIRADORAKernelOp 
    MLIRADORAAnalysis MLIRADORALowerings MLIRADORATransforms
    MLIRADORATransformsUtility MLIRADORALoopTransforms
    MLIRADORATaskPipeline
    MLIRADORAKernelExtract GeneralCDFGGen
    ADORATensorInterface ADORATensorOps ADORATensorLower
    ADORATensorTransforms ADORATensorUtility ADORATensorOpGemmLower ADORATensorOpConvLower
  EXPORT ADORATargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}       
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}      
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}       
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}  
)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
    PATTERN "*.h"  PATTERN "*.hh"  PATTERN "*.hpp"  PATTERN "*.h.inc"
    PATTERN "CMakeLists.txt" EXCLUDE
    PATTERN "*.td"           EXCLUDE
    PATTERN "*.td.inc"       EXCLUDE
    PATTERN "*.cmake"        EXCLUDE
    PATTERN "*.in"           EXCLUDE
)

install(DIRECTORY ${PROJECT_BINARY_DIR}/include/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.inc"
)

### generate ADORATargets.cmake
set(ADORA_INSTALL_CMAKEDIR ${CMAKE_INSTALL_LIBDIR}/cmake/ADORA)

install(EXPORT ADORATargets
  FILE ADORATargets.cmake
  DESTINATION ${ADORA_INSTALL_CMAKEDIR}
)

# ADORAConfig.cmake / ADORAConfigVersion.cmake
configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/lib/cmake/ADORA/ADORAConfig.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/ADORA/ADORAConfig.cmake"
  INSTALL_DESTINATION "${ADORA_INSTALL_CMAKEDIR}"
  NO_SET_AND_CHECK_MACRO
  NO_CHECK_REQUIRED_COMPONENTS_MACRO
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/ADORA/ADORAConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY SameMajorVersion
)

# 3) Install Config/Version into the same directory as Targets
install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/ADORA/ADORAConfig.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/ADORA/ADORAConfigVersion.cmake"
  DESTINATION "${ADORA_INSTALL_CMAKEDIR}"
)

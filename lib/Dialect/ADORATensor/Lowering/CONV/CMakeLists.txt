# lib/Dialect/ADORATensor/Lowering/CONV/CMakeLists.txt

set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORATensorOpConvLower
  DirectConv.cpp
  Im2ColConv.cpp
)

target_link_libraries(ADORATensorOpConvLower
  ADORATensorOps
  ADORATensorInterface
  MLIRADORAKernelExtract
  MLIRADORATransformsUtility
  ADORATensorOpGemmLower 
)
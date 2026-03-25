set(SPDLOG_NO_EXCEPTIONS ON)

add_library(ADORAMapTensorOp
  tensorop.cpp 
  
  mapGemm.cpp
  mapConv.cpp
)

target_link_libraries(ADORAMapTensorOp
  ADORATensorOps
  ADORATensorInterface
  ADORATensorOpGemmLower
  ADORATensorOpConvLower
)
# add_subdirectory(Gemm)
set(SPDLOG_NO_EXCEPTIONS ON)

add_library(CGRAEmit
  EmitCGRACall.cpp 
  EmitMixedJsonPytest.cpp
  EmitVitisSDK.cpp 
  EmitPytest.cpp 
  EmitUtility.cpp
)

add_dependencies(CGRAEmit MLIRADORAOps)

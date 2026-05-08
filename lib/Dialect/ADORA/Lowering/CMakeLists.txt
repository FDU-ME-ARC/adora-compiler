add_mlir_conversion_library(MLIRADORALowerings
  KernelCallToLLVM.cpp
  MathRewrite.cpp
  ADORAToSCF.cpp
  AsyncRuntimeToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/Lowering
  ${PROJ_INCLUDE_DIR}/ADORA/Dialect/ADORA/IR

  DEPENDS
  ADORALowerPassIncGen
  MLIRADORAOps

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRAnalysis
  MLIRFuncDialect
  MLIRLLVMCommonConversion
  MLIRLLVMDialect
  MLIRTransformUtils
  )

# This includes several libs that extends the linking time
# only the necessary libs are included instead
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

# To solve error: exception handling disabled, use '-fexceptions' to enable
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DSPDLOG_NO_EXCEPTIONS ")

set(LIBS
  # ${dialect_libs}
  # ${conversion_libs}

  # Core
  MLIRIR
  MLIRTransforms
  MLIROptLib
  
  # Dialects
  MLIRLLVMDialect
  MLIRLinalgDialect
  MLIRMemRefDialect
  MLIRAffineDialect
  MLIRArithDialect
  MLIRMathDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRMLProgramDialect
  MLIRTensorDialect
  MLIRDLTIDialect

 
  # Transforms
  MLIRReconcileUnrealizedCasts

  # Test Passes

  
  # Conversion

  #ADORA
  MLIRADORAOps
  GeneralCDFGGen
  # MLIRADORAMisc
  MLIRADORATransforms
  # MLIRADORALowerings

  #ADORATensor
  ADORATensorOps

  #Mapper
  CGRAMapperLIB
)

set(SOURCES
  cgra-mapper.cpp
)

add_llvm_executable(cgra-mapper ${SOURCES})

# llvm_update_compile_flags(cgra-mapper)
target_link_libraries(cgra-mapper PRIVATE ${LIBS})
llvm_update_compile_flags(cgra-mapper)

mlir_check_link_libraries(cgra-mapper)

install(TARGETS cgra-mapper)

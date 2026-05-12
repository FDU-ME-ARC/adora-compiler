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
  # Listed explicitly so --start-group/--end-group below encloses both the
  # caller (mapGemm.cpp / mapConv.cpp in ADORAMapTensorOp) and the definer
  # (pipeline_scheduler.cpp in CGRAMapperLIB) of mlir::ADORA::schedulePipeline.
  # Otherwise CMake appends ADORAMapTensorOp as a PUBLIC transitive dep of
  # CGRAMapperLIB *after* --end-group and GNU ld misses the back-edge.
  ADORAMapTensorOp
)

set(SOURCES
  cgra-mapper.cpp
)

add_llvm_executable(cgra-mapper ${SOURCES})

# llvm_update_compile_flags(cgra-mapper)
# NOTE: CGRAMapperLIB and ADORAMapTensorOp have a circular reference:
#   - ADORAMapTensorOp (mapper/src/tensorop/mapGemm.cpp, mapConv.cpp) calls
#     mlir::ADORA::schedulePipeline
#   - schedulePipeline is defined in CGRAMapperLIB (mapper/src/mapper/pipeline_scheduler.cpp)
#   - CGRAMapperLIB in turn link-depends on ADORAMapTensorOp
# Single-pass left-to-right static-archive resolution therefore leaves
# schedulePipeline undefined. Wrap the static libs in --start-group/--end-group
# so ld rescans until all references are resolved. Apple ld64 links static
# archives multi-pass by default, so only add the flags for GNU-like linkers.
if(NOT APPLE AND NOT MSVC)
  target_link_libraries(cgra-mapper PRIVATE
    -Wl,--start-group
    ${LIBS}
    -Wl,--end-group
  )
else()
  target_link_libraries(cgra-mapper PRIVATE ${LIBS})
endif()
llvm_update_compile_flags(cgra-mapper)

mlir_check_link_libraries(cgra-mapper)



install(TARGETS cgra-mapper)
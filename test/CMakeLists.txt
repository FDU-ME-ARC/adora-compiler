set(ADORA_TEST_DEPENDS
  cgra-opt
  tensor-opt
  cgra-mapper
  FileCheck
)

set(ADORA_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(ADORA_TEST_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(ADORA_TOOLS_DIR ${LLVM_RUNTIME_OUTPUT_INTDIR})

set(ADORA_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(ADORA_BINARY_DIR ${CMAKE_BINARY_DIR})

# where your built tool is
set(ADORA_TOOLS_DIR ${CMAKE_BINARY_DIR}/bin)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

add_lit_testsuite(check-adora "Running the ADORA regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS cgra-opt FileCheck
  DEPENDS tensor-opt FileCheck
  DEPENDS cgra-mapper FileCheck
)


add_lit_testsuites(ADORA ${ADORA_TEST_DIR}
  DEPENDS ${ADORA_TEST_DEPENDS}
)

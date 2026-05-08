# PR3 — ADORA async-event runtime stub (host reference implementation).
# Link against this shared library when running mlir-cpu-runner integration
# tests that exercise the adora-lower-async-tokens + adora-to-llvm-async-runtime
# pass pipeline.

add_library(adora_async_rt SHARED adora_async_rt.c)
target_include_directories(adora_async_rt PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adora_async_rt PRIVATE pthread)

# Also expose a static variant for embedded / bare-metal targets.
add_library(adora_async_rt_static STATIC adora_async_rt.c)
target_include_directories(adora_async_rt_static PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adora_async_rt_static PRIVATE pthread)

add_executable(mllm-bench-i main.cpp)
target_link_libraries(mllm-bench-i PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-bench-i PRIVATE ${MLLM_INCLUDE_DIR})

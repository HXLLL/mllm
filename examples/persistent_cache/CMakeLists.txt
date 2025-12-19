add_executable(mllm-persistent-cache main.cpp)
target_link_libraries(mllm-persistent-cache PRIVATE MllmRT MllmCPUBackend)
target_include_directories(mllm-persistent-cache PRIVATE ${MLLM_INCLUDE_DIR})


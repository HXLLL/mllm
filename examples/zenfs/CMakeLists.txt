add_executable(zenfs main.cpp)

target_link_libraries(zenfs PRIVATE
  MllmRT 
  MllmCPUBackend
)

target_include_directories(zenfs PRIVATE
  ${MLLM_INCLUDE_DIR}
)


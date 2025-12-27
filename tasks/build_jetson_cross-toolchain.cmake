set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   "/home/xiaolonghuang/x-tools/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER   "/home/xiaolonghuang/x-tools/aarch64-unknown-linux-gnu/bin/aarch64-unknown-linux-gnu-g++")

set(CMAKE_SYSROOT "/home/xiaolonghuang/toolchains/l4t-gcc/aarch64-buildroot-linux-gnu/sysroot")
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

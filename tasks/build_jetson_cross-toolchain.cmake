set(CMAKE_BUILD_RPATH "\$ORIGIN")

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(CMAKE_C_COMPILER   "/home/xiaolonghuang/x-tools/aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc")
set(CMAKE_CXX_COMPILER   "/home/xiaolonghuang/x-tools/aarch64-linux-gnu/bin/aarch64-linux-gnu-g++")

set(CMAKE_SYSROOT "/home/xiaolonghuang/jetroot")
set(CMAKE_SYSROOT_LINK "/home/xiaolonghuang/jetroot")

set(aa_lib_path "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-gnu")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -B${aa_lib_path} -Wl,-rpath-link,${aa_lib_path}  -I${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -B${aa_lib_path}  -Wl,-rpath-link,${aa_lib_path} -I${CMAKE_SYSROOT}/usr/include/aarch64-linux-gnu")

set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

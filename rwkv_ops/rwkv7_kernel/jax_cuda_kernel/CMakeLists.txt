cmake_minimum_required(VERSION 3.18)
project(wkv7 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 1. 找到 Python 解释器
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# 2. 让 Python 告诉你 XLA 头文件在哪
execute_process(
  COMMAND "${Python3_EXECUTABLE}" "-c" "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot get XLA include dir from jax.ffi")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# 3. 产出共享库
add_library(wkv7 SHARED wkv7_ffi.cu wkv7_ffi.cc)
target_include_directories(wkv7 PRIVATE ${XLA_INCLUDE_DIR})
target_link_libraries(wkv7 PRIVATE CUDA::cudart)
set_target_properties(wkv7 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    PREFIX "")
install(TARGETS wkv7 DESTINATION .)
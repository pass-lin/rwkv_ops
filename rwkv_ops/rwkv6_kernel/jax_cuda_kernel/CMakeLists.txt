cmake_minimum_required(VERSION 3.18)
project(wkv6 LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot get XLA include dir from jax.ffi")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# RWKV-6 CUDA kernel 在编译期固定 head size (_N_) 与最大序列长度 (_T_)。
if(NOT DEFINED HEAD_SIZE)
  set(HEAD_SIZE 64)
endif()
if(NOT DEFINED MAX_SEQUENCE_LENGTH)
  set(MAX_SEQUENCE_LENGTH 4096)
endif()
message(STATUS "HEAD_SIZE = ${HEAD_SIZE}")
message(STATUS "MAX_SEQUENCE_LENGTH = ${MAX_SEQUENCE_LENGTH}")

add_library(wkv6 SHARED wkv6_ffi.cu)

target_include_directories(wkv6 PRIVATE ${XLA_INCLUDE_DIR})
target_link_libraries(wkv6 PRIVATE CUDA::cudart)
target_compile_features(wkv6 PUBLIC cxx_std_17)
set_target_properties(wkv6 PROPERTIES
    CUDA_STANDARD              17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
    PREFIX                     ""
)

target_compile_definitions(wkv6 PRIVATE _N_=${HEAD_SIZE} _T_=${MAX_SEQUENCE_LENGTH})

# 将 .so 直接安装到源码目录，与 wkv6_jax.py 同级，方便 ctypes.CDLL 加载。
install(TARGETS wkv6
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")

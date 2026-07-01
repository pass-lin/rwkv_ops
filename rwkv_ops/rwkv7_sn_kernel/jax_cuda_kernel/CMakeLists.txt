cmake_minimum_required(VERSION 3.18)
project(wkv7_sn LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---------- 1. 找到 Python ----------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------- 2. 取 XLA 头文件路径 ----------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot get XLA include dir from jax.ffi")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# ---------- 3. 生成共享库 ----------
add_library(wkv7_sn SHARED wkv7_sn_ffi.cu)

# 3-1. 头文件搜索路径
target_include_directories(wkv7_sn PRIVATE ${XLA_INCLUDE_DIR})

# 3-2. 链接 CUDA 运行时
target_link_libraries(wkv7_sn PRIVATE CUDA::cudart)

# 3-3. 关键：C++17 / CUDA17 标准
target_compile_features(wkv7_sn PUBLIC cxx_std_17)
set_target_properties(wkv7_sn PROPERTIES
    CUDA_STANDARD          17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX                 ""        # 去掉默认的 "lib" 前缀
)

# ---------- 4. 安装 ----------
install(TARGETS wkv7_sn
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")

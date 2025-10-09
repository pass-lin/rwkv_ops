cmake_minimum_required(VERSION 3.18)
project(wkv7 LANGUAGES CXX CUDA)

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
add_library(wkv7 SHARED wkv7_ffi.cu)

# 3-1. 头文件搜索路径
target_include_directories(wkv7 PRIVATE ${XLA_INCLUDE_DIR})

# 3-2. 链接 CUDA 运行时
target_link_libraries(wkv7 PRIVATE CUDA::cudart)

# 3-3. 关键：C++17 / CUDA17 标准
target_compile_features(wkv7 PUBLIC cxx_std_17)
set_target_properties(wkv7 PROPERTIES
    CUDA_STANDARD          17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX                 ""        # 去掉默认的 "lib" 前缀
)

# ---------- 4. 安装 ----------
# 把 .so 直接装到源码目录（与 wkv7_jax.py 同一级），方便 ctypes.CDLL 加载
install(TARGETS wkv7
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")   # Windows 用 RUNTIME
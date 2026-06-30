cmake_minimum_required(VERSION 3.18)
project(wkv6 LANGUAGES CXX CUDA)

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

# ---------- 3. 编译期特化参数 ----------
# RWKV-6 CUDA kernel 在编译期固定 head size (_N_) 与最大序列长度 (_T_)。
# Python 侧会在调用 CMake 时传入对应的值。
if(NOT DEFINED HEAD_SIZE)
  set(HEAD_SIZE 64)
endif()
if(NOT DEFINED MAX_SEQUENCE_LENGTH)
  set(MAX_SEQUENCE_LENGTH 4096)
endif()
message(STATUS "HEAD_SIZE = ${HEAD_SIZE}")
message(STATUS "MAX_SEQUENCE_LENGTH = ${MAX_SEQUENCE_LENGTH}")

# ---------- 4. 生成共享库 ----------
add_library(wkv6 SHARED wkv6_ffi.cu)

# 4-1. 头文件搜索路径
target_include_directories(wkv6 PRIVATE ${XLA_INCLUDE_DIR})

# 4-2. 链接 CUDA 运行时
target_link_libraries(wkv6 PRIVATE CUDA::cudart)

# 4-3. 关键：C++17 / CUDA17 标准
target_compile_features(wkv6 PUBLIC cxx_std_17)
set_target_properties(wkv6 PROPERTIES
    CUDA_STANDARD              17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE  ON
    PREFIX                     ""        # 去掉默认的 "lib" 前缀
)

# 4-4. 编译期宏
target_compile_definitions(wkv6 PRIVATE _N_=${HEAD_SIZE} _T_=${MAX_SEQUENCE_LENGTH})

# ---------- 5. 安装 ----------
# 把 .so 直接装到源码目录（与 wkv6_jax.py 同一级），方便 ctypes.CDLL 加载
install(TARGETS wkv6
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")   # Windows 用 RUNTIME

cmake_minimum_required(VERSION 3.18)
project(wkv7_single_step LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# ---------- 1. 找到 Python ----------
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# ---------- 2. 取 XLA 头文件路径 ----------
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import jax; print(jax.ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot get XLA include dir from jax.ffi")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# ---------- 3. 生成共享库 ----------
add_library(wkv7_single_step SHARED wkv7_single_step_ffi.cu)

# 3-1. 头文件搜索路径
target_include_directories(wkv7_single_step PRIVATE ${XLA_INCLUDE_DIR})

# 3-2. 链接 CUDA 运行时
target_link_libraries(wkv7_single_step PRIVATE CUDA::cudart)

# 3-3. 关键：C++17 / CUDA17 标准
target_compile_features(wkv7_single_step PUBLIC cxx_std_17)
set_target_properties(wkv7_single_step PROPERTIES
    CUDA_STANDARD          17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX                 ""        # 去掉默认的 "lib" 前缀
)

# ---------- 4. 安装 ----------
# 把 .so 直接装到源码目录，方便 ctypes.CDLL 加载
install(TARGETS wkv7_single_step
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")   # Windows 用 RUNTIME
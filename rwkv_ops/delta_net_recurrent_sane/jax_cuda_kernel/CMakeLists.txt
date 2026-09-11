cmake_minimum_required(VERSION 3.18)
project(delta_net_recurrent_sane LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

# 找到 Python
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# 取 XLA 头文件路径
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "Cannot get XLA include dir from jax.ffi")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# 生成共享库
add_library(delta_net_recurrent_sane SHARED delta_net_recurrent_sane_ffi.cu)

# 头文件搜索路径
target_include_directories(delta_net_recurrent_sane PRIVATE ${XLA_INCLUDE_DIR})

# 链接 CUDA 运行时
target_link_libraries(delta_net_recurrent_sane PRIVATE CUDA::cudart)

# 关键：C++17 / CUDA17 标准
target_compile_features(delta_net_recurrent_sane PUBLIC cxx_std_17)
set_target_properties(delta_net_recurrent_sane PROPERTIES
    CUDA_STANDARD          17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX                 ""        # 去掉默认的 "lib" 前缀
)

# 安装
# 把 .so 直接装到源码目录（与 delta_net_recurrent_sane_jax.py 同一级），方便 ctypes.CDLL 加载
install(TARGETS delta_net_recurrent_sane
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")   # Windows 用 RUNTIME

cmake_minimum_required(VERSION 3.18)
project(mhu_jax LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)

# 获取XLA头文件路径
execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "from jax import ffi; print(ffi.include_dir())"
  OUTPUT_VARIABLE XLA_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NOT XLA_INCLUDE_DIR)
  message(FATAL_ERROR "无法从jax.ffi获取XLA头文件路径，请确保JAX版本>=0.4.31")
endif()
message(STATUS "XLA include directory: ${XLA_INCLUDE_DIR}")

# 设置公共头文件路径
set(COMMON_KERNEL_DIR "${CMAKE_SOURCE_DIR}/../common_kernel")

# 生成共享库（库名改为mhu_ffi避免冲突）
add_library(mhu_ffi SHARED mhu_ffi.cu)

# 包含路径
target_include_directories(mhu_ffi PRIVATE 
    ${XLA_INCLUDE_DIR}
    ${COMMON_KERNEL_DIR}/include
    ${COMMON_KERNEL_DIR}/kernels
)

# 链接CUDA运行时
target_link_libraries(mhu_ffi PRIVATE CUDA::cudart)

# 编译选项
target_compile_features(mhu_ffi PUBLIC cxx_std_17)
set_target_properties(mhu_ffi PROPERTIES
    CUDA_STANDARD 17
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""  # 移除lib前缀
    OUTPUT_NAME "mhu"  # 输出文件名仍为mhu.so
)

# 安装到源码目录（关键：与Python查找路径一致）
install(TARGETS mhu_ffi
        LIBRARY DESTINATION "${CMAKE_SOURCE_DIR}"
        RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}")
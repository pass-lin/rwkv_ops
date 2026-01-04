#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include "../include/mhc_types.h"

namespace mhc {

template<int BLOCK_SIZE>
// [修改]: size 参数改为 int64_t
__global__ void float_to_bf16_kernel(floatX* __restrict__ out, const float* __restrict__ inp, int64_t size) {
    // [修改]: idx 改为 int64_t，并强制转换 blockIdx.x 避免 32 位乘法溢出
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = to_bf(inp[idx]); // 使用新定义的工具
    }
}

template<int BLOCK_SIZE>
// [修改]: size 参数改为 int64_t
__global__ void bf16_to_float_kernel(float* __restrict__ out, const floatX* __restrict__ inp, int64_t size) {
    // [修改]: idx 改为 int64_t，并强制转换 blockIdx.x
    int64_t idx = (int64_t)blockIdx.x * BLOCK_SIZE + threadIdx.x;
    if (idx < size) {
        out[idx] = to_float(inp[idx]); // 使用新定义的工具
    }
}

// [修改]: size 参数改为 int64_t
inline void float_to_bf16(floatX* out, const float* inp, int64_t size, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    // num_blocks 本身通常不会溢出 int (除非 size > 5000亿)，但计算过程需用 64 位
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    float_to_bf16_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

// [修改]: size 参数改为 int64_t
inline void bf16_to_float(float* out, const floatX* inp, int64_t size, cudaStream_t stream = nullptr) {
    constexpr int BLOCK_SIZE = 256;
    int num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    bf16_to_float_kernel<BLOCK_SIZE><<<num_blocks, BLOCK_SIZE, 0, stream>>>(out, inp, size);
}

__device__ __forceinline__ float fast_exp(float x) {
    return __expf(x);
}

__device__ __forceinline__ float fast_sigmoid(float x) {
    return __frcp_rn(1.0f + __expf(-x));
}

} // namespace mhc
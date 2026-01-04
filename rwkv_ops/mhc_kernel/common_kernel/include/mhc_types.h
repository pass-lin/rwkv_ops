#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cublasLt.h>
#include <assert.h>
#include <cstdint>

namespace mhc {

using floatX = __nv_bfloat16;
using floatN = float;

// 定义统一的转换工具，供所有 .cuh 和 .cu 使用
__device__ inline float to_float(const floatX& u) {
    return __bfloat162float(u);
}

__device__ inline floatX to_bf(const float& u) {
    #if __CUDA_ARCH__ >= 800
        return __float2bfloat16(u);
    #else
        // 兼容旧架构或强制舍入
        return __float2bfloat16_rn(u);
    #endif
}


struct MHCConfig {
    int sinkhorn_iters;
    int nC;
    float eps;
    bool use_pdl;
};

struct RMSNormParams {
    int n;
    float eps;
};

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

inline void check_cublas(cublasStatus_t status, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cuBLAS error at %s:%d: %d\n", file, line, (int)status);
        exit(EXIT_FAILURE);
    }
}
// 错误检查宏
#define CHECK_CUDA(call) \
do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    } \
} while (0)

#define CHECK_CUDA(call) mhc::check_cuda((call), __FILE__, __LINE__)
#define CHECK_CUBLAS(call) mhc::check_cublas((call), __FILE__, __LINE__)
} // namespace mhc

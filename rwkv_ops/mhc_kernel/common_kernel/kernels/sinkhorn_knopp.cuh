#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include "../include/mhc_types.h"
#include "type_conversions.cuh"

namespace mhc {

/**
 * 1. 前向传播 Kernel (保持不变，但确保 eps 使用一致)
 */
template<int BLOCK_SIZE>
__global__ void sinkhorn_knopp_fwd_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int M, int N, int num_iters, float eps) {
    
    extern __shared__ float smem[];
    float* tile = smem; 
    float* row_sums = smem + M * N; 
    float* col_sums = row_sums + M;

    int tid = threadIdx.x;
    int64_t total = (int64_t)M * N;

    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        tile[i] = __expf(inp[i]);
    }
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        // 行归一化
        for (int r = tid; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) sum += tile[(int64_t)r * N + c];
            row_sums[r] = 1.0f / (sum + eps); 
        }
        __syncthreads();

        for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
            tile[i] *= row_sums[i / N];
        }
        __syncthreads();

        // 列归一化
        for (int c = tid; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) sum += tile[(int64_t)r * N + c];
            col_sums[c] = 1.0f / (sum + eps);
        }
        __syncthreads();

        for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
            tile[i] *= col_sums[i % N];
        }
        __syncthreads();
    }

    for (int64_t i = tid; i < total; i += BLOCK_SIZE) out[i] = tile[i];
}

/**
 * 2. 反向传播 Kernel (修正：匹配自动微分迭代)
 */
template<int BLOCK_SIZE>
__global__ void sinkhorn_knopp_bwd_kernel(
    float* __restrict__ d_inp,
    const float* __restrict__ grad,
    const float* __restrict__ out_fwd,
    int M, int N, int num_iters, float eps) {
    
    extern __shared__ float smem[];
    float* P = smem;               // 前向输出 P [M*N]
    float* dP = smem + M * N;      // 传入梯度 G [M*N]
    float* alpha = smem + 2 * M * N; // 辅助变量 alpha [M]
    float* beta = alpha + M;       // 辅助变量 beta [N]

    int tid = threadIdx.x;
    int64_t total = (int64_t)M * N;

    // 加载数据
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        P[i] = out_fwd[i];
        dP[i] = grad[i];
    }
    __syncthreads();

    // 核心：Sinkhorn 梯度的迭代解法 (对应 PyTorch 的循环展开)
    // 我们需要求解 alpha 和 beta 满足：
    // alpha_i = sum_j (P_ij * (dP_ij - beta_j))
    // beta_j = sum_i (P_ij * (dP_ij - alpha_i))
    
    // 初始化 alpha, beta 为 0
    for(int i = tid; i < M; i += BLOCK_SIZE) alpha[i] = 0.0f;
    for(int j = tid; j < N; j += BLOCK_SIZE) beta[j] = 0.0f;
    __syncthreads();

    for (int iter = 0; iter < num_iters; iter++) {
        // 更新 beta
        for (int j = tid; j < N; j += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int i = 0; i < M; i++) {
                sum += P[(int64_t)i * N + j] * (dP[(int64_t)i * N + j] - alpha[i]);
            }
            beta[j] = sum;
        }
        __syncthreads();

        // 更新 alpha
        for (int i = tid; i < M; i += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int j = 0; j < N; j++) {
                sum += P[(int64_t)i * N + j] * (dP[(int64_t)i * N + j] - beta[j]);
            }
            alpha[i] = sum;
        }
        __syncthreads();
    }

    // 最终梯度公式: dL/dA = P * (G - alpha - beta)
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        int64_t r = i / N;
        int64_t c = i % N;
        d_inp[i] = P[i] * (dP[i] - alpha[r] - beta[c]);
    }
}

/* -------------------- API 接口 -------------------- */

inline void sinkhorn_knopp_forward(
    float* out, const float* inp, 
    int M, int N, int num_iters, float eps, 
    cudaStream_t stream = nullptr) {
    
    const int BLOCK_SIZE = 256;
    size_t smem_size = ((size_t)M * N + M + N) * sizeof(float);
    sinkhorn_knopp_fwd_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
        out, inp, M, N, num_iters, eps
    );
}

inline void sinkhorn_knopp_backward(
    float* d_inp, const float* grad, const float* M_out, const float* M_inp, 
    int N, int num_iters, float eps, 
    cudaStream_t stream = nullptr) {
    
    const int BLOCK_SIZE = 256;
    // 显存占用: P(N*N) + dP(N*N) + alpha(N) + beta(N)
    size_t smem_size = (2 * (size_t)N * N + 2 * N) * sizeof(float);

    sinkhorn_knopp_bwd_kernel<BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
        d_inp, grad, M_out, N, N, num_iters, eps
    );
}

} // namespace mhc
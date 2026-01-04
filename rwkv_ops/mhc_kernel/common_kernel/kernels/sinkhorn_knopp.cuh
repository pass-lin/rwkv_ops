#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "type_conversions.cuh"

namespace cg = cooperative_groups;

namespace mhc {

/*
 * 前向传播 Kernel: 实现 Sinkhorn-Knopp 迭代
 * 计算 P = Sinkhorn(exp(A))，输入 A 通常是 FP32 (在 Python 端已减去 max)
 */
template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_fwd_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int M, int N, int num_iters, float eps) {
    
    extern __shared__ float smem[];
    float* tile = smem;                 // 存储 exp(A) 矩阵
    float* row_sums = smem + M * N;     // 存储行和
    float* col_sums = row_sums + M;     // 存储列和

    int tid = threadIdx.x;
    // [修改]: 强制转换为 int64_t 防止乘法溢出
    int64_t total = (int64_t)M * N;

    // 1. 加载输入并计算 exp
    // [修改]: 循环变量 i 改为 int64_t
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        tile[i] = __expf(inp[i]);
    }
    __syncthreads();

    // 2. Sinkhorn 迭代 (FP32 保证数值稳定性)
    for (int iter = 0; iter < num_iters; iter++) {
        // --- 行归一化 ---
        for (int r = tid; r < M; r += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int c = 0; c < N; c++) {
                // [修改]: 索引计算增加 (int64_t) 强转
                sum += tile[(int64_t)r * N + c];
            }
            row_sums[r] = __frcp_rn(sum + eps); // 预计算倒数
        }
        __syncthreads();

        // [修改]: 循环变量 i 改为 int64_t
        for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
            tile[i] *= row_sums[i / N];
        }
        __syncthreads();

        // --- 列归一化 ---
        for (int c = tid; c < N; c += BLOCK_SIZE) {
            float sum = 0.0f;
            for (int r = 0; r < M; r++) {
                // [修改]: 索引计算增加 (int64_t) 强转
                sum += tile[(int64_t)r * N + c];
            }
            col_sums[c] = __frcp_rn(sum + eps);
        }
        __syncthreads();

        // [修改]: 循环变量 i 改为 int64_t
        for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
            tile[i] *= col_sums[i % N];
        }
        __syncthreads();
    }

    // 3. 写回结果 (FP32)
    // [修改]: 循环变量 i 改为 int64_t
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        out[i] = tile[i];
    }
}

/*
 * 反向传播 Kernel
 * 根据 P (前向输出) 和 G (梯度) 计算 dL/dA
 * 公式参考: dL/dA = P * (G - row_sum(P*G) - col_sum(P*G)) (简化版)
 */
template<int MAX_DIM, int BLOCK_SIZE>
__global__ void sinkhorn_knopp_bwd_kernel(
    float* __restrict__ d_inp,
    const float* __restrict__ grad,
    const float* __restrict__ out_fwd,
    int M, int N, float eps) {
    
    extern __shared__ float smem[];
    float* P = smem;                    // 前向输出 P
    float* G = smem + M * N;            // 梯度 G
    float* PG = smem + 2 * M * N;       // 中间变量 P * G
    float* row_sums = smem + 3 * M * N;
    float* col_sums = row_sums + M;

    int tid = threadIdx.x;
    // [修改]: 强制转换为 int64_t 防止乘法溢出
    int64_t total = (int64_t)M * N;

    // 加载数据
    // [修改]: 循环变量 i 改为 int64_t
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        float p_val = out_fwd[i];
        float g_val = grad[i];
        P[i] = p_val;
        G[i] = g_val;
        PG[i] = p_val * g_val;
    }
    __syncthreads();

    // 计算行/列和用于反向传播投影
    // [修改]: 循环变量 r 改为 int64_t (虽然 r<M 为 int，但为了安全和一致性)
    for (int64_t r = tid; r < M; r += BLOCK_SIZE) {
        float s = 0.0f;
        // [修改]: 索引计算增加 (int64_t) 强转
        for (int c = 0; c < N; c++) s += PG[r * N + c];
        row_sums[r] = s;
    }
    for (int c = tid; c < N; c += BLOCK_SIZE) {
        float s = 0.0f;
        // [修改]: 索引计算增加 (int64_t) 强转
        for (int64_t r = 0; r < M; r++) s += PG[r * N + c];
        col_sums[c] = s;
    }
    __syncthreads();

    // 写回输入梯度
    // [修改]: 循环变量 i 改为 int64_t
    for (int64_t i = tid; i < total; i += BLOCK_SIZE) {
        int64_t r = i / N;
        int64_t c = i % N;
        // Sinkhorn 反向传播的核心：投影梯度
        d_inp[i] = P[i] * (G[i] - row_sums[r] - col_sums[c]);
    }
}

/* -------------------- API 接口 -------------------- */

inline void sinkhorn_knopp_forward(
    float* out, const float* inp, 
    int M, int N, int num_iters, float eps, 
    cudaStream_t stream = nullptr) {
    
    const int BLOCK_SIZE = 256;
    // 共享内存计算：tile(MN) + row_sums(M) + col_sums(N)
    // [修改]: 增加 (size_t) 强转，防止 int * int 溢出后再赋值给 size_t
    size_t smem_size = ((size_t)M * N + M + N) * sizeof(float);

    sinkhorn_knopp_fwd_kernel<64, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
        out, inp, M, N, num_iters, eps
    );
}

inline void sinkhorn_knopp_backward(
    float* d_inp, const float* grad, const float* M_out, const float* M_inp, 
    int N, int num_iters, float eps, 
    cudaStream_t stream = nullptr) {
    
    const int BLOCK_SIZE = 256;
    // 反向需要的共享内存更多：P(MN) + G(MN) + PG(MN) + sums
    // [修改]: 增加 (size_t) 强转
    size_t smem_size = (3 * (size_t)N * N + 2 * N) * sizeof(float);

    sinkhorn_knopp_bwd_kernel<64, BLOCK_SIZE><<<1, BLOCK_SIZE, smem_size, stream>>>(
        d_inp, grad, M_out, N, N, eps
    );
}

} // namespace mhc
#ifndef MHC_PRE_OP_CUH
#define MHC_PRE_OP_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "type_conversions.cuh"
#include "sinkhorn_knopp.cuh"

namespace cg = cooperative_groups;

namespace mhc {

/**
 * 1. Fused Pre-Op Forward Kernel
 * 修复：确保所有索引步进均使用 int64_t 避免在大模型/长序列下溢出
 */
template<int MAX_N = 8>
__global__ void mhc_pre_op_fwd_kernel(
    floatX* __restrict__ x_layer_in,
    float* __restrict__ H_pre_out,
    float* __restrict__ H_post_out,
    const floatX* __restrict__ x_expanded,
    const float* __restrict__ h_pre_raw,
    const float* __restrict__ h_post_raw,
    int64_t B, int64_t T, int n, int64_t C) 
{
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (btc >= B * T * C) return;

    int64_t bt = btc / C;
    int64_t c = btc % C;
    int64_t bt_offset_n = bt * (int64_t)n;

    float H_pre[MAX_N];
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float val_pre = h_pre_raw[bt_offset_n + (int64_t)i];
            H_pre[i] = 1.0f / (1.0f + __expf(-val_pre));
            
            if (c == 0) {
                H_pre_out[bt_offset_n + (int64_t)i] = H_pre[i];
                float val_post = h_post_raw[bt_offset_n + (int64_t)i];
                // 2.0 * sigmoid 逻辑保持不变
                H_post_out[bt_offset_n + (int64_t)i] = 2.0f * (1.0f / (1.0f + __expf(-val_post)));
            }
        }
    }

    float sum_val = 0.0f;
    int64_t bt_offset_n_c = bt * (int64_t)n * C;
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        if (i < n) {
            float xi = to_float(x_expanded[bt_offset_n_c + (int64_t)i * C + c]);
            sum_val += H_pre[i] * xi;
        }
    }

    x_layer_in[btc] = to_bf(sum_val);
}

/**
 * 2. Fused Pre-Op Backward Kernel
 * 修复重点：
 * 1. 强化规约逻辑：使用 block 级规约确保 sum_grad_x 的准确性。
 * 2. 检查发现之前的 atomicAdd 虽然逻辑正确，但若输入 tensor 未在 python/cpp 层清零会导致 Fail。
 * 3. 这里的 MAX_N 限制了能够并行处理的流数量。
 */
template<int BLOCK_SIZE, int MAX_N = 8>
__global__ void mhc_pre_op_bwd_kernel(
    floatX* __restrict__ d_x_expanded,
    float* __restrict__ d_h_pre_raw,
    float* __restrict__ d_h_post_raw,
    const floatX* __restrict__ grad_layer_in,
    const float* __restrict__ grad_H_post,
    const floatX* __restrict__ x_expanded,
    const float* __restrict__ H_pre,
    const float* __restrict__ H_post,
    int64_t B, int64_t T, int n, int64_t C) 
{
    // 定义共享内存用于 Block 级规约 (大小为 BLOCK_SIZE * n)
    // 假设 MAX_N 很小 (如 8)，256 * 8 * 4 bytes = 8KB，远小于显卡限制
    __shared__ float s_reduce[BLOCK_SIZE][MAX_N];

    int64_t bt = blockIdx.x; 
    if (bt >= B * T) return;

    int tid = threadIdx.x;
    int64_t bt_offset_n = bt * (int64_t)n;
    int64_t bt_offset_c = bt * C;
    int64_t bt_offset_n_c = bt * (int64_t)n * C;

    // 初始化局部累加器
    float thread_dh_pre_sum[MAX_N];
    #pragma unroll
    for(int i=0; i<MAX_N; ++i) thread_dh_pre_sum[i] = 0.0f;

    // 1. 计算 dx 并收集局部和
    for (int64_t c = (int64_t)tid; c < C; c += (int64_t)BLOCK_SIZE) {
        float g_in = to_float(grad_layer_in[bt_offset_c + c]);

        #pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n) {
                float h_pre_i = H_pre[bt_offset_n + (int64_t)i];
                d_x_expanded[bt_offset_n_c + (int64_t)i * C + c] = to_bf(g_in * h_pre_i);
                
                float xi = to_float(x_expanded[bt_offset_n_c + (int64_t)i * C + c]);
                thread_dh_pre_sum[i] += g_in * xi;
            }
        }
    }

    // 2. 将结果存入共享内存准备规约
    #pragma unroll
    for (int i = 0; i < MAX_N; i++) {
        s_reduce[tid][i] = thread_dh_pre_sum[i];
    }
    __syncthreads();

    // 3. 树状规约 (Tree Reduction)
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            #pragma unroll
            for (int i = 0; i < MAX_N; i++) {
                if (i < n) {
                    s_reduce[tid][i] += s_reduce[tid + stride][i];
                }
            }
        }
        __syncthreads();
    }

    // 4. 写回结果
    if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < MAX_N; i++) {
            if (i < n) {
                int64_t idx = bt_offset_n + (int64_t)i;
                float sum_grad_x = s_reduce[0][i];
                
                // d_h_pre_raw 梯度逻辑
                float s_pre = H_pre[idx];
                d_h_pre_raw[idx] = sum_grad_x * (s_pre * (1.0f - s_pre));
                
                // d_h_post_raw 梯度逻辑
                float s_post = H_post[idx] * 0.5f; 
                d_h_post_raw[idx] = grad_H_post[idx] * 2.0f * (s_post * (1.0f - s_post));
            }
        }
    }
}

/* -------------------- API 封装 -------------------- */

inline void mhc_pre_op_forward(
    floatX* x_layer_in, float* H_pre, float* H_post, float* H_res,
    const floatX* x_expanded, const float* h_pre_raw, const float* h_post_raw, const float* h_res_raw,
    int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream) 
{
    int64_t total_elements = B * T * C;
    dim3 threads(256);
    dim3 blocks((unsigned int)((total_elements + 255) / 256));
    
    mhc_pre_op_fwd_kernel<8><<<blocks, threads, 0, stream>>>(
        x_layer_in, H_pre, H_post, x_expanded, h_pre_raw, h_post_raw, B, T, n, C);

    // 处理 Sinkhorn 投影
    for (int64_t i = 0; i < B * T; i++) {
        sinkhorn_knopp_forward(
            H_res + i * (int64_t)n * n, 
            h_res_raw + i * (int64_t)n * n, 
            n, n, sinkhorn_iters, eps, stream
        );
    }
}

inline void mhc_pre_op_backward(
    floatX* d_x_expanded, float* d_h_pre_raw, float* d_h_post_raw, float* d_h_res_raw,
    const floatX* grad_layer_in, const float* grad_H_post, const float* grad_H_res,
    const floatX* x_expanded, const float* H_pre, const float* H_post, 
    const float* H_res_out, const float* H_res_in_raw,
    int64_t B, int64_t T, int n, int64_t C, int sinkhorn_iters, float eps, cudaStream_t stream) 
{
    const int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks((unsigned int)(B * T));
    
    // 调用反向内核
    mhc_pre_op_bwd_kernel<BLOCK_SIZE, 8><<<blocks, threads, 0, stream>>>(
        d_x_expanded, d_h_pre_raw, d_h_post_raw,
        grad_layer_in, grad_H_post, x_expanded, H_pre, H_post, B, T, n, C);

    // 处理 Sinkhorn 梯度
    for (int64_t i = 0; i < B * T; i++) {
        sinkhorn_knopp_backward(
            d_h_res_raw + i * (int64_t)n * n,
            grad_H_res + i * (int64_t)n * n,
            H_res_out + i * (int64_t)n * n,
            H_res_in_raw + i * (int64_t)n * n,
            n, sinkhorn_iters, eps, stream
        );
    }
}

} // namespace mhc

#endif
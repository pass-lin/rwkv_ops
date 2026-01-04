#ifndef MHC_POST_OP_CUH
#define MHC_POST_OP_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"
#include "type_conversions.cuh"

namespace cg = cooperative_groups;

namespace mhc {

/**
 * 1. Fused Forward Kernel
 * 公式: x_next[b,t,i,c] = sum_j(H_res[b,t,i,j] * x_expanded[b,t,j,c]) + layer_out[b,t,c] * H_post[b,t,i]
 * 精度策略：所有中间累加使用 FP32
 */
template<int MAX_N = 8>
__global__ void mhc_post_op_fwd_kernel(
    floatX* __restrict__ x_next,           // [B, T, n, C]
    const floatX* __restrict__ layer_out,  // [B, T, C]
    const floatX* __restrict__ x_expanded, // [B, T, n, C]
    const float* __restrict__ H_post,      // [B, T, n]
    const float* __restrict__ H_res,       // [B, T, n, n]
    int64_t B, int64_t T, int n, int64_t C) 
{
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y; // 输出流索引

    if (btc < B * T * C && i < n) {
        int64_t bt = btc / C;
        int64_t c = btc % C;
        // [修改]: 增加 (int64_t)n 强转
        int64_t bt_offset_n_c = bt * (int64_t)n * C;

        // --- Stream Mix 部分 ---
        float mixed_val = 0.0f;
        // [修改]: 增加 (int64_t)n 和 (int64_t)i 强转，防止 i*n 在 32 位下溢出
        int64_t res_base = bt * (int64_t)n * n + (int64_t)i * n;
        #pragma unroll
        for (int j = 0; j < MAX_N; j++) {
            if (j < n) {
                float w_res = H_res[res_base + j];
                float val_x = to_float(x_expanded[bt_offset_n_c + (int64_t)j * C + c]);
                mixed_val += w_res * val_x;
            }
        }

        // --- Stream Distribute 部分 ---
        float l_val = to_float(layer_out[btc]);
        // [修改]: 增加 (int64_t)n 强转
        float w_post = H_post[bt * (int64_t)n + i];
        float dist_val = l_val * w_post;

        x_next[bt_offset_n_c + (int64_t)i * C + c] = to_bf(mixed_val + dist_val);
    }
}

/**
 * 2. Fused Backward Full Kernel
 * 计算: 
 * dl = sum_i(grad_next_i * H_post_i)
 * dx_j = sum_i(grad_next_i * H_res_ij)
 * dH_post_i = sum_c(grad_next_i * layer_out)
 * dH_res_ij = sum_c(grad_next_i * x_expanded_j)
 */
template<int BLOCK_SIZE, int MAX_N = 8>
__global__ void mhc_post_op_bwd_full_kernel(
    floatX* __restrict__ d_layer_out,      // [B, T, C]
    floatX* __restrict__ d_x_expanded,     // [B, T, n, C]
    float* __restrict__ d_H_post,          // [B, T, n]
    float* __restrict__ d_H_res,           // [B, T, n, n]
    const floatX* __restrict__ grad_next,  // [B, T, n, C]
    const floatX* __restrict__ layer_out,  // [B, T, C]
    const floatX* __restrict__ x_expanded, // [B, T, n, C]
    const float* __restrict__ H_post,
    const float* __restrict__ H_res,
    int64_t B, int64_t T, int n, int64_t C) 
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int64_t bt = blockIdx.x;
    if (bt >= B * T) return;

    // [修改]: 增加 (int64_t)n 强转
    int64_t bt_offset_n_c = bt * (int64_t)n * C;
    int64_t bt_offset_c = bt * C;

    // 局部存储用于参数梯度的 Reduce
    // dw_post: [n], dw_res: [n, n]
    float thread_dw_post[MAX_N] = {0.0f};
    float thread_dw_res[MAX_N][MAX_N] = {0.0f};

    // --- 1. 计算数据梯度 (dl, dx) ---
    // 每个线程处理一个通道 c
    for (int64_t c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float l_val = to_float(layer_out[bt_offset_c + c]);
        
        // 先读取所有流在该通道的梯度
        float g_vals[MAX_N];
        #pragma unroll
        for(int i=0; i<MAX_N; i++) {
            if(i < n) g_vals[i] = to_float(grad_next[bt_offset_n_c + (int64_t)i * C + c]);
        }

        // 计算 dl (数据梯度)
        float dl_sum = 0.0f;
        #pragma unroll
        for(int i=0; i<MAX_N; i++) {
            // [修改]: 增加 (int64_t)n 强转
            if(i < n) dl_sum += g_vals[i] * H_post[bt * (int64_t)n + i];
        }
        d_layer_out[bt_offset_c + c] = to_bf(dl_sum);

        // 计算 dx (数据梯度) 和 累加参数梯度局部和
        #pragma unroll
        for(int j=0; j<MAX_N; j++) {
            if (j < n) {
                float dx_j = 0.0f;
                float xj_val = to_float(x_expanded[bt_offset_n_c + (int64_t)j * C + c]);
                
                #pragma unroll
                for(int i=0; i<MAX_N; i++) {
                    if (i < n) {
                        // [修改]: 增加 (int64_t)n 和 (int64_t)i 强转
                        dx_j += g_vals[i] * H_res[bt * (int64_t)n * n + (int64_t)i * n + j];
                        // 顺便计算 dH_res 的线程局部部分
                        thread_dw_res[i][j] += g_vals[i] * xj_val;
                    }
                }
                d_x_expanded[bt_offset_n_c + (int64_t)j * C + c] = to_bf(dx_j);
            }
        }

        // 计算 dH_post 的线程局部部分
        #pragma unroll
        for(int i=0; i<MAX_N; i++) {
            if(i < n) thread_dw_post[i] += g_vals[i] * l_val;
        }
    }

    // --- 2. 参数梯度规约 (C 维度的 Reduction) ---
    // 使用 Warp Shuffle 规约并写回
    #pragma unroll
    for(int i=0; i<MAX_N; i++) {
        if(i < n) {
            float sum_p = cg::reduce(warp, thread_dw_post[i], cg::plus<float>());
            // [修改]: 增加 (int64_t)n 强转
            if (warp.thread_rank() == 0) atomicAdd(&d_H_post[bt * (int64_t)n + i], sum_p);
            
            #pragma unroll
            for(int j=0; j<MAX_N; j++) {
                if(j < n) {
                    float sum_r = cg::reduce(warp, thread_dw_res[i][j], cg::plus<float>());
                    // [修改]: 增加 (int64_t)n 和 (int64_t)i 强转
                    if (warp.thread_rank() == 0) atomicAdd(&d_H_res[bt * (int64_t)n * n + (int64_t)i * n + j], sum_r);
                }
            }
        }
    }
}

/* -------------------- API 包装函数 -------------------- */

inline void mhc_post_op_forward(
    floatX* x_next, const floatX* layer_out, const floatX* x_expanded,
    const float* H_post, const float* H_res,
    int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) 
{
    int64_t total_btc = B * T * C;
    dim3 threads(256);
    dim3 blocks((total_btc + 255) / 256, n);
    mhc_post_op_fwd_kernel<8><<<blocks, threads, 0, stream>>>(
        x_next, layer_out, x_expanded, H_post, H_res, B, T, n, C);
}

inline void mhc_post_op_backward_full(
    floatX* d_layer_out, floatX* d_x_expanded, float* d_H_post, float* d_H_res,
    const floatX* grad_next, const floatX* layer_out, const floatX* x_expanded,
    const float* H_post, const float* H_res,
    int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) 
{
    // 每个 Block 负责一个 Token (bt) 的所有通道规约
    const int BLOCK_SIZE = 256;
    dim3 threads(BLOCK_SIZE);
    dim3 blocks(B * T);
    mhc_post_op_bwd_full_kernel<BLOCK_SIZE, 8><<<blocks, threads, 0, stream>>>(
        d_layer_out, d_x_expanded, d_H_post, d_H_res,
        grad_next, layer_out, x_expanded, H_post, H_res, B, T, n, C);
}

} // namespace mhc

#endif
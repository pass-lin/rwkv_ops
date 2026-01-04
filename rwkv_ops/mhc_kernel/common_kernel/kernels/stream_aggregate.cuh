#ifndef MHC_STREAM_AGGREGATE_CUH
#define MHC_STREAM_AGGREGATE_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

/**
 * 1. 前向传播: Out = sum(Inp * H_pre, dim=-2)
 * Out shape: [B*T, C], Inp shape: [B*T, n, C], H_pre: [B*T, n] 或 [n]
 * 每个线程负责输出的一个元素 Out[bt, c]
 */
template<bool PER_TOKEN_H = true>
__global__ void stream_aggregate_fwd_kernel(
    floatX* __restrict__ out,
    const floatX* __restrict__ inp,
    const float* __restrict__ H_pre,
    int64_t BT, int n, int64_t C) {
    
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (btc >= BT * C) return;

    int64_t bt = btc / C;
    int64_t c = btc % C;

    float sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 8; i++) { // 假设 n 最大为 8
        if (i < n) {
            // 根据 H_pre 的形状决定索引方式
            float h_val = PER_TOKEN_H ? H_pre[bt * n + i] : H_pre[i];
            float x_val = to_float(inp[bt * n * C + (int64_t)i * C + c]);
            sum += h_val * x_val;
        }
    }
    out[btc] = to_bf(sum);
}

/**
 * 2. 反向传播 dx: d_inp[bt, i, c] = d_out[bt, c] * H_pre[bt, i]
 */
template<bool PER_TOKEN_H = true>
__global__ void stream_aggregate_bwd_dx_kernel(
    floatX* __restrict__ d_inp,
    const float* __restrict__ d_out, // 使用 FP32 梯度
    const float* __restrict__ H_pre,
    int64_t BT, int n, int64_t C) {
    
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y; // 流索引

    if (btc < BT * C && i < n) {
        int64_t bt = btc / C;
        int64_t c = btc % C;

        float h_val = PER_TOKEN_H ? H_pre[bt * n + i] : H_pre[i];
        float grad_val = d_out[btc];
        
        d_inp[bt * n * C + (int64_t)i * C + c] = to_bf(grad_val * h_val);
    }
}

/**
 * 3. 反向传播 dH: dH[bt, i] = sum_c(d_out[bt, c] * inp[bt, i, c])
 * 每个 Block 负责计算 dH 的一个元素 (bt, i)，在 C 维度进行规约
 */
template<int BLOCK_SIZE>
__global__ void stream_aggregate_bwd_dh_kernel(
    float* __restrict__ d_H_pre,
    const float* __restrict__ d_out,
    const floatX* __restrict__ inp,
    int64_t BT, int n, int64_t C) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // blockIdx.x 对应 (B*T), blockIdx.y 对应流索引 i
    int64_t bt = blockIdx.x;
    int i = blockIdx.y;

    if (bt >= BT || i >= n) return;

    extern __shared__ float s_reduce[];

    float thread_sum = 0.0f;
    int64_t out_offset = bt * C;
    int64_t inp_offset = bt * n * C + (int64_t)i * C;

    // 1. 线程局部求和
    for (int64_t c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float g_val = d_out[out_offset + c];
        float x_val = to_float(inp[inp_offset + c]);
        thread_sum += g_val * x_val;
    }

    // 2. 并行规约 (Warp -> Shared Mem -> Block)
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) s_reduce[warp_id] = warp_sum;
    block.sync();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            // 写回 dH
            d_H_pre[bt * n + i] = block_sum;
        }
    }
}

/* -------------------- API 包装函数 -------------------- */

inline void stream_aggregate_forward(
    floatX* out, const floatX* inp, const float* H_pre,
    int64_t BT, int n, int64_t C, bool per_token, cudaStream_t stream) {
    
    dim3 threads(256);
    dim3 blocks((BT * C + 255) / 256);

    if (per_token) {
        stream_aggregate_fwd_kernel<true><<<blocks, threads, 0, stream>>>(out, inp, H_pre, BT, n, C);
    } else {
        stream_aggregate_fwd_kernel<false><<<blocks, threads, 0, stream>>>(out, inp, H_pre, BT, n, C);
    }
}

inline void stream_aggregate_backward(
    floatX* d_inp, float* d_H_pre, const float* d_out, 
    const floatX* inp, const float* H_pre,
    int64_t BT, int n, int64_t C, bool per_token, cudaStream_t stream) {
    
    // 1. 计算 dx
    dim3 threads_dx(256);
    dim3 blocks_dx((BT * C + 255) / 256, n);
    if (per_token) {
        stream_aggregate_bwd_dx_kernel<true><<<blocks_dx, threads_dx, 0, stream>>>(d_inp, d_out, H_pre, BT, n, C);
    } else {
        stream_aggregate_bwd_dx_kernel<false><<<blocks_dx, threads_dx, 0, stream>>>(d_inp, d_out, H_pre, BT, n, C);
    }

    // 2. 计算 dH (并行规约版)
    constexpr int DH_BLOCK_SIZE = 256;
    dim3 grid_dh(BT, n);
    size_t smem_size = (DH_BLOCK_SIZE / 32) * sizeof(float);
    stream_aggregate_bwd_dh_kernel<DH_BLOCK_SIZE>
        <<<grid_dh, DH_BLOCK_SIZE, smem_size, stream>>>(d_H_pre, d_out, inp, BT, n, C);
}

} // namespace mhc

#endif
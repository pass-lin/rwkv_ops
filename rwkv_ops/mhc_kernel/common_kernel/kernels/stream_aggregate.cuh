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
 * 前向传播：高精度版
 * Out = sum(H_i * X_i)
 */
template<bool PER_TOKEN_H>
__global__ void stream_aggregate_fwd_kernel(
    floatX* __restrict__ out,
    const floatX* __restrict__ inp,
    const float* __restrict__ H_pre,
    int64_t BT, int n, int64_t C) {
    
    // blockIdx.x 强转为 int64_t，防止 32 位溢出
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (btc >= BT * C) return;

    int64_t bt = btc / C;
    int64_t c = btc % C;

    // 核心：使用 FP32 寄存器进行所有乘加运算
    float sum = 0.0f;

    #pragma unroll
    for (int i = 0; i < 8; i++) { // 假设 n <= 8，由 mHC 论文设定
        if (i < n) {
            // [修改]: bt * n 可能超过 32 位，且 n 为 int，显式强转 n 为 int64_t
            float h_val = PER_TOKEN_H ? H_pre[bt * (int64_t)n + i] : H_pre[i];
            // 关键：读取 bf16 后立即转为 fp32 参与运算
            // [修改]: 显式强转 n 为 int64_t，确保 bt * n * C 全程为 64 位运算
            float x_val = to_float(inp[bt * (int64_t)n * C + (int64_t)i * C + c]);
            sum += h_val * x_val;
        }
    }
    // 最后一次性转回 bf16
    out[btc] = to_bf(sum);
}

/**
 * 反向传播 dx: d_inp = d_out * H_pre
 */
template<bool PER_TOKEN_H>
__global__ void stream_aggregate_bwd_dx_kernel(
    floatX* __restrict__ d_inp,
    const float* __restrict__ d_out, 
    const float* __restrict__ H_pre,
    int64_t BT, int n, int64_t C) {
    
    // blockIdx.x 强转为 int64_t
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y; 

    if (btc < BT * C && i < n) {
        int64_t bt = btc / C;
        // [修改]: bt * n 增加 (int64_t)n 强转
        float h_val = PER_TOKEN_H ? H_pre[bt * (int64_t)n + i] : H_pre[i];
        float grad_val = d_out[btc]; // 接收 FP32 梯度
        
        // [修改]: bt * n * C 增加 (int64_t)n 强转
        d_inp[bt * (int64_t)n * C + (int64_t)i * C + (btc % C)] = to_bf(grad_val * h_val);
    }
}

/**
 * 反向传播 dH: d_H = sum_over_C(d_out * inp)
 * 采用并行规约（Parallel Reduction）以保证高精度
 */
template<int BLOCK_SIZE>
__global__ void stream_aggregate_bwd_dh_kernel(
    float* __restrict__ d_H_pre,
    const float* __restrict__ d_out,
    const floatX* __restrict__ inp,
    int64_t BT, int n, int64_t C) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // blockIdx.x 在 C++ 侧对应 BT，若 BT 很大这里 int64_t 是必要的
    int64_t bt = blockIdx.x; 
    int i = blockIdx.y;
    if (bt >= BT || i >= n) return;

    extern __shared__ float s_reduce[];

    float thread_sum = 0.0f;
    for (int64_t c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float g_val = d_out[bt * C + c]; 
        // [修改]: bt * n * C 增加 (int64_t)n 强转
        float x_val = to_float(inp[bt * (int64_t)n * C + (int64_t)i * C + c]);
        thread_sum += g_val * x_val;
    }

    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) s_reduce[warp_id] = warp_sum;
    block.sync();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());
        // [修改]: bt * n 增加 (int64_t)n 强转
        if (lane_id == 0) d_H_pre[bt * (int64_t)n + i] = block_sum;
    }
}

inline void stream_aggregate_forward(floatX* out, const floatX* inp, const float* H_pre, int64_t BT, int n, int64_t C, bool per_token, cudaStream_t stream) {
    dim3 threads(256);
    // 注意: BT * C 如果极大，超过 uint32 范围，dim3.x 会截断，这是 CUDA 硬件限制。
    // 但计算过程本身 BT*C 是 int64 不会溢出。
    dim3 blocks((BT * C + 255) / 256);
    if (per_token) stream_aggregate_fwd_kernel<true><<<blocks, threads, 0, stream>>>(out, inp, H_pre, BT, n, C);
    else stream_aggregate_fwd_kernel<false><<<blocks, threads, 0, stream>>>(out, inp, H_pre, BT, n, C);
}

inline void stream_aggregate_backward(floatX* d_inp, float* d_H_pre, const float* d_out, const floatX* inp, const float* H_pre, int64_t BT, int n, int64_t C, bool per_token, cudaStream_t stream) {
    dim3 threads_dx(256);
    dim3 blocks_dx((BT * C + 255) / 256, n);
    if (per_token) stream_aggregate_bwd_dx_kernel<true><<<blocks_dx, threads_dx, 0, stream>>>(d_inp, d_out, H_pre, BT, n, C);
    else stream_aggregate_bwd_dx_kernel<false><<<blocks_dx, threads_dx, 0, stream>>>(d_inp, d_out, H_pre, BT, n, C);

    constexpr int DH_BLOCK_SIZE = 256;
    dim3 grid_dh(BT, n);
    size_t smem_size = (DH_BLOCK_SIZE / 32) * sizeof(float);
    stream_aggregate_bwd_dh_kernel<DH_BLOCK_SIZE><<<grid_dh, DH_BLOCK_SIZE, smem_size, stream>>>(d_H_pre, d_out, inp, BT, n, C);
}

} // namespace mhc
#endif
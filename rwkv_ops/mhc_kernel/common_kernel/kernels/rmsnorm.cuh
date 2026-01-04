#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

/* -------------------- 前向传播 (修正索引溢出) -------------------- */
template<int BLOCK_SIZE>
__global__ void rmsnorm_fwd_kernel(
    floatX* __restrict__ out, 
    const floatX* __restrict__ inp, 
    int N, int C, float eps) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // 使用 int64_t 接收 blockIdx.x
    int64_t row_idx = blockIdx.x;
    if (row_idx >= N) return;

    // 关键修正：强制 size_t 运算，防止 N * C 溢出
    size_t offset = (size_t)row_idx * C;
    const floatX* x_ptr = inp + offset;
    floatX* o_ptr = out + offset;

    // 共享内存用于存储每个 Warp 的局部和
    extern __shared__ float s_reduce[]; 

    // 1. 计算平方和 (使用 FP32 累加)
    float thread_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = mhc::to_float(x_ptr[i]); 
        thread_sum_sq += val * val;
    }

    // 2. Warp 级规约
    float warp_sum = cg::reduce(warp, thread_sum_sq, cg::plus<float>());
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_reduce[warp_id] = warp_sum;
    block.sync();

    // 3. Block 级规约
    if (warp_id == 0) {
        float b_sum = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        b_sum = cg::reduce(warp, b_sum, cg::plus<float>());
        if (lane_id == 0) s_reduce[0] = b_sum;
    }
    block.sync();

    // 4. 计算 RMS 逆
    float rms_inv = rsqrtf((s_reduce[0] / (float)C) + eps);

    // 5. 写回结果
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float val = mhc::to_float(x_ptr[i]);
        o_ptr[i] = mhc::to_bf(val * rms_inv);
    }
}

/* -------------------- 反向传播 (修正索引溢出) -------------------- */
template<int BLOCK_SIZE>
__global__ void rmsnorm_bwd_kernel(
    floatX* __restrict__ dx,
    const floatX* __restrict__ grad,
    const floatX* __restrict__ x,
    int N, int C, float eps) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int64_t row_idx = blockIdx.x;
    if (row_idx >= N) return;

    // 关键修正：强制 size_t 运算
    size_t offset = (size_t)row_idx * C;
    const floatX* g_ptr = grad + offset;
    const floatX* x_ptr = x + offset;
    floatX* dx_ptr = dx + offset;

    extern __shared__ float s_mem[]; 
    int num_warps = BLOCK_SIZE / 32;
    float* s_sum_sq = s_mem;
    float* s_dot = s_mem + num_warps;

    // 1. 局部累加
    float t_sum_sq = 0.0f;
    float t_dot = 0.0f;
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float xv = mhc::to_float(x_ptr[i]);
        float gv = mhc::to_float(g_ptr[i]);
        t_sum_sq += xv * xv;
        t_dot += gv * xv;
    }

    // 2. Warp 级规约
    float w_sum = cg::reduce(warp, t_sum_sq, cg::plus<float>());
    float w_dot = cg::reduce(warp, t_dot, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) {
        s_sum_sq[warp_id] = w_sum;
        s_dot[warp_id] = w_dot;
    }
    block.sync();

    // 3. Block 级规约
    if (warp_id == 0) {
        float v1 = (lane_id < num_warps) ? s_sum_sq[lane_id] : 0.0f;
        float v2 = (lane_id < num_warps) ? s_dot[lane_id] : 0.0f;
        s_sum_sq[0] = cg::reduce(warp, v1, cg::plus<float>());
        s_dot[0] = cg::reduce(warp, v2, cg::plus<float>());
    }
    block.sync();

    // 4. 计算中间项
    float r2_inv = 1.0f / (s_sum_sq[0] / (float)C + eps); 
    float rms_inv = sqrtf(r2_inv);
    float projection = s_dot[0] * (r2_inv * rms_inv) / (float)C;

    // 5. 应用公式并写回
    for (int i = threadIdx.x; i < C; i += BLOCK_SIZE) {
        float xv = mhc::to_float(x_ptr[i]);
        float gv = mhc::to_float(g_ptr[i]);
        dx_ptr[i] = mhc::to_bf(gv * rms_inv - xv * projection);
    }
}

/* -------------------- 包装函数 -------------------- */

inline void rmsnorm_forward(floatX* out, const floatX* inp, int N, int C, float eps, cudaStream_t stream) {
    const int BLOCK_SIZE = 256;
    size_t smem = (BLOCK_SIZE / 32) * sizeof(float);
    // Grid size 使用 int64 兼容的 N
    rmsnorm_fwd_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE, smem, stream>>>(out, inp, N, C, eps);
}

inline void rmsnorm_backward(floatX* dx, const floatX* grad, const floatX* x, int N, int C, float eps, cudaStream_t stream) {
    const int BLOCK_SIZE = 256;
    size_t smem = (BLOCK_SIZE / 32) * 2 * sizeof(float);
    rmsnorm_bwd_kernel<BLOCK_SIZE><<<N, BLOCK_SIZE, smem, stream>>>(dx, grad, x, N, C, eps);
}

} // namespace mhc
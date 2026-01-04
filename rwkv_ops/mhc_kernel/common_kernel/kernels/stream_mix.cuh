#ifndef MHC_STREAM_MIX_CUH
#define MHC_STREAM_MIX_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

/**
 * 1. 前向传播: Out = M @ Inp
 * Shape: M [B, T, n, n] (FP32), Inp [B, T, n, C] (BF16) -> Out [B, T, n, C] (BF16)
 * 公式: out[b, t, i, c] = \sum_{j=0}^{n-1} M[b, t, i, j] * inp[b, t, j, c]
 */
__global__ void stream_mix_fwd_kernel(
    floatX* __restrict__ out,
    const floatX* __restrict__ inp,
    const float* __restrict__ M,
    int64_t B, int64_t T, int n, int64_t C) {
    
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y; // 目标流索引 (Row of M)

    if (btc < B * T * C && i < n) {
        int64_t b_t = btc / C;
        int64_t c = btc % C;

        float sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; j++) { // 假设 n 最大为 8，可以根据需要调整或改用循环
            if (j < n) {
                float m_val = M[b_t * n * n + (int64_t)i * n + j];
                float x_val = to_float(inp[b_t * n * C + (int64_t)j * C + c]);
                sum += m_val * x_val;
            }
        }
        out[b_t * n * C + (int64_t)i * C + c] = to_bf(sum);
    }
}

/**
 * 2. 反向传播 dx: dx = M^T @ grad
 * 公式: dx[b, t, j, c] = \sum_{i=0}^{n-1} grad[b, t, i, c] * M[b, t, i, j]
 */
__global__ void stream_mix_bwd_dx_kernel(
    floatX* __restrict__ dx,
    const float* __restrict__ grad, // 使用 FP32 梯度以保证精度
    const float* __restrict__ M,
    int64_t B, int64_t T, int n, int64_t C) {
    
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y; // 输入流索引 (Column of M)

    if (btc < B * T * C && j < n) {
        int64_t b_t = btc / C;
        int64_t c = btc % C;

        float sum = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (i < n) {
                float m_val = M[b_t * n * n + (int64_t)i * n + j]; // 注意 M 这里的索引是 [i, j]
                float g_val = grad[b_t * n * C + (int64_t)i * C + c];
                sum += m_val * g_val;
            }
        }
        dx[b_t * n * C + (int64_t)j * C + c] = to_bf(sum);
    }
}

/**
 * 3. 反向传播 dM (优化版): dM = grad @ Inp^T
 * 公式: dM[b, t, i, j] = \sum_{c=0}^{C-1} grad[b, t, i, c] * inp[b, t, j, c]
 * 每个 Block 负责计算 dM 的一个元素，利用共享内存进行并行规约
 */
template<int BLOCK_SIZE>
__global__ void stream_mix_bwd_dm_optimized_kernel(
    float* __restrict__ dm,
    const float* __restrict__ grad,
    const floatX* __restrict__ inp,
    int64_t B, int64_t T, int n, int64_t C) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // blockIdx.x 对应序列维度 (B*T)
    // blockIdx.y 对应 M 的行 i
    // blockIdx.z 对应 M 的列 j
    int64_t bt = blockIdx.x;
    int i = blockIdx.y;
    int j = blockIdx.z;

    if (bt >= B * T || i >= n || j >= n) return;

    extern __shared__ float s_reduce[]; 

    float thread_sum = 0.0f;
    int64_t grad_offset = bt * n * C + (int64_t)i * C;
    int64_t inp_offset = bt * n * C + (int64_t)j * C;

    // 1. 线程局部求和
    for (int64_t c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float g_val = grad[grad_offset + c];
        float x_val = to_float(inp[inp_offset + c]);
        thread_sum += g_val * x_val;
    }

    // 2. Warp 级规约
    float warp_sum = cg::reduce(warp, thread_sum, cg::plus<float>());

    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    if (lane_id == 0) {
        s_reduce[warp_id] = warp_sum;
    }
    block.sync();

    // 3. Block 级规约 (由第一个 Warp 完成)
    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) {
            dm[bt * n * n + (int64_t)i * n + j] = block_sum;
        }
    }
}

/* -------------------- API 包装函数 -------------------- */

inline void stream_mix_forward(
    floatX* out, const floatX* inp, const float* M,
    int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) {
    
    dim3 threads(256);
    // x方向覆盖总元素，y方向负责矩阵行索引
    dim3 blocks((B * T * C + 255) / 256, n);
    stream_mix_fwd_kernel<<<blocks, threads, 0, stream>>>(out, inp, M, B, T, n, C);
}

inline void stream_mix_backward(
    floatX* dx, float* dm, const float* grad, const floatX* inp, const float* M,
    int64_t B, int64_t T, int n, int64_t C, cudaStream_t stream) {
    
    // 1. 计算 dx
    dim3 threads_dx(256);
    dim3 blocks_dx((B * T * C + 255) / 256, n);
    stream_mix_bwd_dx_kernel<<<blocks_dx, threads_dx, 0, stream>>>(dx, grad, M, B, T, n, C);

    // 2. 计算 dm (每个元素一个 Block 以实现 C 轴并行规约)
    constexpr int DM_BLOCK_SIZE = 256;
    dim3 grid_dm(B * T, n, n);
    size_t smem_size = (DM_BLOCK_SIZE / 32) * sizeof(float);
    stream_mix_bwd_dm_optimized_kernel<DM_BLOCK_SIZE>
        <<<grid_dm, DM_BLOCK_SIZE, smem_size, stream>>>(dm, grad, inp, B, T, n, C);
}

} // namespace mhc

#endif
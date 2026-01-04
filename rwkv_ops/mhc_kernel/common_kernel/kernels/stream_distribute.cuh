#ifndef MHC_STREAM_DISTRIBUTE_CUH
#define MHC_STREAM_DISTRIBUTE_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "../include/mhc_types.h"

namespace cg = cooperative_groups;

namespace mhc {

/**
 * Forward: Out = Inp * H_post
 * Shape: Inp [B, T, C], H_post [B, T, n] -> Out [B, T, n, C]
 */
__global__ void stream_distribute_fwd_kernel(
    floatX* __restrict__ out,
    const floatX* __restrict__ inp,
    const float* __restrict__ H_post,
    int64_t B, int64_t T, int n, int64_t C) {
    
    // 索引加固：强制使用 int64_t 防止 (B*T*C) 超过 21 亿
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y; // 流索引

    if (btc < B * T * C && i < n) {
        int64_t bt = btc / C;
        int64_t c = btc % C;

        float val = to_float(inp[btc]);
        float weight = H_post[bt * n + i];
        
        // 计算 64 位偏移量
        int64_t target_idx = bt * n * C + (int64_t)i * C + c;
        out[target_idx] = to_bf(val * weight);
    }
}

/**
 * Backward dx: dx = sum_i(grad_i * H_post_i)
 * 精度：FP32 累加
 */
__global__ void stream_distribute_bwd_dx_kernel(
    floatX* __restrict__ dx,
    const floatX* __restrict__ grad,
    const float* __restrict__ H_post,
    int64_t B, int64_t T, int n, int64_t C) {
    
    int64_t btc = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (btc >= B * T * C) return;

    int64_t bt = btc / C;
    int64_t c = btc % C;

    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        // 强制 64 位偏移计算
        float g = to_float(grad[bt * n * C + (int64_t)i * C + c]);
        float w = H_post[bt * n + i];
        sum += g * w; 
    }
    dx[btc] = to_bf(sum);
}

/**
 * Backward dH: dH = sum_c(grad * inp)
 * 精度：利用 Warp Shuffle 在通道维度 C 上进行全精度规约
 */
template<int BLOCK_SIZE>
__global__ void stream_distribute_bwd_dh_kernel(
    float* __restrict__ d_H_post,
    const floatX* __restrict__ grad,
    const floatX* __restrict__ inp,
    int64_t B, int64_t T, int n, int64_t C) {
    
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int64_t bt = blockIdx.x;
    int i = blockIdx.y;
    if (bt >= B * T || i >= n) return;

    int64_t base_grad = bt * n * C + (int64_t)i * C;
    int64_t base_inp = bt * C;

    float thread_sum = 0.0f;
    for (int64_t c = threadIdx.x; c < C; c += BLOCK_SIZE) {
        float g = to_float(grad[base_grad + c]);
        float x = to_float(inp[base_inp + c]);
        thread_sum += g * x;
    }

    // Block 级规约
    float sum = cg::reduce(warp, thread_sum, cg::plus<float>());
    static __shared__ float s_reduce[32];
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (lane_id == 0) s_reduce[warp_id] = sum;
    block.sync();

    if (warp_id == 0) {
        float val = (lane_id < (BLOCK_SIZE / 32)) ? s_reduce[lane_id] : 0.0f;
        float block_sum = cg::reduce(warp, val, cg::plus<float>());
        if (lane_id == 0) d_H_post[bt * n + i] = block_sum;
    }
}

} // namespace mhc
#endif
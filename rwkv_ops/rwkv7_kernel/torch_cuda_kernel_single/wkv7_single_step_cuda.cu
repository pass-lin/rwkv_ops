// RWKV-7 PyTorch 单步 CUDA kernel。
#include <cuda_bf16.h>
#include <assert.h>
#include <cstdint>

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf &u) {
    return __bfloat162float(u);
}

__device__ inline bf to_bf(const float &u) {
    return __float2bfloat16_rn(u);
}

typedef bf *__restrict__ F_;

// RWKV-7 单步前向 kernel（T=1）。
//
// 每个 block 处理一个 (batch, head)，计算单步 delta rule 并输出 y 与下一状态。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, H, C], bfloat16, row-major。
//   h0_: [B, H, C, C], float32, row-major。初始状态。
//   y_:  [B, H, C], bfloat16, row-major。输出 y。
//   h1_: [B, H, C, C], float32, row-major。输出状态。
//
// Grid/block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (C, 1)，C 为 head_size。
//
// 编译期宏:
//   _C_: head_size，必须被 4 整除。
template<int C>
__launch_bounds__(C, 2)
__global__ void forward_single_step_kernel(
    int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    float *h0_,
    bf *y_,
    float *h1_
) {

    int bb = blockIdx.y;  // Batch index
    int hh = blockIdx.x;  // Head index
    int i = threadIdx.x;  // Row index (0..C-1)

    // Load parameters for this (bb, hh, i)
    // Shape: (B, H, C)
    int64_t param_idx = (int64_t)bb * H * C + hh * C + i;

    float w_val = to_float(w_[param_idx]);
    w_val = __expf(-__expf(w_val));  // Decay factor
    float q_val = to_float(q_[param_idx]);
    float k_val = to_float(k_[param_idx]);
    float v_val = to_float(v_[param_idx]);  // Load per-thread v
    float a_val = to_float(a_[param_idx]);
    float b_val = to_float(b_[param_idx]);

    // Load state row i from h0_: (B, H, C, C)
    int64_t h0_base = (int64_t)bb * H * C * C + hh * C * C + i * C;
    float state_row[C];
#pragma unroll
    for (int j = 0; j < C; j++) {
        state_row[j] = h0_[h0_base + j];
    }

    // Share vectors across threads in block (each thread loads one element)
    __shared__ float shared_a[C], shared_b[C], shared_w[C], shared_k[C], shared_q[C];

    shared_a[i] = a_val;
    shared_b[i] = b_val;
    shared_w[i] = w_val;
    shared_k[i] = k_val;
    shared_q[i] = q_val;
    __syncthreads();

    // Compute sa = sum_j(a[j] * state[i][j])
    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < C; j++) {
        sa += shared_a[j] * state_row[j];
    }

    // Update state row i and compute output element i
    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < C; j++) {
        state_row[j] = state_row[j] * shared_w[j] + sa * shared_b[j] + shared_k[j] * v_val;
        y += state_row[j] * shared_q[j];
    }

    // Write output y[i]: (B, H, C)
    int64_t y_idx = (int64_t)bb * H * C + hh * C + i;
    y_[y_idx] = to_bf(y);

    // Write new state row i to h1_: (B, H, C, C)
    int64_t h1_base = (int64_t)bb * H * C * C + hh * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        h1_[h1_base + j] = state_row[j];
    }
}


// PyTorch 单步前向 C 接口。
//
// Args:
//   w, q, k, v, a, b: [B, H, C], bfloat16, row-major。
//   h0: [B, H, C, C], float32, row-major。初始状态。
//   y:  [B, H, C], bfloat16, row-major。输出 y。
//   h1: [B, H, C, C], float32, row-major。输出状态。
//
// 编译期宏:
//   _C_: head_size。
void cuda_forward_single_step(
    int B, int H,
    bf *w, bf *q, bf *k, bf *v, bf *a, bf *b,
    float *h0, bf *y, float *h1
) {
    dim3 blocks(H, B);
    dim3 threads(_C_);

    forward_single_step_kernel<_C_><<<blocks, threads>>>(
        H, w, q, k, v, a, b, h0, y, h1
    );
}
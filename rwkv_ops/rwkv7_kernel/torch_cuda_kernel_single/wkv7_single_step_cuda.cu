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

// Single-step forward kernel for T=1
template<int C>  
__launch_bounds__(C, 2)
__global__ void forward_single_step_kernel(
    int H,  // Number of heads
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    float *h0_,  // (B, H, C, C) - input state
    bf *y_,      // (B, H, C) - output
    float *h1_   // (B, H, C, C) - output state
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


void cuda_forward_single_step(
    int B, int H,
    bf *w, bf *q, bf *k, bf *v, bf *a, bf *b,
    float *h0, bf *y, float *h1
) {
    dim3 blocks(H, B);  // (num_heads, batch_size)
    dim3 threads(_C_);  // HEAD_SIZE

    forward_single_step_kernel<_C_><<<blocks, threads>>>(
        H, w, q, k, v, a, b, h0, y, h1
    );
}
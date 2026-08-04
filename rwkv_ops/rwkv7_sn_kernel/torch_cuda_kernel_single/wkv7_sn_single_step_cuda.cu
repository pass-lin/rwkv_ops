// RWKV-7 State Neutralization PyTorch 单步 CUDA kernel。
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

// RWKV-7-SN 单步前向 kernel（T=1），按 do_sn 选择是否执行 State Neutralization。
//
// 每个 block 处理一个 (batch, head)。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, H, C], bfloat16, row-major。
//   tau_:  [B, H], float32。SN 阈值。
//   do_sn_: [B], int8。非零表示对该 sample 执行 SN。
//   h0_: [B, H, C, C], float32, row-major。初始状态。
//   y_:  [B, H, C], bfloat16, row-major。输出 y。
//   h1_: [B, H, C, C], float32, row-major。输出状态。
//
// Grid/block: grid (H, B)，block (C, 1)。
// 编译期宏: _C_ 为 head_size。
template<int C>
__launch_bounds__(C, 2)
__global__ void forward_single_step_sn_kernel(
    int B, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    const float* __restrict__ tau_,
    const int8_t* __restrict__ do_sn_,
    float* h0_, bf* y_, float* h1_)
{
    int bb = blockIdx.y;
    int hh = blockIdx.x;
    int i = threadIdx.x;

    int64_t param_idx = (int64_t)bb * H * C + hh * C + i;

    float w_val = to_float(w_[param_idx]);
    w_val = __expf(-__expf(w_val));
    float q_val = to_float(q_[param_idx]);
    float k_val = to_float(k_[param_idx]);
    float v_val = to_float(v_[param_idx]);
    float a_val = to_float(a_[param_idx]);
    float b_val = to_float(b_[param_idx]);

    int64_t h0_base = (int64_t)bb * H * C * C + hh * C * C + i * C;
    float state_row[C];
#pragma unroll
    for (int j = 0; j < C; j++) {
        state_row[j] = h0_[h0_base + j];
    }

    __shared__ float shared_a[C], shared_b[C], shared_w[C], shared_k[C], shared_q[C];
    shared_a[i] = a_val;
    shared_b[i] = b_val;
    shared_w[i] = w_val;
    shared_k[i] = k_val;
    shared_q[i] = q_val;
    __syncthreads();

    float sa = 0.0f;
#pragma unroll
    for (int j = 0; j < C; j++) {
        sa += shared_a[j] * state_row[j];
    }

    float y = 0.0f;
#pragma unroll
    for (int j = 0; j < C; j++) {
        state_row[j] = state_row[j] * shared_w[j] + sa * shared_b[j] + shared_k[j] * v_val;
        y += state_row[j] * shared_q[j];
    }

    int64_t y_idx = (int64_t)bb * H * C + hh * C + i;
    y_[y_idx] = to_bf(y);

    if (do_sn_ != nullptr && do_sn_[bb] != 0) {
        float tau = tau_[bb * H + hh];
        if (tau > 0.0f) {
#pragma unroll
            for (int j = 0; j < C; j++) {
                state_row[j] = tau * tanhf(state_row[j] / tau);
            }
        }
    }

    int64_t h1_base = (int64_t)bb * H * C * C + hh * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        h1_[h1_base + j] = state_row[j];
    }
}

// PyTorch SN 单步前向 C 接口。
//
// Args:
//   w, q, k, v, a, b: [B, H, C], bfloat16, row-major。
//   tau:  [B, H], float32。
//   do_sn: [B], int8。
//   h0: [B, H, C, C], float32, row-major。初始状态。
//   y:  [B, H, C], bfloat16, row-major。输出 y。
//   h1: [B, H, C, C], float32, row-major。输出状态。
void cuda_forward_single_step_sn(
    int B, int H,
    bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
    const float* tau, const int8_t* do_sn,
    float* h0, bf* y, float* h1)
{
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_single_step_sn_kernel<C><<<blocks, threads>>>(
        B, H, w, q, k, v, a, b, tau, do_sn, h0, y, h1);
}

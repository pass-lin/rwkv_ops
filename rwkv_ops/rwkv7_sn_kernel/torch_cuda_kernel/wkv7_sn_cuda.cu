// RWKV-7 State Neutralization PyTorch CUDA 训练/推理 kernel。

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
typedef bf * __restrict__ F_;

// mask: [B, T//16]，>0 表示在该 chunk 边界执行 State Neutralization。

// RWKV-7-SN 带 mask 前向训练 kernel。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步；在每个 chunk 边界按 mask
// 对 state 执行 State Neutralization（state = tau * tanh(state / tau)）。
// 输出始终基于 SN 之前的 state。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。阈值，>1。
//   mask_: [B, T//16], float32, row-major。>0 执行 SN。
//   y_: [B, T, H, K], bfloat16, row-major。输出。
//   s_: [B, T//16, H, K, K], float32, row-major。SN 之前的 state checkpoint。
//   sa_: [B, T, H, K], float32, row-major。中间量 sa 供反向使用。
//   h0_: [B, H, K, K], float32, row-major。初始 state。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel_sn(int T, int H,
     F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
     const float* __restrict__ tau_,
     const float* __restrict__ mask_,
     bf* y_, float* s_, float* sa_, float* h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int64_t h0_base = ((int64_t)bb*H + hh)*C*C + i*C;
    #pragma unroll
    for (int j = 0; j < C; j++) state[j] = h0_[h0_base + j];

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = 0; t < T; t++) {
        int64_t ind = (int64_t)bb*T*H*C + (int64_t)t*H*C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) sa += a[j] * state[j];
        sa_[ind] = sa;

        float v_val = to_float(v_[ind]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float &s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v_val;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            int64_t base = ((int64_t)bb*H+hh)*num_chunks*C*C + (int64_t)chunk*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) s_[base + j*C] = state[j];

            int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
            float m = mask_[cond_idx];
            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            #pragma unroll
            for (int j = 0; j < C; j++) {
                state[j] = state[j] * (1.0f - m) + m * tau * tanhf(state[j] / tau);
            }
        }
    }
}

// RWKV-7-SN 带 mask 反向训练 kernel。
//
// 每个 block 处理一个 (batch, head)，从 T-1 倒序扫描到 0；在 chunk 边界先对
// dstate 乘 sech2 完成 SN 梯度回传，并规约得到 dtau。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。
//   mask_: [B, T//16], float32, row-major。
//   dy_: [B, T, H, K], bfloat16, row-major。输出梯度。
//   s_: [B, T//16, H, K, K], float32, row-major。前向保存的 SN 之前 state checkpoint。
//   sa_: [B, T, H, K], float32, row-major。前向保存的中间量 sa。
//   dht_, dh0_: [B, H, K, K], float32, row-major。最终/初始 state 梯度。
//   dtau_: [B, T//16, H], float32, row-major。tau 梯度。
//   dw_, dq_, dk_, dv_, da_, db_: [B, T, H, K], bfloat16, row-major。各输入梯度。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void backward_kernel_sn(int T, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    const float* __restrict__ tau_,
    const float* __restrict__ mask_,
    F_ dy_,
    float * __restrict__ s_, float * __restrict__ sa_,
    float * __restrict__ dht_, float * __restrict__ dh0_,
    float * __restrict__ dtau_,
    bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {

    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    int64_t dht_base = ((int64_t)bb*H + hh)*C*C + i*C;
    #pragma unroll
    for (int j = 0; j < C; j++) {
        dstate[j] = dht_[dht_base + j];
        dstateT[j] = dht_[dht_base + j];
    }
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    __shared__ float dtau_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = T-1; t >= 0; t--) {
        int64_t ind = (int64_t)bb*T*H*C + (int64_t)t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        v[i] = to_float(v_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            int64_t base = ((int64_t)bb*H+hh)*num_chunks*C*C + (int64_t)chunk*C*C + i*C;
            const float4* s4 = (const float4*)(s_ + base);
            #pragma unroll
            for (int j4 = 0; j4 < C/4; j4++) {
                float4 q_vec = s4[j4];
                const int j = j4 * 4;
                stateT[j+0] = q_vec.x; stateT[j+1] = q_vec.y;
                stateT[j+2] = q_vec.z; stateT[j+3] = q_vec.w;
            }

            int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            float m = mask_[cond_idx];
            float inv_tau = 1.0f / tau;
            float dtau_local = 0.0f;
            #pragma unroll
            for (int j = 0; j < C; j++) {
                float u = stateT[j] * inv_tau;
                float tnh = tanhf(u);
                float sech2 = 1.0f - tnh * tnh;
                float blend = (1.0f - m) + m * sech2;
                dtau_local += m * dstate[j] * (tnh - u * sech2);
                dstate[j]  *= blend;
                dstateT[j] *= blend;
            }
            dtau_shared[i] = dtau_local;
            __syncthreads();
            #pragma unroll
            for (int stride = C/2; stride > 0; stride /= 2) {
                if (i < stride) dtau_shared[i] += dtau_shared[i + stride];
                __syncthreads();
            }
            if (i == 0) {
                dtau_[bb * num_chunks * H + chunk * H + hh] = dtau_shared[0];
            }
        }
        float dq_val = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) dq_val += stateT[j] * dy[j];
        dq_[ind] = to_bf(dq_val);

        float iwi = 1.0f/(wi + 0.000001f);
        #pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) da += stateT[j] * dSb_shared[j];
        da_[ind] = to_bf(da);

        #pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
            if (t == 0) dh0_[dht_base + j] = dstate[j];
        }
    }
}

// RWKV-7-SN 带 mask 前向推理 kernel。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步；在每个 chunk 边界按 mask
// 对 state 执行 State Neutralization，只输出 y 与最终 state。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。
//   mask_: [B, T//16], float32, row-major。>0 执行 SN。
//   y_: [B, T, H, K], bfloat16, row-major。输出。
//   s_: [B, H, K, K], float32, row-major。最终 state。
//   h0_: [B, H, K, K], float32, row-major。初始 state。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void forward_inference_kernel_sn(int T, int H,
                                             F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                             const float* __restrict__ tau_,
                                             const float* __restrict__ mask_,
                                             bf *y_, float *s_, float *h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = 0; t < T; ++t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();
        float sa = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) sa += a[j] * state[j];
        float v_val = to_float(v_[ind]);
        float y = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            float &s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v_val;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
            float m = mask_[cond_idx];
            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                state[j] = state[j] * (1.0f - m) + m * tau * tanhf(state[j] / tau);
            }
        }
    }

    int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) s_[base + j] = state[j];
}

// RWKV-7-SN 无 mask 前向训练 kernel。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步；在每个 chunk 边界无条件对
// state 执行 State Neutralization。输出始终基于 SN 之前的 state。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。
//   y_: [B, T, H, K], bfloat16, row-major。输出。
//   s_: [B, T//16, H, K, K], float32, row-major。SN 之前的 state checkpoint。
//   sa_: [B, T, H, K], float32, row-major。中间量 sa 供反向使用。
//   h0_: [B, H, K, K], float32, row-major。初始 state。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel_sn_no_mask(int T, int H,
                                          F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                          const float* __restrict__ tau_,
                                          bf* y_, float* s_, float* sa_, float* h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = 0; t < T; ++t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();

        float sa = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) sa += a[j] * state[j];
        sa_[ind] = sa;

        float v_val = to_float(v_[ind]);
        float y = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            float &s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v_val;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                           (int64_t)chunk * C * C + i;
            #pragma unroll
            for (int j = 0; j < C; ++j) s_[base + j * C] = state[j];

            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                state[j] = tau * tanhf(state[j] / tau);
            }
        }
    }
}

// RWKV-7-SN 无 mask 反向训练 kernel。
//
// 每个 block 处理一个 (batch, head)，从 T-1 倒序扫描到 0；在 chunk 边界无条件对
// dstate 乘 sech2 完成 SN 梯度回传，并规约得到 dtau。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。
//   dy_: [B, T, H, K], bfloat16, row-major。输出梯度。
//   s_: [B, T//16, H, K, K], float32, row-major。前向保存的 SN 之前 state checkpoint。
//   sa_: [B, T, H, K], float32, row-major。前向保存的中间量 sa。
//   dht_, dh0_: [B, H, K, K], float32, row-major。
//   dtau_: [B, T//16, H], float32, row-major。tau 梯度。
//   dw_, dq_, dk_, dv_, da_, db_: [B, T, H, K], bfloat16, row-major。各输入梯度。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void backward_kernel_sn_no_mask(int T, int H,
                                           F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                           const float* __restrict__ tau_,
                                           F_ dy_,
                                           float * __restrict__ s_, float * __restrict__ sa_,
                                           float * __restrict__ dht_, float * __restrict__ dh0_,
                                           float * __restrict__ dtau_,
                                           bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    int64_t dht_base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) {
        dstate[j] = dht_[dht_base + j];
        dstateT[j] = dht_[dht_base + j];
    }
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    __shared__ float dtau_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = T - 1; t >= 0; --t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        v[i] = to_float(v_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                           (int64_t)chunk * C * C + i * C;
            const float4* s4 = (const float4*)(s_ + base);
            #pragma unroll
            for (int j4 = 0; j4 < C / 4; ++j4) {
                float4 q_vec = s4[j4];
                const int j = j4 * 4;
                stateT[j + 0] = q_vec.x;
                stateT[j + 1] = q_vec.y;
                stateT[j + 2] = q_vec.z;
                stateT[j + 3] = q_vec.w;
            }

            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            float inv_tau = 1.0f / tau;
            float dtau_local = 0.0f;
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                float u = stateT[j] * inv_tau;
                float tnh = tanhf(u);
                float sech2 = 1.0f - tnh * tnh;
                dtau_local += dstate[j] * (tnh - u * sech2);
                dstate[j]  *= sech2;
                dstateT[j] *= sech2;
            }
            dtau_shared[i] = dtau_local;
            __syncthreads();
            #pragma unroll
            for (int stride = C / 2; stride > 0; stride /= 2) {
                if (i < stride) dtau_shared[i] += dtau_shared[i + stride];
                __syncthreads();
            }
            if (i == 0) {
                dtau_[bb * num_chunks * H + chunk * H + hh] = dtau_shared[0];
            }
        }
        float dq_val = 0;
        #pragma unroll
        for (int j = 0; j < C; ++j) dq_val += stateT[j] * dy[j];
        dq_[ind] = to_bf(dq_val);

        float iwi = 1.f / (wi + 0.000001f);
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            dw += dstateT[j] * stateT[j];
            dk += dstateT[j] * v[j];
            dv += dstate[j] * k[j];
            dSb += dstate[j] * b[j];
            db += dstateT[j] * sa[j];
        }
        dw_[ind] = to_bf(dw * wi * wi_fac);
        dk_[ind] = to_bf(dk);
        dv_[ind] = to_bf(dv);
        db_[ind] = to_bf(db);

        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();

        float da = 0;
        #pragma unroll
        for (int j = 0; j < C; ++j) da += stateT[j] * dSb_shared[j];
        da_[ind] = to_bf(da);

        #pragma unroll
        for (int j = 0; j < C; ++j) {
            dstate[j]  = dstate[j] * w[j] + dSb * a[j];
            dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
            if (t == 0) dh0_[dht_base + j] = dstate[j];
        }
    }
}

// RWKV-7-SN 无 mask 前向推理 kernel。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步；在每个 chunk 边界无条件对
// state 执行 State Neutralization，只输出 y 与最终 state。
//
// Args:
//   w_, q_, k_, v_, a_, b_: [B, T, H, K], bfloat16, row-major。
//   tau_: [B, T//16, H], float32, row-major。
//   y_: [B, T, H, K], bfloat16, row-major。输出。
//   s_: [B, H, K, K], float32, row-major。最终 state。
//   h0_: [B, H, K, K], float32, row-major。初始 state。
//
// Grid / Block:
//   grid  (H, B)，每个 block 对应一个 (head, batch)。
//   block (_C_,)，每个线程处理 head_size 中一个位置。
//
// 编译期宏:
//   _C_: head_size。
//   _CHUNK_LEN_: chunk 长度，固定 16。
template<int C> __launch_bounds__(C, 2)
__global__ void forward_inference_kernel_sn_no_mask(int T, int H,
                                                    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                                    const float* __restrict__ tau_,
                                                    bf *y_, float *s_, float *h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];

    const int num_chunks = T / _CHUNK_LEN_;

    for (int t = 0; t < T; ++t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();
        float sa = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) sa += a[j] * state[j];
        float v_val = to_float(v_[ind]);
        float y = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            float &s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v_val;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int chunk = t / _CHUNK_LEN_;
            float tau = tau_[bb * num_chunks * H + chunk * H + hh];
            #pragma unroll
            for (int j = 0; j < C; ++j) {
                state[j] = tau * tanhf(state[j] / tau);
            }
        }
    }

    int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) s_[base + j] = state[j];
}


// RWKV-7-SN 带 mask 前向训练 host 包装函数。
//
// Args:
//   w, q, k, v, a, b: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32。
//   mask: [B, T//16], float32。
//   y, s, sa, h0: 形状/dtype 见对应 kernel。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_forward_sn(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                     const float* tau, const float* mask,
                     bf* y, float* s, float* sa, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_kernel_sn<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, mask, y, s, sa, h0);
}

// RWKV-7-SN 带 mask 反向训练 host 包装函数。
//
// Args:
//   w, q, k, v, a, b, dy: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32；mask: [B, T//16], float32。
//   s, sa, dht, dh0, dtau: float32，形状见对应 kernel。
//   dw, dq, dk, dv, da, db: [B, T, H, K], bfloat16。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_backward_sn(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                      const float* tau, const float* mask, bf* dy, float* s, float* sa,
                      float* dht, float* dh0, float* dtau,
                      bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    backward_kernel_sn<C><<<blocks, threads>>>(
        T, H, w, q, k, v, a, b, tau, mask, dy, s, sa, dht, dh0, dtau,
        dw, dq, dk, dv, da, db);
}

// RWKV-7-SN 带 mask 前向推理 host 包装函数。
//
// Args:
//   w, q, k, v, a, b: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32；mask: [B, T//16], float32。
//   y: [B, T, H, K], bfloat16；s, h0: [B, H, K, K], float32。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_forward_inference_sn(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                               const float* tau, const float* mask,
                               bf* y, float* s, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_inference_kernel_sn<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, mask, y, s, h0);
}

// RWKV-7-SN 无 mask 前向训练 host 包装函数。
//
// Args:
//   w, q, k, v, a, b: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32。
//   y, s, sa, h0: 形状/dtype 见对应 kernel。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_forward_sn_no_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                             const float* tau,
                             bf* y, float* s, float* sa, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_kernel_sn_no_mask<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, y, s, sa, h0);
}

// RWKV-7-SN 无 mask 反向训练 host 包装函数。
//
// Args:
//   w, q, k, v, a, b, dy: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32。
//   s, sa, dht, dh0, dtau: float32，形状见对应 kernel。
//   dw, dq, dk, dv, da, db: [B, T, H, K], bfloat16。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_backward_sn_no_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                              const float* tau, bf* dy, float* s, float* sa,
                              float* dht, float* dh0, float* dtau,
                              bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    backward_kernel_sn_no_mask<C><<<blocks, threads>>>(
        T, H, w, q, k, v, a, b, tau, dy, s, sa, dht, dh0, dtau,
        dw, dq, dk, dv, da, db);
}

// RWKV-7-SN 无 mask 前向推理 host 包装函数。
//
// Args:
//   w, q, k, v, a, b: [B, T, H, K], bfloat16。
//   tau: [B, T//16, H], float32。
//   y: [B, T, H, K], bfloat16；s, h0: [B, H, K, K], float32。
//
// 编译期宏:
//   _C_: head_size；_CHUNK_LEN_: chunk 长度，固定 16。
void cuda_forward_inference_sn_no_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                                       const float* tau,
                                       bf* y, float* s, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_inference_kernel_sn_no_mask<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, y, s, h0);
}

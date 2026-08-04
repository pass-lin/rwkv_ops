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

/* mask: [B, T//16]; m > 0 means apply State Neutralization at chunk boundary. */

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

/*  无 mask 训练前向 Kernel  */
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

/*  无 mask 训练反向 Kernel  */
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

/*  无 mask 推理 Kernel  */
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


/*  Host 接口  */
void cuda_forward_sn(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                     const float* tau, const float* mask,
                     bf* y, float* s, float* sa, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_kernel_sn<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, mask, y, s, sa, h0);
}

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

void cuda_forward_inference_sn(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                               const float* tau, const float* mask,
                               bf* y, float* s, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_inference_kernel_sn<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, mask, y, s, h0);
}

/*  Host 接口（无 mask）  */
void cuda_forward_sn_no_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                             const float* tau,
                             bf* y, float* s, float* sa, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_kernel_sn_no_mask<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, y, s, sa, h0);
}

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

void cuda_forward_inference_sn_no_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                                       const float* tau,
                                       bf* y, float* s, float* h0) {
    constexpr int C = _C_;
    dim3 blocks(H, B);
    dim3 threads(C);
    forward_inference_kernel_sn_no_mask<C><<<blocks, threads>>>(T, H, w, q, k, v, a, b, tau, y, s, h0);
}

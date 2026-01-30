#include <cuda_bf16.h>
#include <assert.h>
#include <cstdint>

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) {
    return __bfloat162float(u);
}

__device__ inline bf to_bf(const float & u) {
    return __float2bfloat16_rn(u);
}
typedef bf * __restrict__ F_;


template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel(int T, int H,
     F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
      bf* y_, float* s_, float* sa_, float* h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int64_t h0_base = ((int64_t)bb*H + hh)*C*C + i*C; 
    #pragma unroll
    for (int j = 0; j < C; j++) state[j] = h0_[h0_base + j];
    
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
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) s_[base + j*C] = state[j];
        }
    }
}

template<int C> __launch_bounds__(C, 2)
__global__ void backward_kernel(int T, int H, 
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
    float * __restrict__ s_, float * __restrict__ sa_,
    float * __restrict__ dht_, float * __restrict__ dh0_,
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
    float qi, wi, ki, ai, bi, dyi;

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
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i*C;
            const float4* s4 = (const float4*)(s_ + base);
            #pragma unroll
            for (int j4 = 0; j4 < C/4; j4++) {
                float4 q_vec = s4[j4];
                const int j = j4 * 4;
                stateT[j+0] = q_vec.x; stateT[j+1] = q_vec.y;
                stateT[j+2] = q_vec.z; stateT[j+3] = q_vec.w;
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
__global__ void forward_inference_kernel(int T, int H,
                                         F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                         bf *y_, float *s_, float *h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C; 
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];
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
    }
    int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) s_[base + j] = state[j];
}

template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel_with_mask(int T, int H,
     F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
     bf* y_, float* s_, float* sa_, float* h0_,
     const float* __restrict__ mask_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    int64_t h0_base = ((int64_t)bb*H + hh)*C*C + i*C; 
    #pragma unroll
    for (int j = 0; j < C; j++) state[j] = h0_[h0_base + j];
    
    for (int t = 0; t < T; t++) {
        int64_t ind = (int64_t)bb*T*H*C + (int64_t)t*H*C + hh * C + i;
        float m = mask_[bb * T + t];

        __syncthreads();
        q[i] = to_float(q_[ind]);
        w[i] = __expf(-__expf(to_float(w_[ind])));
        k[i] = to_float(k_[ind]);
        a[i] = to_float(a_[ind]);
        b[i] = to_float(b_[ind]);
        __syncthreads();
        
        float sa_val = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) sa_val += a[j] * state[j];
        sa_[ind] = sa_val;
        
        float v_val = to_float(v_[ind]);
        float y = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float s_prev = state[j];
            float s_cand = s_prev * w[j] + sa_val * b[j] + k[j] * v_val;
            y += s_cand * q[j];
            state[j] = m * s_cand + (1.0f - m) * s_prev;
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i;
            #pragma unroll
            for (int j = 0; j < C; j++) s_[base + j*C] = state[j];
        }
    }
}

template<int C> __launch_bounds__(C, 2)
__global__ void backward_kernel_with_mask(int T, int H, 
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, 
    const float* __restrict__ mask_,
    F_ dy_,
    float * __restrict__ s_, float * __restrict__ sa_,
    float * __restrict__ dht_, float * __restrict__ dh0_,
    bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_)
 {
    
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    
    int64_t dht_base = ((int64_t)bb*H + hh)*C*C + i*C;
    #pragma unroll
    for (int j = 0; j < C; j++) {
        dstate[j] = dht_[dht_base + j];
        dstateT[j] = dht_[dht_base + j];
    }
    
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int64_t ind = (int64_t)bb*T*H*C + (int64_t)t*H*C + hh * C + i;
        
        float m = mask_[bb * T + t];
        float one_minus_m = 1.0f - m;
        
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
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i*C;
            const float4* s4 = (const float4*)(s_ + base);
            #pragma unroll
            for (int j4 = 0; j4 < C/4; j4++) {
                float4 vec = s4[j4];
                const int j = j4 * 4;
                stateT[j+0] = vec.x;
                stateT[j+1] = vec.y;
                stateT[j+2] = vec.z;
                stateT[j+3] = vec.w;
            }
        }
        
        // 【修复1】dq计算：基于实际用于输出的状态
        float dq_val = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            // m=1: 使用存储状态 stateT (即 S_t)
            // m=0: 使用候选状态 S_cand = S_{t-1}*w + sa*b + v*k
            float s_cand = stateT[j] * wi + sa[j] * bi + v[j] * ki;
            float s_for_dq = m * stateT[j] + one_minus_m * s_cand;
            dq_val += s_for_dq * dy[j];
        }
        dq_[ind] = to_bf(dq_val);
        
        // 【修复2】累加当前输出梯度到总梯度
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        
        // 【修复3】保存完整梯度（包含未来穿透+当前输出）
        float dstate_old[C], dstateT_old[C];
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dstate_old[j] = dstate[j];
            dstateT_old[j] = dstateT[j];
        }
        
        // 【修复4】逆推S_{t-1}（仅当m=1时需要，m=0时保持现状）
        float iwi = 1.0f/(wi + 0.000001f);
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float s_prev = (stateT[j] - v[j]*ki - sa[j]*bi) * iwi;
            stateT[j] = m * s_prev + one_minus_m * stateT[j];
        }
        
        // 【修复5】分离参数计算用的梯度
        // m=1: 使用完整梯度（因为参数影响存储状态）
        // m=0: 仅用当前步梯度（因为参数不影响存储状态，只影响候选状态）
        float dstate_param[C], dstateT_param[C];
        #pragma unroll
        for (int j = 0; j < C; j++) {
            float dstate_curr = dyi * q[j];
            float dstateT_curr = qi * dy[j];
            dstate_param[j] = m * dstate_old[j] + one_minus_m * dstate_curr;
            dstateT_param[j] = m * dstateT_old[j] + one_minus_m * dstateT_curr;
        }
        
        // 【修复6】参数梯度计算（使用dstate_param和逆推后的stateT）
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
        #pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT_param[j] * stateT[j];
            dk += dstateT_param[j] * v[j];
            dv += dstate_param[j] * k[j];
            dSb += dstate_param[j] * b[j];
            db += dstateT_param[j] * sa[j];
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
        for (int j = 0; j < C; j++) {
            da += stateT[j] * dSb_shared[j];
        }
        da_[ind] = to_bf(da);
        
        // 【修复7 - 关键修正】状态梯度回传
        // 当m=1: S_t = \tilde{S}_t, 梯度通过trans标准回传
        // 当m=0: S_t = S_{t-1}, 梯度有两条路径：
        //   1. 未来梯度直接穿透：dstate_old - dstate_param
        //   2. 当前梯度通过候选状态回传：trans(dstate_param)
        #pragma unroll
        for (int j = 0; j < C; j++) {
            // 标准RNN回传（基于dstate_param）
            float trans_row = dstate_param[j] * w[j] + dSb * a[j];
            float trans_col = dstateT_param[j] * wi + ai * dSb_shared[j];
            
            // 穿透部分（仅当m=0时存在）
            float penetration_row = dstate_old[j] - dstate_param[j];
            float penetration_col = dstateT_old[j] - dstateT_param[j];
            
            // 关键修正：m=0时需要同时保留trans和penetration
            // 原错误代码：dstate[j] = m * trans_row + one_minus_m * penetration_row;
            // 修正后：always保留trans，m=0时额外加上penetration
            dstate[j] = trans_row + one_minus_m * penetration_row;
            dstateT[j] = trans_col + one_minus_m * penetration_col;
            
            if (t == 0) {
                dh0_[dht_base + j] = dstate[j];
            }
        }
    }
}

template<int C> __launch_bounds__(C, 2)
__global__ void forward_inference_kernel_with_mask(int T, int H,
                                         F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                         bf *y_, float *s_, float *h0_,
                                         const float* __restrict__ mask_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C; 
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];
    for (int t = 0; t < T; ++t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        float m = mask_[bb * T + t];
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
            float s_prev = state[j];
            float s_cand = s_prev * w[j] + sa * b[j] + k[j] * v_val;
            y += s_cand * q[j];
            state[j] = m * s_cand + (1.0f - m) * s_prev;
        }
        y_[ind] = to_bf(y);
    }
    int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
    #pragma unroll
    for (int j = 0; j < C; ++j) s_[base + j] = state[j];
}

/* -------------------- C 接口函数 -------------------- */
void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa, float* h0) {
    forward_kernel<_C_><<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa,h0);
}
void cuda_backward(int B, int T, int H,
     bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy,
    float*s, float*sa,float*dht,float*dh0,
    bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<_C_><<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dht,dh0,dw,dq,dk,dv,dz,da);
}
void cuda_forward_inference(int B, int T, int H, 
                            bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, 
                            bf* y, float* s, float* h0) {
    forward_inference_kernel<_C_><<<dim3(H, B), dim3(_C_)>>>(T, H, w, q, k, v, a, b, y, s, h0);
}

// 带 Mask 版本接口（新增）
void cuda_forward_with_mask(int B, int T, int H, bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                            bf* y, float* s, float* sa, float* h0, float* mask) {
    forward_kernel_with_mask<_C_><<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,a,b,y,s,sa,h0,mask);
}
void cuda_backward_with_mask(int B, int T, int H,
                             bf* w, bf* q, bf* k, bf* v, bf* a, bf* b,
                             float* mask, bf* dy,
                             float* s, float* sa, float* dht, float* dh0,
                             bf* dw, bf* dq, bf* dk, bf* dv, bf* da, bf* db) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel_with_mask<_C_><<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,a,b,mask,dy,s,sa,dht,dh0,dw,dq,dk,dv,da,db);
}
void cuda_forward_inference_with_mask(int B, int T, int H, 
                                      bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, 
                                      bf* y, float* s, float* h0, float* mask) {
    forward_inference_kernel_with_mask<_C_><<<dim3(H, B), dim3(_C_)>>>(T, H, w, q, k, v, a, b, y, s, h0, mask);
}
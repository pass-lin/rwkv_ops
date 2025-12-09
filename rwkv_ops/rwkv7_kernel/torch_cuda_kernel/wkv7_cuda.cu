#include <cuda_bf16.h>
#include <assert.h>
#include <cstdint> // 必须引入，用于 int64_t

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) {
    return __bfloat162float(u);
}

__device__ inline bf to_bf(const float & u) {
    return __float2bfloat16_rn(u);
}
typedef bf * __restrict__ F_;

__global__ void forward_kernel(int T, int H,
     F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
      bf* y_, float* s_, float* sa_, float* h0_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] =  {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];

    // 【修改】h0_base 使用 int64_t 并强制转换
    int64_t h0_base = ((int64_t)bb*H + hh)*C*C + i*C; 

#pragma unroll
        for (int j = 0; j < C; j++) {
            state[j] = h0_[h0_base + j];
        }
    for (int t = 0; t < T; t++) {
        // 【修改】ind 使用 int64_t，防止 B*T*H*C 溢出
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
        for (int j = 0; j < C; j++) {
            sa += a[j] * state[j];
        }
        sa_[ind] = sa;
        float v = to_float(v_[ind]);
        float y = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float& s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = to_bf(y);

        if ((t+1)%_CHUNK_LEN_ == 0) {
            // 【修改】base 是溢出的主要原因 (State Size)，必须用 int64_t
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

__global__ void backward_kernel(int T, int H, 
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
float * __restrict__ s_, float * __restrict__ sa_,
float * __restrict__ dht_,float * __restrict__ dh0_,
bf* dw_, bf* dq_, bf* dk_, bf* dv_, bf* da_, bf* db_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] =  {0}, dstate[C] =  {0}, dstateT[C] =  {0};
    
    // 【修改】dht_base 升级为 int64_t
    int64_t dht_base = ((int64_t)bb*H + hh)*C*C + i*C;

#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dht_[dht_base + j];
            dstateT[j] = dht_[dht_base + j];
        }
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        // 【修改】ind 升级为 int64_t
        int64_t ind = (int64_t)bb*T*H*C + (int64_t)t*H*C + hh * C + i;
        
        __syncthreads();
        q[i] = qi = to_float(q_[ind]);
        float wi_fac = -__expf(to_float(w_[ind]));
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = to_float(k_[ind]);
        a[i] = ai = to_float(a_[ind]);
        b[i] = bi = to_float(b_[ind]);
        v[i] = to_float(v_[ind]);
        dy[i] = dyi = to_float(dy_[ind]);
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            // 【修改】base 升级为 int64_t
            int64_t base = ((int64_t)bb*H+hh)*(T/_CHUNK_LEN_)*C*C + ((int64_t)t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) {
                stateT[j] = s_[base + j];
            }
        }
        float dq = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dq += stateT[j]*dy[j];
        }
        dq_[ind] = to_bf(dq);
        float iwi = 1.0f/(wi+0.000001f);
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0, dk = 0, dv = 0, db = 0, dSb = 0;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
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
            da += stateT[j]*dSb_shared[j];
        }
        da_[ind] = to_bf(da);
#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j] = dstate[j]*w[j] + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
            if (t==0){
                dh0_[dht_base + j] = dstate[j];
            }
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*y, float*s, float*sa, float* h0) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,y,s,sa,h0);
}

void cuda_backward(int B, int T, int H,
     bf*w, bf*q, bf*k, bf*v, bf*z, bf*a, bf*dy,
    float*s, float*sa,float*dht,float*dh0,
    bf*dw, bf*dq, bf*dk, bf*dv, bf*dz, bf*da
    ) {
    assert(T%_CHUNK_LEN_ == 0);
    backward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,z,a,dy,s,sa,dht,dh0,dw,dq,dk,dv,dz,da);
}
/* -------------------- 推理专用 Kernel -------------------- */
__global__ void forward_inference_kernel(int T, int H,
                                         F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                                         bf *y_, float *s_, float *h0_) {
    constexpr int C = _C_;
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

        // sa 临时计算，不保存到全局内存
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
    
    // 仅写入最终状态 (B, H, K, K)
    int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; ++j) s_[base + j] = state[j];
}

// C 接口
void cuda_forward_inference(int B, int T, int H, 
                            bf* w, bf* q, bf* k, bf* v, bf* a, bf* b, 
                            bf* y, float* s, float* h0) {
    forward_inference_kernel<<<dim3(H, B), dim3(_C_)>>>(T, H, w, q, k, v, a, b, y, s, h0);
}
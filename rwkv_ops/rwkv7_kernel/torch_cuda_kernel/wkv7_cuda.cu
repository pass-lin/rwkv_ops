#include <cuda_bf16.h>
#include <assert.h>

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf & u) {
    return __bfloat162float(u);
}

__device__ inline bf to_bf(const float & u) {
    return __float2bfloat16_rn(u);
}
typedef bf * __restrict__ F_;

__global__ void forward_kernel(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, bf* y_, float* s_, float* sa_, float* h0_) {
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] =  {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int h0_base =( (bb*H + hh)*C + i)*C;
#pragma unroll
        for (int j = 0; j < C; j++) {
            state[j] = h0_[h0_base + j];
        }
    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
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
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) {
                s_[base + j*C] = state[j];
            }
        }
    }
}

void cuda_forward(int B, int T, int H, bf*w, bf*q, bf*k, bf*v,  bf*a, bf*y, float*s, float*sa, float* h0) {
    forward_kernel<<<dim3(H,B), dim3(_C_)>>>(T,H,w,q,k,v,a,y,s,sa,h0);
}


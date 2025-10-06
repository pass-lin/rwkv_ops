/* wkv7_ffi.cu
 * JAX-FFI 版 WKV7 前向 kernel
 * 编译：nvcc -c wkv7_ffi.cu -o wkv7_ffi.cu.o -std=c++17 -O3
 */
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define CHUNK_LEN 16
using bf = __nv_bfloat16;

/* ---------- 工具 ---------- */
__device__ float to_f(bf x) { return __bfloat162float(x); }
__device__ bf   to_bf(float x){ return __float2bfloat16_rn(x); }

/* ---------- Forward kernel ---------- */
template <int C>
__global__ void wkv7_fwd_kernel(
    int B, int T, int H,
    const bf* __restrict__ w, const bf* __restrict__ q,
    const bf* __restrict__ k, const bf* __restrict__ v,
    const bf* __restrict__ a,const bf* __restrict__ b,
    const float* __restrict__ h0,
    bf*   __restrict__ y,
    float* __restrict__ s,
    float* __restrict__ sa
)
{
    int bb = blockIdx.y, hh = blockIdx.x, ii = threadIdx.x;
    float state[C];
    int h0_base = ((bb*H + hh)*C + ii)*C;
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0[h0_base + j];

    __shared__ float sq[C], sk[C], sw[C], sa_[C], sb[C], sv[C];

    for (int t = 0; t < T; ++t) {
        int idx = bb*T*H*C + t*H*C + hh*C + ii;
        __syncthreads();
        sq[ii] = to_f(q[idx]);
        sw[ii] = expf(-expf(to_f(w[idx])));
        sk[ii] = to_f(k[idx]);
        sa_[ii] = to_f(a[idx]);
        sb[ii] = to_f(b[idx]);
        sv[ii] = to_f(v[idx]);
        __syncthreads();

        float ssa = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) ssa += sa_[j]*state[j];
        sa[idx] = ssa;

        float yy = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            state[j] = state[j]*sw[j] + ssa*sb[j] + sk[j]*sv[ii];
            yy += state[j]*sq[j];
        }
        y[idx] = to_bf(yy);

        if ((t+1)%CHUNK_LEN == 0) {
            int base = (bb*H + hh)*(T/CHUNK_LEN)*C*C + (t/CHUNK_LEN)*C*C + ii*C;
            #pragma unroll
            for (int j = 0; j < C; ++j) s[base + j] = state[j];
        }
    }
}

/* ---------- C 接口 ---------- */
extern "C" void wkv7_fwd_cuda(
        int B, int T, int H, int C,
        const void* w, const void* q,
        const void* k, const void* v,
        const void* a,const void* b,
        const float* h0,
        void* y, float* s, float* sa
                            
    )
{
    const bf* w_bf = static_cast<const bf*>(w);
    const bf* q_bf = static_cast<const bf*>(q);
    const bf* k_bf = static_cast<const bf*>(k);
    const bf* v_bf = static_cast<const bf*>(v);
    const bf* a_bf = static_cast<const bf*>(a);
    const bf* b_bf = static_cast<const bf*>(b);
    bf* y_bf = static_cast<bf*>(y);

    dim3 blocks(H, B), threads(C);
    switch (C) {
        case 32:  wkv7_fwd_kernel<32> <<<blocks, threads>>>(B,T,H, w_bf,q_bf,k_bf,v_bf,a_bf,b_bf, h0, y_bf,s,sa); break;
        case 64:  wkv7_fwd_kernel<64> <<<blocks, threads>>>(B,T,H, w_bf,q_bf,k_bf,v_bf,a_bf,b_bf, h0, y_bf,s,sa); break;
        case 128: wkv7_fwd_kernel<128><<<blocks,threads>>>(B,T,H, w_bf,q_bf,k_bf,v_bf,a_bf,b_bf, h0, y_bf,s,sa); break;
        case 256: wkv7_fwd_kernel<256><<<blocks,threads>>>(B,T,H, w_bf,q_bf,k_bf,v_bf,a_bf,b_bf, h0, y_bf,s,sa); break;
    }
}
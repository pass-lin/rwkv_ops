#include <assert.h>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

typedef float * __restrict__ F_;

/* ========== 你的 kernel 完全一致，仅拷贝开始 ========== */
__global__ void forward_kernel(int T, int H,
     F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, float* h0_,
      float* y_, float* s_, float* sa_)
{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    int h0_base = ((bb*H + hh)*C + i)*C;
#pragma unroll
    for (int j = 0; j < C; j++) state[j] = h0_[h0_base + j];

    for (int t = 0; t < T; t++) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = q_[ind];
        w[i] = __expf(-__expf(w_[ind]));
        k[i] = k_[ind];
        a[i] = a_[ind];
        b[i] = b_[ind];
        __syncthreads();

        float sa = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) sa += a[j] * state[j];
        sa_[ind] = sa;

        float v = v_[ind];
        float y = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) {
            float &s = state[j];
            s = s * w[j] + sa * b[j] + k[j] * v;
            y += s * q[j];
        }
        y_[ind] = y;

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i;
#pragma unroll
            for (int j = 0; j < C; j++) s_[base + j*C] = state[j];
        }
    }
}

__global__ void backward_kernel(int T, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
    float * __restrict__ s_, float * __restrict__ sa_,
    float * __restrict__ dht_, float * __restrict__ dh0_,
    float* dw_, float* dq_, float* dk_, float* dv_, float* da_, float* db_)
{
    constexpr int C = _C_;
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    int dht_base = ((bb*H + hh)*C + i)*C;
#pragma unroll
    for (int j = 0; j < C; j++) {
        dstate[j]  = dht_[dht_base + j];
        dstateT[j] = dht_[dht_base + j];
    }
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T-1; t >= 0; t--) {
        int ind = bb*T*H*C + t*H*C + hh * C + i;
        __syncthreads();
        q[i] = qi = q_[ind];
        float wi_fac = -__expf(w_[ind]);
        w[i] = wi = __expf(wi_fac);
        k[i] = ki = k_[ind];
        a[i] = ai = a_[ind];
        b[i] = bi = b_[ind];
        v[i] = v_[ind];
        dy[i] = dyi = dy_[ind];
        sa[i] = sa_[ind];
        __syncthreads();

        if ((t+1)%_CHUNK_LEN_ == 0) {
            int base = (bb*H+hh)*(T/_CHUNK_LEN_)*C*C + (t/_CHUNK_LEN_)*C*C + i*C;
#pragma unroll
            for (int j = 0; j < C; j++) stateT[j] = s_[base + j];
        }
        float dq = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) dq += stateT[j]*dy[j];
        dq_[ind] = dq;

        float iwi = 1.f/(wi + 1e-6f);
#pragma unroll
        for (int j = 0; j < C; j++) {
            stateT[j] = (stateT[j] - ki*v[j] - bi*sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        float dw = 0.f, dk = 0.f, dv = 0.f, db = 0.f, dSb = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dw += dstateT[j]*stateT[j];
            dk += dstateT[j]*v[j];
            dv += dstate[j]*k[j];
            dSb += dstate[j]*b[j];
            db += dstateT[j]*sa[j];
        }
        dw_[ind] = dw * wi * wi_fac;
        dk_[ind] = dk;
        dv_[ind] = dv;
        db_[ind] = db;
        __syncthreads();
        dSb_shared[i] = dSb;
        __syncthreads();
        float da = 0.f;
#pragma unroll
        for (int j = 0; j < C; j++) da += stateT[j]*dSb_shared[j];
        da_[ind] = da;
#pragma unroll
        for (int j = 0; j < C; j++) {
            dstate[j]  = dstate[j]*w[j]  + dSb * a[j];
            dstateT[j] = dstateT[j]*wi + ai * dSb_shared[j];
            if (t == 0) dh0_[dht_base + j] = dstate[j];
        }
    }
}
/* ========== 内核结束 ========== */

/* ---------- 前向宿主函数 ---------- */
ffi::Error WKV7FwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> w,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> k,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> z,
    ffi::Buffer<ffi::F32> a,
    ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::F32> y,
    ffi::ResultBuffer<ffi::F32> s,
    ffi::ResultBuffer<ffi::F32> sa)
{
    constexpr int C = _C_;
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2];
    const dim3 block(C);
    const dim3 grid(H, B);

    forward_kernel<<<grid, block, 0, stream>>>(
        T, H,
        w.typed_data(), q.typed_data(), k.typed_data(), v.typed_data(),
        z.typed_data(), a.typed_data(), h0.typed_data(),
        y->typed_data(), s->typed_data(), sa->typed_data());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA forward_kernel error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Fwd, WKV7FwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()               // 5  w
        .Arg<ffi::Buffer<ffi::F32>>()               // 6  q
        .Arg<ffi::Buffer<ffi::F32>>()               // 7  k
        .Arg<ffi::Buffer<ffi::F32>>()               // 8  v
        .Arg<ffi::Buffer<ffi::F32>>()               // 9  z
        .Arg<ffi::Buffer<ffi::F32>>()               // 10 a
        .Arg<ffi::Buffer<ffi::F32>>()               // 11 h0
        .Ret<ffi::Buffer<ffi::F32>>()               // 12 y
        .Ret<ffi::Buffer<ffi::F32>>()               // 13 s
        .Ret<ffi::Buffer<ffi::F32>>()   
    ,{xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

/* ---------- 反向宿主函数 ---------- */
ffi::Error WKV7BwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::F32> w,
    ffi::Buffer<ffi::F32> q,
    ffi::Buffer<ffi::F32> k,
    ffi::Buffer<ffi::F32> v,
    ffi::Buffer<ffi::F32> z,   // kernel a_
    ffi::Buffer<ffi::F32> a,   // kernel b_
    ffi::Buffer<ffi::F32> dy,
    ffi::Buffer<ffi::F32> s,
    ffi::Buffer<ffi::F32> sa,
    ffi::Buffer<ffi::F32> dht,
    ffi::ResultBuffer<ffi::F32> dh0,
    ffi::ResultBuffer<ffi::F32> dw,
    ffi::ResultBuffer<ffi::F32> dq,
    ffi::ResultBuffer<ffi::F32> dk,
    ffi::ResultBuffer<ffi::F32> dv,
    ffi::ResultBuffer<ffi::F32> da,
    ffi::ResultBuffer<ffi::F32> db
)
{
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2];
    constexpr int C = _C_;
    const dim3 block(C);
    const dim3 grid(H, B);

    backward_kernel<<<grid, block, 0, stream>>>(
        T, H,
        w.typed_data(), q.typed_data(), k.typed_data(), 
        v.typed_data(),z.typed_data(), a.typed_data(),
        dy.typed_data(),s.typed_data(), sa.typed_data(),
        dht.typed_data(), dh0->typed_data(),dw->typed_data(), 
        dq->typed_data(), dk->typed_data(), dv->typed_data(),
        da->typed_data(), db->typed_data());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA backward_kernel error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Bwd, WKV7BwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()  // stream
        .Arg<ffi::Buffer<ffi::F32>>()   // w
        .Arg<ffi::Buffer<ffi::F32>>()   // q
        .Arg<ffi::Buffer<ffi::F32>>()   // k
        .Arg<ffi::Buffer<ffi::F32>>()   // v
        .Arg<ffi::Buffer<ffi::F32>>()   // z  -> kernel a_
        .Arg<ffi::Buffer<ffi::F32>>()   // a  -> kernel b_
        .Arg<ffi::Buffer<ffi::F32>>()   // dy
        .Arg<ffi::Buffer<ffi::F32>>()   // s
        .Arg<ffi::Buffer<ffi::F32>>()   // sa
        .Arg<ffi::Buffer<ffi::F32>>()   // dht
        .Ret<ffi::Buffer<ffi::F32>>()   // dh0
        .Ret<ffi::Buffer<ffi::F32>>()   // dw
        .Ret<ffi::Buffer<ffi::F32>>()   // dq
        .Ret<ffi::Buffer<ffi::F32>>()   // dk
        .Ret<ffi::Buffer<ffi::F32>>()   // dv
        .Ret<ffi::Buffer<ffi::F32>>()   // da
        .Ret<ffi::Buffer<ffi::F32>>()   // db
    ,{xla::ffi::Traits::kCmdBufferCompatible});  // cudaGraph enabled

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include <vector>
#include <cstdint>
// ref link:https://github.com/BlinkDL/RWKV-CUDA/tree/main/rwkv7_fast_fused
namespace ffi = xla::ffi;

/* -------------------- 类型别名 -------------------- */
using bf = __nv_bfloat16;

/* -------------------- 设备端辅助 -------------------- */
__device__ inline float to_float(const bf &u) {
    return __bfloat162float(u);
}
__device__ inline bf to_bf(const float &u) {
    return __float2bfloat16_rn(u);
}
typedef bf *__restrict__ F_;

/* -------------------- Kernel -------------------- */
// 【优化1】模板化 + launch_bounds，提升 Occupancy
template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel(int T, int H,
                               F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
                               bf *y_, float *s_, float *sa_, float *h0_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C; 
    
    #pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];

    for (int t = 0; t < T; ++t) {
        // 【优化2】强制 int64_t 防止溢出
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
            int64_t base = ((int64_t)bb * H + hh) * (T / _CHUNK_LEN_) * C * C +
                           ((int64_t)t / _CHUNK_LEN_) * C * C + i;
            #pragma unroll
            for (int j = 0; j < C; ++j) s_[base + j * C] = state[j];
        }
    }
}

// 【优化3】反向 Kernel：模板化 + launch_bounds + float4 向量加载
template<int C> __launch_bounds__(C, 2)
__global__ void backward_kernel(int T, int H,
                                F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_, F_ dy_,
                                float *s_, float *sa_, float *dht_, float *dh0_,
                                bf *dw_, bf *dq_, bf *dk_, bf *dv_, bf *da_, bf *db_) {
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};
    
    int64_t dht_base = ((int64_t)bb * H + hh) * C * C + i * C;

    #pragma unroll
    for (int j = 0; j < C; ++j) {
        dstate[j]  = dht_[dht_base + j];
        dstateT[j] = dht_[dht_base + j];
    }
    __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C], dSb_shared[C];
    float qi, wi, ki, ai, bi, dyi;

    for (int t = T - 1; t >= 0; --t) {
        int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
        
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

        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int64_t base = ((int64_t)bb * H + hh) * (T / _CHUNK_LEN_) * C * C +
                           ((int64_t)t / _CHUNK_LEN_) * C * C + i * C;
            
            // 【优化4】float4 向量加载，带宽利用率提升 4倍
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
        }
        
        float dq_val = 0.f;
        #pragma unroll
        for (int j = 0; j < C; ++j) dq_val += stateT[j] * dy[j];
        dq_[ind] = to_bf(dq_val);

        float iwi = 1.f / (wi + 1e-6f);
        #pragma unroll
        for (int j = 0; j < C; ++j) {
            stateT[j] = (stateT[j] - ki * v[j] - bi * sa[j]) * iwi;
            dstate[j] += dyi * q[j];
            dstateT[j] += qi * dy[j];
        }
        
        float dw = 0.f, dk = 0.f, dv = 0.f, db = 0.f, dSb = 0.f;
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
        
        float da = 0.f;
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

/* -------------------- 推理专用 Kernel -------------------- */
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

/* -------------------- Host 函数（参数名已统一） -------------------- */
static ffi::Error WKV7FwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> w,
    ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k,
    ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a,  // 原'z'，直接对应 kernel 的 a_
    ffi::Buffer<ffi::BF16> b,  // 原'a'，直接对应 kernel 的 b_
    ffi::Buffer<ffi::F32>  h0,
    ffi::ResultBuffer<ffi::BF16> y,
    ffi::ResultBuffer<ffi::F32>  s,
    ffi::ResultBuffer<ffi::F32>  sa)
{
    constexpr int C = _C_;
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2];
    dim3 block(C);
    dim3 grid(H, B);

    // 【关键】模板实例化调用，参数直接映射
    forward_kernel<_C_><<<grid, block, 0, stream>>>(
        T, H,
        reinterpret_cast<bf *>(w.typed_data()),
        reinterpret_cast<bf *>(q.typed_data()),
        reinterpret_cast<bf *>(k.typed_data()),
        reinterpret_cast<bf *>(v.typed_data()),
        reinterpret_cast<bf *>(a.typed_data()),  // 直接映射到 a_
        reinterpret_cast<bf *>(b.typed_data()),  // 直接映射到 b_
        reinterpret_cast<bf *>(y->typed_data()),
        s->typed_data(),
        sa->typed_data(),
        h0.typed_data());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA forward_kernel error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

static ffi::Error WKV7BwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> w,
    ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k,
    ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a,  // 原'z'，直接对应 kernel 的 a_
    ffi::Buffer<ffi::BF16> b,  // 原'a'，直接对应 kernel 的 b_
    ffi::Buffer<ffi::BF16> dy,
    ffi::Buffer<ffi::F32>  s,
    ffi::Buffer<ffi::F32>  sa,
    ffi::Buffer<ffi::F32>  dht,
    ffi::ResultBuffer<ffi::F32> dh0,
    ffi::ResultBuffer<ffi::BF16> dw,
    ffi::ResultBuffer<ffi::BF16> dq,
    ffi::ResultBuffer<ffi::BF16> dk,
    ffi::ResultBuffer<ffi::BF16> dv,
    ffi::ResultBuffer<ffi::BF16> da,
    ffi::ResultBuffer<ffi::BF16> db)
{
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2];
    constexpr int C = _C_;
    dim3 block(C);
    dim3 grid(H, B);

    // 【关键】模板实例化调用，参数直接映射
    backward_kernel<_C_><<<grid, block, 0, stream>>>(
        T, H,
        reinterpret_cast<bf *>(w.typed_data()),
        reinterpret_cast<bf *>(q.typed_data()),
        reinterpret_cast<bf *>(k.typed_data()),
        reinterpret_cast<bf *>(v.typed_data()),
        reinterpret_cast<bf *>(a.typed_data()),  // 直接映射到 a_
        reinterpret_cast<bf *>(b.typed_data()),  // 直接映射到 b_
        reinterpret_cast<bf *>(dy.typed_data()),
        s.typed_data(),
        sa.typed_data(),
        dht.typed_data(),
        dh0->typed_data(),
        reinterpret_cast<bf *>(dw->typed_data()),
        reinterpret_cast<bf *>(dq->typed_data()),
        reinterpret_cast<bf *>(dk->typed_data()),
        reinterpret_cast<bf *>(dv->typed_data()),
        reinterpret_cast<bf *>(da->typed_data()),
        reinterpret_cast<bf *>(db->typed_data()));

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA backward_kernel error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

static ffi::Error WKV7InferenceHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> w,
    ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k,
    ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a,  // 直接对应 kernel 的 a_
    ffi::Buffer<ffi::BF16> b,  // 直接对应 kernel 的 b_
    ffi::Buffer<ffi::F32>  h0,
    ffi::ResultBuffer<ffi::BF16> y,
    ffi::ResultBuffer<ffi::F32>  s)
{
    constexpr int C = _C_;
    auto dims = w.dimensions();
    int B = dims[0], T = dims[1], H = dims[2];
    dim3 block(C);
    dim3 grid(H, B);

    // 【关键】模板实例化调用，参数直接映射
    forward_inference_kernel<_C_><<<grid, block, 0, stream>>>(
        T, H,
        reinterpret_cast<bf *>(w.typed_data()),
        reinterpret_cast<bf *>(q.typed_data()),
        reinterpret_cast<bf *>(k.typed_data()),
        reinterpret_cast<bf *>(v.typed_data()),
        reinterpret_cast<bf *>(a.typed_data()),  // 直接映射到 a_
        reinterpret_cast<bf *>(b.typed_data()),  // 直接映射到 b_
        reinterpret_cast<bf *>(y->typed_data()),
        s->typed_data(),
        h0.typed_data());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA forward_inference_kernel error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

/* -------------------- FFI 注册（参数名已对齐） -------------------- */
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Fwd, WKV7FwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // q
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // a (原z)
        .Arg<ffi::Buffer<ffi::BF16>>()   // b (原a)
        .Arg<ffi::Buffer<ffi::F32>>()    // h0
        .Ret<ffi::Buffer<ffi::BF16>>()   // y
        .Ret<ffi::Buffer<ffi::F32>>()    // s
        .Ret<ffi::Buffer<ffi::F32>>()    // sa
, {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Bwd, WKV7BwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // q
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // a (原z)
        .Arg<ffi::Buffer<ffi::BF16>>()   // b (原a)
        .Arg<ffi::Buffer<ffi::BF16>>()   // dy
        .Arg<ffi::Buffer<ffi::F32>>()    // s
        .Arg<ffi::Buffer<ffi::F32>>()    // sa
        .Arg<ffi::Buffer<ffi::F32>>()    // dht
        .Ret<ffi::Buffer<ffi::F32>>()   // dh0
        .Ret<ffi::Buffer<ffi::BF16>>()   // dw
        .Ret<ffi::Buffer<ffi::BF16>>()   // dq
        .Ret<ffi::Buffer<ffi::BF16>>()   // dk
        .Ret<ffi::Buffer<ffi::BF16>>()   // dv
        .Ret<ffi::Buffer<ffi::BF16>>()   // da
        .Ret<ffi::Buffer<ffi::BF16>>()   // db
, {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7Inference, WKV7InferenceHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // q
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // a
        .Arg<ffi::Buffer<ffi::BF16>>()   // b
        .Arg<ffi::Buffer<ffi::F32>>()    // h0
        .Ret<ffi::Buffer<ffi::BF16>>()   // y
        .Ret<ffi::Buffer<ffi::F32>>()    // s (final state)
, {ffi::Traits::kCmdBufferCompatible});
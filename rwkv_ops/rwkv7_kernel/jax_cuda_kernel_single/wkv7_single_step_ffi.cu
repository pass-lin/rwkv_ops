#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include <vector>
#include <cstdint>

namespace ffi = xla::ffi;
using bf = __nv_bfloat16;

/* -------------------- 设备端辅助 -------------------- */
__device__ inline float to_float(const bf &u) {
    return __bfloat162float(u);
}
__device__ inline bf to_bf(const float &u) {
    return __float2bfloat16_rn(u);
}
typedef bf *__restrict__ F_;

/* -------------------- 前向 Kernel（修复） -------------------- */
template<int C> 
__launch_bounds__(C, 2)
__global__ void forward_kernel_single_step(
    int B, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    bf *y_, float *s_, float *h0_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    
    // 加载初始状态 (B, H, C, C)
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; ++j) state[j] = h0_[h0_base + j];

    // 单步索引: (B, H, C)
    int64_t ind = (int64_t)bb * H * C + hh * C + i;
    
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

    // 写入最终状态
    int64_t s_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; ++j) s_[s_base + j] = state[j];
}

/* -------------------- 反向 Kernel（补充） -------------------- */
template<int C> 
__launch_bounds__(C, 2) 
__global__ void backward_kernel_single_step(
    int B, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ dy_,
    float *s_, float *dht_, bf *dw_, bf *dq_, bf *dk_, bf *dv_, bf *da_, bf *db_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float stateT[C] = {0}, dstate[C] = {0};
    
    int64_t dht_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; ++j) dstate[j] = dht_[dht_base + j];

    __shared__ float w[C], q[C], k[C], v[C], dy[C];
    int64_t ind = (int64_t)bb * H * C + hh * C + i;
    
    __syncthreads();
    q[i] = to_float(q_[ind]);
    float wi_fac = -__expf(to_float(w_[ind]));
    w[i] = __expf(wi_fac);
    k[i] = to_float(k_[ind]);
    v[i] = to_float(v_[ind]);
    dy[i] = to_float(dy_[ind]);
    __syncthreads();

    // 从 s_ 加载 stateT（float4 优化可在此处添加）
    int64_t s_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
    for (int j = 0; j < C; ++j) stateT[j] = s_[s_base + j];

    float dq_val = 0.f, dw_val = 0.f, dk_val = 0.f, dv_val = 0.f, da_val = 0.f, db_val = 0.f;
    float iwi = 1.0f / (w[i] + 1e-6f);
    
#pragma unroll
    for (int j = 0; j < C; ++j) {
        stateT[j] = (stateT[j] - k[i] * v[j]) * iwi;
        dstate[j] += dy[i] * q[j];
        
        dq_val += stateT[j] * dy[j];
        dw_val += dstate[j] * stateT[j];
        dk_val += dstate[j] * v[j];
        dv_val += dstate[j] * k[j];
    }
    
    dq_[ind] = to_bf(dq_val);
    dw_[ind] = to_bf(dw_val * w[i] * wi_fac);
    dk_[ind] = to_bf(dk_val);
    dv_[ind] = to_bf(dv_val);
}

/* -------------------- Host 函数（修复调用） -------------------- */
static ffi::Error WKV7SingleStepFwdHost(
    cudaStream_t stream,
    ffi::Buffer<ffi::BF16> w,
    ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k,
    ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a,
    ffi::Buffer<ffi::BF16> b,
    ffi::Buffer<ffi::F32>  h0,
    ffi::ResultBuffer<ffi::BF16> y,
    ffi::ResultBuffer<ffi::F32>  s)
{
    auto dims = w.dimensions();
    int B = dims[0], H = dims[1];
    constexpr int C = _C_;  // 从编译选项获取
    dim3 block(C);
    dim3 grid(H, B);

    // ✅ 修复：显式指定模板参数 <_C_>
    forward_kernel_single_step<_C_><<<grid, block, 0, stream>>>(
        B, H,
        reinterpret_cast<bf *>(w.typed_data()),
        reinterpret_cast<bf *>(q.typed_data()),
        reinterpret_cast<bf *>(k.typed_data()),
        reinterpret_cast<bf *>(v.typed_data()),
        reinterpret_cast<bf *>(a.typed_data()),
        reinterpret_cast<bf *>(b.typed_data()),
        reinterpret_cast<bf *>(y->typed_data()),
        s->typed_data(),
        h0.typed_data());

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return ffi::Error::Internal(
            std::string("CUDA forward_kernel_single_step error: ") + cudaGetErrorString(err));
    return ffi::Error::Success();
}

/* -------------------- FFI 符号注册 -------------------- */
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv7SingleStepFwd, WKV7SingleStepFwdHost,
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
        .Ret<ffi::Buffer<ffi::F32>>()    // s
, {ffi::Traits::kCmdBufferCompatible});
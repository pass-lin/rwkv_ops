#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include <vector>
#include <cstdint>

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

/* -------------------- 优化后的单步前向 Kernel -------------------- */
// 【优化1】模板化 + launch_bounds，提升 Occupancy
// 【优化2】float4 向量加载/存储，带宽利用率提升 4 倍
template<int C> __launch_bounds__(C, 2)
__global__ void forward_kernel_single_step(
    int B, int H,
    F_ w_, F_ q_, F_ k_, F_ v_, F_ a_, F_ b_,
    bf *y_, float *s_, float *h0_)
{
    int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
    float state[C] = {0};
    __shared__ float q[C], k[C], w[C], a[C], b[C];
    
    // 【优化3】使用 float4 加载初始状态 (B, H, C, C)
    int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
    const float4* h04 = reinterpret_cast<const float4*>(h0_ + h0_base);
    #pragma unroll
    for (int j4 = 0; j4 < C / 4; ++j4) {
        const int j = j4 * 4;
        float4 val = h04[j4];
        state[j]     = val.x;
        state[j + 1] = val.y;
        state[j + 2] = val.z;
        state[j + 3] = val.w;
    }
    // 处理 C 不是 4 的倍数的情况（安全兜底）
    #pragma unroll
    for (int j = (C / 4) * 4; j < C; ++j) {
        state[j] = h0_[h0_base + j];
    }

    // 单步索引: (B, H, C)
    int64_t ind = (int64_t)bb * H * C + hh * C + i;
    
    __syncthreads();
    q[i] = to_float(q_[ind]);
    w[i] = __expf(-__expf(to_float(w_[ind])));
    k[i] = to_float(k_[ind]);
    a[i] = to_float(a_[ind]);
    b[i] = to_float(b_[ind]);
    __syncthreads();

    // 实时计算 sa，不存储
    float sa = 0.f;
    #pragma unroll
    for (int j = 0; j < C; ++j) sa += a[j] * state[j];

    // 状态更新与输出计算
    float v_val = to_float(v_[ind]);
    float y = 0.f;
    #pragma unroll
    for (int j = 0; j < C; ++j) {
        float &s = state[j];
        s = s * w[j] + sa * b[j] + k[j] * v_val;
        y += s * q[j];
    }
    y_[ind] = to_bf(y);  // y 输出形状: (B, H, C)

    // ✅ 修复：使用 float4 存储最终状态 (B, H, C, C)
    int64_t s_base = ((int64_t)bb * H + hh) * C * C + i * C;
    float4* s4 = reinterpret_cast<float4*>(s_ + s_base);
    #pragma unroll
    for (int j4 = 0; j4 < C / 4; ++j4) {
        const int j = j4 * 4;
        s4[j4] = make_float4(state[j], state[j + 1], state[j + 2], state[j + 3]);
    }
    // 处理 C 不是 4 的倍数的情况（安全兜底）
    #pragma unroll
    for (int j = (C / 4) * 4; j < C; ++j) {
        s_[s_base + j] = state[j];
    }
}

/* -------------------- Host 函数（参数名已统一） -------------------- */
static ffi::Error WKV7SingleStepFwdHost(
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
    auto dims = w.dimensions();
    int B = dims[0], H = dims[1];
    constexpr int C = _C_;
    dim3 block(C);
    dim3 grid(H, B);

    // 【关键】模板实例化调用
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

/* -------------------- FFI 符号注册（参数名已对齐） -------------------- */
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
// RWKV-6 JAX FFI CUDA kernel。

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <xla/ffi/api/ffi.h>
#include <cassert>
#include <cstdint>

namespace ffi = xla::ffi;

using bf16 = __nv_bfloat16;
using fp16 = __half;
using fp32 = float;

static_assert(_N_ % 4 == 0, "the size of head must be the times of 4.");

namespace {

template <typename F>
__device__ __forceinline__ float to_float(const F &x) {
    return static_cast<float>(x);
}
template <typename F>
__device__ __forceinline__ F from_float(float x) {
    return F(x);
}

// RWKV-6 前向核心：每个 thread 负责 head 维度的一个位置 i，沿 T 扫描。
//
// Args:
//   state: 当前 thread 的 _N_ 维状态数组（寄存器）。
//   _r/_k/_v/_w/_u: [B, H, T, N]（row-major，T 为慢变维），bfloat16。
//   _y: [B, H, T, N]，bfloat16。输出。
//
// 编译期宏:
//   _N_: head size，必须被 4 整除。
template <typename F_in, typename F_out>
__device__ void kernel_forward_core(const int B, const int T, const int C, const int H,
                                    const int b, const int h, const int i, const float *state,
                                    const F_in *__restrict__ const _r,
                                    const F_in *__restrict__ const _k,
                                    const F_in *__restrict__ const _v,
                                    const F_in *__restrict__ _w,
                                    const F_in *__restrict__ _u,
                                    F_out *__restrict__ const _y)
{
    _u += h * _N_;

    __shared__ float r[_N_], k[_N_], u[_N_], w[_N_];

    __syncthreads();
    u[i] = to_float(_u[i]);
    __syncthreads();

    for (int t = b * T * C + h * _N_ + i; t < (b + 1) * T * C + h * _N_ + i; t += C)
    {
        __syncthreads();
        w[i] = __expf(-__expf(to_float(_w[t])));
        r[i] = to_float(_r[t]);
        k[i] = to_float(_k[t]);
        __syncthreads();

        const float v = to_float(_v[t]);
        float y = 0;

#pragma unroll
        for (int j = 0; j < _N_; j += 4)
        {
            const float4 &r_ = (float4 &)(r[j]);
            const float4 &k_ = (float4 &)(k[j]);
            const float4 &w_ = (float4 &)(w[j]);
            const float4 &u_ = (float4 &)(u[j]);
            float4 &s = (float4 &)(state[j]);
            float4 x;

            x.x = k_.x * v;
            x.y = k_.y * v;
            x.z = k_.z * v;
            x.w = k_.w * v;

            y += r_.x * (u_.x * x.x + s.x);
            y += r_.y * (u_.y * x.y + s.y);
            y += r_.z * (u_.z * x.z + s.z);
            y += r_.w * (u_.w * x.w + s.w);

            s.x = s.x * w_.x + x.x;
            s.y = s.y * w_.y + x.y;
            s.z = s.z * w_.z + x.z;
            s.w = s.w * w_.w + x.w;
        }
        _y[t] = from_float<F_out>(y);
    }
}

// RWKV-6 前向（带初始/最终状态）。
//
// grid = (B * H, 1, 1)，block = (_N_, 1, 1)。
// 每个 block 处理一个 (batch, head)，最终写回 final_state。
template <typename F_in, typename F_out>
__global__ void kernel_forward_state(const int B, const int T, const int C, const int H,
                                     const bool is_custom_state, const int32_t *map,
                                     const F_in *__restrict__ const _r,
                                     const F_in *__restrict__ const _k,
                                     const F_in *__restrict__ const _v,
                                     const F_in *__restrict__ _w,
                                     const F_in *__restrict__ _u,
                                     const F_out *__restrict__ _s,
                                     F_out *__restrict__ const _y,
                                     F_out *__restrict__ const _ys)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    float state[_N_] = {0};
    if (is_custom_state)
    {
        assert(map[b] >= 0 && map[b] < B);
        const int64_t input_state_offset = map[b] * H * _N_ * _N_ + h * _N_ * _N_ + i;
        for (int j = 0; j < _N_; j++)
        {
            state[j] = to_float(_s[j * _N_ + input_state_offset]);
        }
    }

    const int64_t current_state_offset = b * H * _N_ * _N_ + h * _N_ * _N_ + i;

    kernel_forward_core(B, T, C, H, b, h, i, state, _r, _k, _v, _w, _u, _y);
    for (int j = 0; j < _N_; j++)
    {
        _ys[j * _N_ + current_state_offset] = from_float<F_out>(state[j]);
    }
}

// RWKV-6 前向（无状态）。
//
// grid = (B * H, 1, 1)，block = (_N_, 1, 1)。
template <typename F_in, typename F_out>
__global__ void kernel_forward(const int B, const int T, const int C, const int H,
                               const F_in *__restrict__ const _r,
                               const F_in *__restrict__ const _k,
                               const F_in *__restrict__ const _v,
                               const F_in *__restrict__ _w,
                               const F_in *__restrict__ _u,
                               F_out *__restrict__ const _y)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    float state[_N_] = {0};
    kernel_forward_core(B, T, C, H, b, h, i, state, _r, _k, _v, _w, _u, _y);
}

// RWKV-6 反向：计算 gr 与 gu。
//
// 沿时间正序扫描，复用前向状态递推。_gu 按 batch/head/channel 写出后，
// Python 侧再按 batch 求和 reshape 为 (H, N)。
template <typename F_in, typename F_out>
__global__ void kernel_backward_101(const int B, const int T, const int C, const int H,
                                    const F_in *__restrict__ const _r,
                                    const F_in *__restrict__ const _k,
                                    const F_in *__restrict__ const _v,
                                    const F_in *__restrict__ _w,
                                    const F_in *__restrict__ _u,
                                    const F_out *__restrict__ const _gy,
                                    F_out *__restrict__ const _gr,
                                    F_out *__restrict__ const _gu)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    const float u = to_float(_u[h * _N_ + i]);
    float state[_N_] = {0};

    const int t_0 = b * T * C + h * _N_ + i;
    const int t_T = t_0 + T * C;

    float gu = 0;
    for (int t = t_0; t < t_T; t += C)
    {
        __syncthreads();
        v[i] = to_float(_v[t]);
        gy[i] = to_float(_gy[t]);
        __syncthreads();

        const float k = to_float(_k[t]);
        const float w = __expf(-__expf(to_float(_w[t])));
        float gr = 0, gu_ = 0;

#pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = state[j];
            float x = k * v[j];

            gr += (u * x + s) * gy[j];
            gu_ += x * gy[j];
            s = s * w + x;
        }
        _gr[t] = from_float<F_out>(gr);
        gu += to_float(_r[t]) * gu_;
    }
    _gu[b * C + h * _N_ + i] = from_float<F_out>(gu);
}

// RWKV-6 反向：计算 gk。
//
// 沿时间逆序扫描，用独立状态数组 scccc 递推。指针算术使用 int64_t，
// 避免大 tensor 时 32 位偏移溢出。
template <typename F_in, typename F_out>
__global__ void kernel_backward_102(const int B, const int T, const int C, const int H,
                                    const F_in *__restrict__ const _r,
                                    const F_in *__restrict__ const _k,
                                    const F_in *__restrict__ const _v,
                                    const F_in *__restrict__ _w,
                                    const F_in *__restrict__ _u,
                                    const F_out *__restrict__ const _gy,
                                    F_out *__restrict__ const _gk)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    const float u = to_float(_u[h * _N_ + i]);
    float scccc[_N_] = {0};

    const int t_0 = b * T * C + h * _N_ + i;
    const int t_T_1 = t_0 + (T - 1) * C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        v[i] = to_float(_v[t]);
        gy[i] = to_float(_gy[t]);
        __syncthreads();

        const float rr = to_float(_r[t]);
        const float w = __expf(-__expf(to_float(_w[t])));
        float gk = 0;

#pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = scccc[j];
            float x = rr * gy[j];

            gk += (u * x + s) * v[j];
            s = x + s * w;
        }
        _gk[t] = from_float<F_out>(gk);
    }
}

// RWKV-6 反向：计算 gv。
//
// 沿时间逆序扫描。_u 指针先偏移到当前 head 起点。
template <typename F_in, typename F_out>
__global__ void kernel_backward_103(const int B, const int T, const int C, const int H,
                                    const F_in *__restrict__ const _r,
                                    const F_in *__restrict__ const _k,
                                    const F_in *__restrict__ const _v,
                                    const F_in *__restrict__ _w,
                                    const F_in *__restrict__ _u,
                                    const F_out *__restrict__ const _gy,
                                    F_out *__restrict__ const _gv)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;
    _u += h * _N_;

    __shared__ float u_[_N_], r[_N_], k[_N_], w_[_N_];
    __syncthreads();
    u_[i] = to_float(_u[i]);
    __syncthreads();

    float sdddd[_N_] = {0};

    const int t_0 = b * T * C + h * _N_ + i;
    const int t_T_1 = t_0 + (T - 1) * C;

    for (int t = t_T_1; t >= t_0; t -= C)
    {
        __syncthreads();
        r[i] = to_float(_r[t]);
        k[i] = to_float(_k[t]);
        w_[i] = __expf(-__expf(to_float(_w[t])));
        __syncthreads();

        const float gyy = to_float(_gy[t]);
        float gv = 0;

#pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = sdddd[j];
            float x = gyy * r[j];

            gv += (u_[j] * x + s) * k[j];
            s = x + s * w_[j];
        }
        _gv[t] = from_float<F_out>(gv);
    }
}

// RWKV-6 反向：计算 gw。
//
// 沿时间双向扫描：先逆序累加 sbbbb 缓存，再正序组合得到 gw。
// 数组 sbbbb 的大小依赖编译期 _T_。
template <typename F_in, typename F_out>
__global__ void kernel_backward_201(const int B, const int T, const int C, const int H,
                                    const F_in *__restrict__ const _r,
                                    const F_in *__restrict__ const _k,
                                    const F_in *__restrict__ const _v,
                                    const F_in *__restrict__ _w,
                                    const F_in *__restrict__ _u,
                                    const F_out *__restrict__ const _gy,
                                    F_out *__restrict__ const _gw)
{
    const int b = blockIdx.x / H;
    const int h = blockIdx.x % H;
    const int i = threadIdx.x;

    __shared__ float v[_N_], gy[_N_];
    float saaaa[_N_] = {0}, sbbbb[_T_ - 2 > 0 ? _T_ - 2 : 1] = {0}, scccc[_N_] = {0};

    const int t_0 = b * T * C + h * _N_ + i;
    const int t_1 = t_0 + C;
    const int t_2 = t_0 + 2 * C;
    const int t_T_1 = t_0 + (T - 1) * C;

    for (int t = t_T_1; t > t_1; t -= C)
    {
        __syncthreads();
        gy[i] = to_float(_gy[t]);
        v[i] = to_float(_v[t - 2 * C]);
        __syncthreads();

        const float r = to_float(_r[t]);
        const float w = __expf(-__expf(to_float(_w[t - C])));
        float sum = 0.0f;

#pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = saaaa[j];
            float x = r * gy[j];
            s = (s + x) * w;
            sum += s * v[j];
        }
        sbbbb[(t - t_2) / C] = sum * to_float(_k[t - 2 * C]);
    }

    float sss = sbbbb[0];
    _gw[t_0] = from_float<F_out>(0.0f);
    _gw[t_1] = from_float<F_out>(sss * -__expf(to_float(_w[t_1])));

    for (int t = t_2; t < t_T_1; t += C)
    {
        __syncthreads();
        gy[i] = to_float(_gy[t]);
        v[i] = to_float(_v[t - 2 * C]);
        __syncthreads();

        const float w = __expf(-__expf(to_float(_w[t - C])));
        const float k = to_float(_k[t - 2 * C]);
        float sum = 0.0f;

#pragma unroll
        for (int j = 0; j < _N_; j++)
        {
            float &s = scccc[j];
            float x = k * v[j];
            s = (s + x) * w;
            sum += s * gy[j];
        }
        sss += sbbbb[(t - t_1) / C] - (sum * to_float(_r[t]));
        _gw[t] = from_float<F_out>(sss * -__expf(to_float(_w[t])));
    }
    _gw[t_T_1] = from_float<F_out>(0.0f);
}

// Host 启动函数：带初始/最终状态前向。
template <typename T_in, typename T_out>
void HostApplyRWKVWithState(cudaStream_t stream, int B, int T, int C, int H, bool S,
                            const int32_t *state_map,
                            const T_in *input_r, const T_in *input_k, const T_in *input_v,
                            const T_in *input_w, const T_in *input_u,
                            const T_out *input_s, T_out *output_y, T_out *output_s)
{
    assert(H * _N_ == C);
    kernel_forward_state<<<dim3(B * H), dim3(_N_), _N_ * 4 * sizeof(float), stream>>>(
        B, T, C, H, S, state_map, input_r, input_k, input_v, input_w, input_u, input_s,
        output_y, output_s);
}

// Host 启动函数：无状态前向。
template <typename T_in, typename T_out>
void HostApplyRWKV(cudaStream_t stream, int B, int T, int C, int H,
                   const T_in *input_r, const T_in *input_k, const T_in *input_v,
                   const T_in *input_w, const T_in *input_u, T_out *output_y)
{
    assert(H * _N_ == C);
    kernel_forward<<<dim3(B * H), dim3(_N_), _N_ * 4 * sizeof(float), stream>>>(
        B, T, C, H, input_r, input_k, input_v, input_w, input_u, output_y);
}

// Host 启动函数：反向，启动 4 个 kernel 分别计算 gr/gu、gk、gv、gw。
template <typename T_in, typename T_out>
void HostApplyGradient(cudaStream_t stream, int B, int T, int C, int H,
                       const T_in *r, const T_in *k, const T_in *v, const T_in *w,
                       const T_in *u, const T_out *gy, T_out *gr, T_out *gk, T_out *gv,
                       T_out *gw, T_out *gu)
{
    assert(H * _N_ == C);
    kernel_backward_101<<<dim3(B * H), dim3(_N_), _N_ * 2 * sizeof(float), stream>>>(
        B, T, C, H, r, k, v, w, u, gy, gr, gu);
    kernel_backward_102<<<dim3(B * H), dim3(_N_), _N_ * 2 * sizeof(float), stream>>>(
        B, T, C, H, r, k, v, w, u, gy, gk);
    kernel_backward_103<<<dim3(B * H), dim3(_N_), _N_ * 4 * sizeof(float), stream>>>(
        B, T, C, H, r, k, v, w, u, gy, gv);
    kernel_backward_201<<<dim3(B * H), dim3(_N_), _N_ * 2 * sizeof(float), stream>>>(
        B, T, C, H, r, k, v, w, u, gy, gw);
}

} // namespace

//  XLA FFI Handlers 

// 仅实现 bf16 路径；其它 dtype 通过 native_keras_op 回退。

static ffi::Error Wkv6FwdHost(cudaStream_t stream,
                              ffi::Buffer<ffi::BF16> r,
                              ffi::Buffer<ffi::BF16> k,
                              ffi::Buffer<ffi::BF16> v,
                              ffi::Buffer<ffi::BF16> w,
                              ffi::Buffer<ffi::BF16> u,
                              ffi::ResultBuffer<ffi::BF16> y)
{
    auto dims = r.dimensions();
    int B = dims[0], T = dims[1], C = dims[2];
    int H = C / _N_;
    HostApplyRWKV<bf16, bf16>(
        stream, B, T, C, H,
        reinterpret_cast<bf16 *>(r.typed_data()),
        reinterpret_cast<bf16 *>(k.typed_data()),
        reinterpret_cast<bf16 *>(v.typed_data()),
        reinterpret_cast<bf16 *>(w.typed_data()),
        reinterpret_cast<bf16 *>(u.typed_data()),
        reinterpret_cast<bf16 *>(y->typed_data()));
    return ffi::Error::Success();
}

static ffi::Error Wkv6BwdHost(cudaStream_t stream,
                              ffi::Buffer<ffi::BF16> r,
                              ffi::Buffer<ffi::BF16> k,
                              ffi::Buffer<ffi::BF16> v,
                              ffi::Buffer<ffi::BF16> w,
                              ffi::Buffer<ffi::BF16> u,
                              ffi::Buffer<ffi::BF16> gy,
                              ffi::ResultBuffer<ffi::BF16> gr,
                              ffi::ResultBuffer<ffi::BF16> gk,
                              ffi::ResultBuffer<ffi::BF16> gv,
                              ffi::ResultBuffer<ffi::BF16> gw,
                              ffi::ResultBuffer<ffi::BF16> gu)
{
    auto dims = r.dimensions();
    int B = dims[0], T = dims[1], C = dims[2];
    int H = C / _N_;
    HostApplyGradient<bf16, bf16>(
        stream, B, T, C, H,
        reinterpret_cast<bf16 *>(r.typed_data()),
        reinterpret_cast<bf16 *>(k.typed_data()),
        reinterpret_cast<bf16 *>(v.typed_data()),
        reinterpret_cast<bf16 *>(w.typed_data()),
        reinterpret_cast<bf16 *>(u.typed_data()),
        reinterpret_cast<bf16 *>(gy.typed_data()),
        reinterpret_cast<bf16 *>(gr->typed_data()),
        reinterpret_cast<bf16 *>(gk->typed_data()),
        reinterpret_cast<bf16 *>(gv->typed_data()),
        reinterpret_cast<bf16 *>(gw->typed_data()),
        reinterpret_cast<bf16 *>(gu->typed_data()));
    return ffi::Error::Success();
}

static ffi::Error Wkv6FwdWithStateHost(cudaStream_t stream,
                                       ffi::Buffer<ffi::BF16> r,
                                       ffi::Buffer<ffi::BF16> k,
                                       ffi::Buffer<ffi::BF16> v,
                                       ffi::Buffer<ffi::BF16> w,
                                       ffi::Buffer<ffi::BF16> u,
                                       ffi::Buffer<ffi::S32> state_map,
                                       ffi::Buffer<ffi::BF16> init_state,
                                       ffi::ResultBuffer<ffi::BF16> y,
                                       ffi::ResultBuffer<ffi::BF16> final_state)
{
    auto dims = r.dimensions();
    int B = dims[0], T = dims[1], C = dims[2];
    int H = C / _N_;
    bool is_custom_state = init_state.element_count() > 0;
    HostApplyRWKVWithState<bf16, bf16>(
        stream, B, T, C, H, is_custom_state,
        const_cast<int32_t *>(state_map.typed_data()),
        reinterpret_cast<bf16 *>(r.typed_data()),
        reinterpret_cast<bf16 *>(k.typed_data()),
        reinterpret_cast<bf16 *>(v.typed_data()),
        reinterpret_cast<bf16 *>(w.typed_data()),
        reinterpret_cast<bf16 *>(u.typed_data()),
        reinterpret_cast<bf16 *>(init_state.typed_data()),
        reinterpret_cast<bf16 *>(y->typed_data()),
        reinterpret_cast<bf16 *>(final_state->typed_data()));
    return ffi::Error::Success();
}

//  FFI 注册 
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv6Fwd, Wkv6FwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // r
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // u
        .Ret<ffi::Buffer<ffi::BF16>>()   // y
, {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv6Bwd, Wkv6BwdHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // r
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // u
        .Arg<ffi::Buffer<ffi::BF16>>()   // gy
        .Ret<ffi::Buffer<ffi::BF16>>()   // gr
        .Ret<ffi::Buffer<ffi::BF16>>()   // gk
        .Ret<ffi::Buffer<ffi::BF16>>()   // gv
        .Ret<ffi::Buffer<ffi::BF16>>()   // gw
        .Ret<ffi::Buffer<ffi::BF16>>()   // gu
, {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    Wkv6FwdWithState, Wkv6FwdWithStateHost,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::Buffer<ffi::BF16>>()   // r
        .Arg<ffi::Buffer<ffi::BF16>>()   // k
        .Arg<ffi::Buffer<ffi::BF16>>()   // v
        .Arg<ffi::Buffer<ffi::BF16>>()   // w
        .Arg<ffi::Buffer<ffi::BF16>>()   // u
        .Arg<ffi::Buffer<ffi::S32>>()    // state_map
        .Arg<ffi::Buffer<ffi::BF16>>()   // init_state
        .Ret<ffi::Buffer<ffi::BF16>>()   // y
        .Ret<ffi::Buffer<ffi::BF16>>()   // final_state
, {ffi::Traits::kCmdBufferCompatible});

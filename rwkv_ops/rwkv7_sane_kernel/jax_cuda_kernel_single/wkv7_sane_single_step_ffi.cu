// RWKV-7-SANE JAX FFI 单步 CUDA kernel。

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <vector>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;
using bf = __nv_bfloat16;

__device__ inline float to_float(const bf &u) { return __bfloat162float(u); }
__device__ inline bf to_bf(const float &u) { return __float2bfloat16_rn(u); }
typedef bf *__restrict__ F_;

// RWKV-7-SANE 单步前向 CUDA kernel。
//
// 每个 block 处理一个 (batch, head)，单步完成 delta-rule 更新并可选执行 SANE。
//
// Args:
//   w, q, k, v, a, b: [B, H, C], bfloat16, row-major。
//   tau: [B, H], float32, row-major。阈值，必须 > 0。
//   do_sane: [B], int32, row-major。非 0 表示对该 sample 执行 SANE。
//   y: [B, H, C], bfloat16, row-major。输出。
//   s: [B, H, C, C], float32, row-major。输出 state。
//   h0: [B, H, C, C], float32, row-major。初始 state。
//
// 编译期宏:
//   _C_: head_size，由 -D_C_ 传入。
template <int C>
__launch_bounds__(C, 2) __global__
    void forward_kernel_single_step_sane(int B, int H, F_ w_, F_ q_, F_ k_,
                                         F_ v_, F_ a_, F_ b_,
                                         const float *__restrict__ tau_,
                                         const int32_t *__restrict__ do_sane_,
                                         bf *y_, float *s_, float *h0_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float state[C] = {0};
  __shared__ float q[C], k[C], w[C], a[C], b[C];

  int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    state[j] = h0_[h0_base + j];

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
  for (int j = 0; j < C; ++j)
    sa += a[j] * state[j];

  float v_val = to_float(v_[ind]);
  float y = 0.f;
#pragma unroll
  for (int j = 0; j < C; ++j) {
    float &s = state[j];
    s = s * w[j] + sa * b[j] + k[j] * v_val;
    y += s * q[j];
  }
  y_[ind] = to_bf(y);

  if (do_sane_[bb] != 0) {
    float tau = tau_[bb * H + hh];
    if (tau > 0.0f) {
#pragma unroll
      for (int j = 0; j < C; ++j)
        state[j] = tau * tanhf(state[j] / tau);
    }
  }

  int64_t s_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    s_[s_base + j] = state[j];
}

// Host wrapper for forward_kernel_single_step_sane。
static ffi::Error WKV7SaneSingleStepFwdHost(
    cudaStream_t stream, ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::S32> do_sane,
    ffi::Buffer<ffi::F32> h0, ffi::ResultBuffer<ffi::BF16> y,
    ffi::ResultBuffer<ffi::F32> s) {
  auto dims = w.dimensions();
  int B = dims[0], H = dims[1];
  constexpr int C = _C_;
  dim3 block(C);
  dim3 grid(H, B);

  forward_kernel_single_step_sane<_C_><<<grid, block, 0, stream>>>(
      B, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      do_sane.typed_data(), reinterpret_cast<bf *>(y->typed_data()),
      s->typed_data(), h0.typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA forward_kernel_single_step_sane error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// XLA FFI handler 注册。
XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneSingleStepFwd, WKV7SaneSingleStepFwdHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::S32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

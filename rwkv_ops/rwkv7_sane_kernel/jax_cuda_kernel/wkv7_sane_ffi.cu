// RWKV-7-SANE JAX FFI CUDA kernel。

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

// RWKV-7-SANE 训练前向 CUDA kernel（带 mask）。
//
// 每个 block 处理一个 (batch, head)，顺序扫描 T 步，在每个 chunk 末尾
// 写出 SANE 之前的 state checkpoint。
//
// Args:
//   w, q, k, v, a, b: [B, H, T, C], bfloat16, row-major。
//   tau: [B, H, T//16], float32, row-major。阈值，必须 > 0。
//   mask: [B, T//16], float32, row-major。>0 表示该 chunk 边界执行 SANE。
//   y: [B, H, T, C], bfloat16, row-major。输出。
//   s: [B, H, T//16, C, C], float32, row-major。SANE 之前的 state checkpoint。
//   sa: [B, H, T, C], float32, row-major。反向所需中间量。
//   h0: [B, H, C, C], float32, row-major。初始 state。
//
// 编译期宏:
//   _C_: head_size，由 -D_C_ 传入。
//   _CHUNK_LEN_: chunk 长度，固定 16。
//
// 指针算术一律使用 64 位整数，防止大 tensor 时 32 位偏移溢出。
template <int C>
__launch_bounds__(C, 2) __global__
    void forward_kernel_sane(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_,
                             F_ b_, const float *__restrict__ tau_,
                             const float *__restrict__ mask_, bf *y_, float *s_,
                             float *sa_, float *h0_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float state[C] = {0};
  __shared__ float q[C], k[C], w[C], a[C], b[C];

  int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    state[j] = h0_[h0_base + j];

  const int num_chunks = T / _CHUNK_LEN_;

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
    for (int j = 0; j < C; ++j)
      sa += a[j] * state[j];
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
      int chunk = t / _CHUNK_LEN_;
      int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                     (int64_t)chunk * C * C + i;
#pragma unroll
      for (int j = 0; j < C; ++j)
        s_[base + j * C] = state[j];

      int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
      float m = mask_[cond_idx];
      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
#pragma unroll
      for (int j = 0; j < C; ++j) {
        state[j] = state[j] * (1.0f - m) + m * tau * tanhf(state[j] / tau);
      }
    }
  }
}

// RWKV-7-SANE 训练反向 CUDA kernel（带 mask）。
//
// 每个 block 处理一个 (batch, head)，逆序扫描 T 步，在每个 chunk 边界先算
// dtau， 再对下游梯度乘 sech2。
//
// Args:
//   w, q, k, v, a, b: [B, H, T, C], bfloat16, row-major。前向输入。
//   tau, mask: 同 forward_kernel_sane。
//   dy: [B, H, T, C], bfloat16, row-major。输出梯度。
//   s, sa: [B, H, T//16, C, C] / [B, H, T, C], float32, row-major。前向保存量。
//   dht: [B, H, C, C], float32, row-major。最终 state 梯度。
//   dh0: [B, H, C, C], float32, row-major。初始 state 梯度输出。
//   dtau: [B, H, T//16], float32, row-major。
//   dw/dq/dk/dv/da/db: [B, H, T, C], bfloat16, row-major。输入梯度输出。
//
// 编译期宏同 forward_kernel_sane。
template <int C>
__launch_bounds__(C, 2) __global__
    void backward_kernel_sane(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_, F_ a_,
                              F_ b_, const float *__restrict__ tau_,
                              const float *__restrict__ mask_, F_ dy_,
                              float *s_, float *sa_, float *dht_, float *dh0_,
                              float *dtau_, bf *dw_, bf *dq_, bf *dk_, bf *dv_,
                              bf *da_, bf *db_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};

  int64_t dht_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j) {
    dstate[j] = dht_[dht_base + j];
    dstateT[j] = dht_[dht_base + j];
  }
  __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C],
      dSb_shared[C];
  __shared__ float dtau_shared[C];
  float qi, wi, ki, ai, bi, dyi;

  const int num_chunks = T / _CHUNK_LEN_;

  for (int t = T - 1; t >= 0; --t) {
    int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
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

    if ((t + 1) % _CHUNK_LEN_ == 0) {
      int chunk = t / _CHUNK_LEN_;
      int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                     (int64_t)chunk * C * C + i * C;
      const float4 *s4 = (const float4 *)(s_ + base);
#pragma unroll
      for (int j4 = 0; j4 < C / 4; ++j4) {
        float4 q_vec = s4[j4];
        const int j = j4 * 4;
        stateT[j + 0] = q_vec.x;
        stateT[j + 1] = q_vec.y;
        stateT[j + 2] = q_vec.z;
        stateT[j + 3] = q_vec.w;
      }

      int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
      float m = mask_[cond_idx];
      float inv_tau = 1.0f / tau;
      float dtau_local = 0.0f;
#pragma unroll
      for (int j = 0; j < C; ++j) {
        float u = stateT[j] * inv_tau;
        float tnh = tanhf(u);
        float sech2 = 1.0f - tnh * tnh;
        float blend = (1.0f - m) + m * sech2;
        dtau_local += m * dstate[j] * (tnh - u * sech2);
        dstate[j] *= blend;
        dstateT[j] *= blend;
      }
      dtau_shared[i] = dtau_local;
      __syncthreads();
#pragma unroll
      for (int stride = C / 2; stride > 0; stride /= 2) {
        if (i < stride)
          dtau_shared[i] += dtau_shared[i + stride];
        __syncthreads();
      }
      if (i == 0) {
        dtau_[bb * num_chunks * H + chunk * H + hh] = dtau_shared[0];
      }
    }

    float dq_val = 0.f;
#pragma unroll
    for (int j = 0; j < C; ++j)
      dq_val += stateT[j] * dy[j];
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
    for (int j = 0; j < C; ++j)
      da += stateT[j] * dSb_shared[j];
    da_[ind] = to_bf(da);

#pragma unroll
    for (int j = 0; j < C; ++j) {
      dstate[j] = dstate[j] * w[j] + dSb * a[j];
      dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
      if (t == 0)
        dh0_[dht_base + j] = dstate[j];
    }
  }
}

// RWKV-7-SANE 推理前向 CUDA kernel（带 mask）。
//
// 仅用于 prefill / 纯推理：不保存 s checkpoint 与 sa，减少显存占用。
// T 必须被 _CHUNK_LEN_ 整除；若需任意长度请用单步 kernel。
//
// Args:
//   y: [B, H, T, C], bfloat16, row-major。输出。
//   s: [B, H, C, C], float32, row-major。最终 state。
//   其余同 forward_kernel_sane。
//
// 编译期宏同 forward_kernel_sane。
template <int C>
__launch_bounds__(C, 2) __global__
    void forward_inference_kernel_sane(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_,
                                       F_ a_, F_ b_,
                                       const float *__restrict__ tau_,
                                       const float *__restrict__ mask_, bf *y_,
                                       float *s_, float *h0_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float state[C] = {0};
  __shared__ float q[C], k[C], w[C], a[C], b[C];

  int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    state[j] = h0_[h0_base + j];

  const int num_chunks = T / _CHUNK_LEN_;

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

    if ((t + 1) % _CHUNK_LEN_ == 0) {
      int chunk = t / _CHUNK_LEN_;
      int64_t cond_idx = (int64_t)bb * num_chunks + chunk;
      float m = mask_[cond_idx];
      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
#pragma unroll
      for (int j = 0; j < C; ++j) {
        state[j] = state[j] * (1.0f - m) + m * tau * tanhf(state[j] / tau);
      }
    }
  }

  int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    s_[base + j] = state[j];
}

// Host wrapper for forward_kernel_sane。
static ffi::Error
WKV7SaneFwdHost(cudaStream_t stream, ffi::Buffer<ffi::BF16> w,
                ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k,
                ffi::Buffer<ffi::BF16> v, ffi::Buffer<ffi::BF16> a,
                ffi::Buffer<ffi::BF16> b, ffi::Buffer<ffi::F32> tau,
                ffi::Buffer<ffi::F32> mask, ffi::Buffer<ffi::F32> h0,
                ffi::ResultBuffer<ffi::BF16> y, ffi::ResultBuffer<ffi::F32> s,
                ffi::ResultBuffer<ffi::F32> sa) {
  constexpr int C = _C_;
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  dim3 block(C);
  dim3 grid(H, B);

  forward_kernel_sane<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      mask.typed_data(), reinterpret_cast<bf *>(y->typed_data()),
      s->typed_data(), sa->typed_data(), h0.typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA forward_kernel_sane error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// Host wrapper for backward_kernel_sane。
static ffi::Error WKV7SaneBwdHost(
    cudaStream_t stream, ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::F32> mask,
    ffi::Buffer<ffi::BF16> dy, ffi::Buffer<ffi::F32> s,
    ffi::Buffer<ffi::F32> sa, ffi::Buffer<ffi::F32> dht,
    ffi::ResultBuffer<ffi::F32> dh0, ffi::ResultBuffer<ffi::F32> dtau,
    ffi::ResultBuffer<ffi::BF16> dw, ffi::ResultBuffer<ffi::BF16> dq,
    ffi::ResultBuffer<ffi::BF16> dk, ffi::ResultBuffer<ffi::BF16> dv,
    ffi::ResultBuffer<ffi::BF16> da, ffi::ResultBuffer<ffi::BF16> db) {
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  constexpr int C = _C_;
  dim3 block(C);
  dim3 grid(H, B);

  backward_kernel_sane<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      mask.typed_data(), reinterpret_cast<bf *>(dy.typed_data()),
      s.typed_data(), sa.typed_data(), dht.typed_data(), dh0->typed_data(),
      dtau->typed_data(), reinterpret_cast<bf *>(dw->typed_data()),
      reinterpret_cast<bf *>(dq->typed_data()),
      reinterpret_cast<bf *>(dk->typed_data()),
      reinterpret_cast<bf *>(dv->typed_data()),
      reinterpret_cast<bf *>(da->typed_data()),
      reinterpret_cast<bf *>(db->typed_data()));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA backward_kernel_sane error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// Host wrapper for forward_inference_kernel_sane。
static ffi::Error
WKV7SaneInferenceHost(cudaStream_t stream, ffi::Buffer<ffi::BF16> w,
                      ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k,
                      ffi::Buffer<ffi::BF16> v, ffi::Buffer<ffi::BF16> a,
                      ffi::Buffer<ffi::BF16> b, ffi::Buffer<ffi::F32> tau,
                      ffi::Buffer<ffi::F32> mask, ffi::Buffer<ffi::F32> h0,
                      ffi::ResultBuffer<ffi::BF16> y,
                      ffi::ResultBuffer<ffi::F32> s) {
  constexpr int C = _C_;
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  dim3 block(C);
  dim3 grid(H, B);

  forward_inference_kernel_sane<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      mask.typed_data(), reinterpret_cast<bf *>(y->typed_data()),
      s->typed_data(), h0.typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA forward_inference_kernel_sane error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// RWKV-7-SANE 训练前向 CUDA kernel（无 mask）。
//
// 在 chunk 边界无条件执行 State Anomaly Neutralization。
// Args 与 forward_kernel_sane 相同，但不读 mask。
template <int C>
__launch_bounds__(C, 2) __global__
    void forward_kernel_sane_no_mask(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_,
                                     F_ a_, F_ b_,
                                     const float *__restrict__ tau_, bf *y_,
                                     float *s_, float *sa_, float *h0_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float state[C] = {0};
  __shared__ float q[C], k[C], w[C], a[C], b[C];

  int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    state[j] = h0_[h0_base + j];

  const int num_chunks = T / _CHUNK_LEN_;

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
    for (int j = 0; j < C; ++j)
      sa += a[j] * state[j];
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
      int chunk = t / _CHUNK_LEN_;
      int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                     (int64_t)chunk * C * C + i;
#pragma unroll
      for (int j = 0; j < C; ++j)
        s_[base + j * C] = state[j];

      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
#pragma unroll
      for (int j = 0; j < C; ++j) {
        state[j] = tau * tanhf(state[j] / tau);
      }
    }
  }
}

// RWKV-7-SANE 训练反向 CUDA kernel（无 mask）。
//
// Args 与 backward_kernel_sane 相同，但不读 mask。
template <int C>
__launch_bounds__(C, 2) __global__
    void backward_kernel_sane_no_mask(int T, int H, F_ w_, F_ q_, F_ k_, F_ v_,
                                      F_ a_, F_ b_,
                                      const float *__restrict__ tau_, F_ dy_,
                                      float *s_, float *sa_, float *dht_,
                                      float *dh0_, float *dtau_, bf *dw_,
                                      bf *dq_, bf *dk_, bf *dv_, bf *da_,
                                      bf *db_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float stateT[C] = {0}, dstate[C] = {0}, dstateT[C] = {0};

  int64_t dht_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j) {
    dstate[j] = dht_[dht_base + j];
    dstateT[j] = dht_[dht_base + j];
  }
  __shared__ float w[C], q[C], k[C], v[C], a[C], b[C], dy[C], sa[C],
      dSb_shared[C];
  __shared__ float dtau_shared[C];
  float qi, wi, ki, ai, bi, dyi;

  const int num_chunks = T / _CHUNK_LEN_;

  for (int t = T - 1; t >= 0; --t) {
    int64_t ind = (int64_t)bb * T * H * C + (int64_t)t * H * C + hh * C + i;
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

    if ((t + 1) % _CHUNK_LEN_ == 0) {
      int chunk = t / _CHUNK_LEN_;
      int64_t base = ((int64_t)bb * H + hh) * num_chunks * C * C +
                     (int64_t)chunk * C * C + i * C;
      const float4 *s4 = (const float4 *)(s_ + base);
#pragma unroll
      for (int j4 = 0; j4 < C / 4; ++j4) {
        float4 q_vec = s4[j4];
        const int j = j4 * 4;
        stateT[j + 0] = q_vec.x;
        stateT[j + 1] = q_vec.y;
        stateT[j + 2] = q_vec.z;
        stateT[j + 3] = q_vec.w;
      }

      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
      float inv_tau = 1.0f / tau;
      float dtau_local = 0.0f;
#pragma unroll
      for (int j = 0; j < C; ++j) {
        float u = stateT[j] * inv_tau;
        float tnh = tanhf(u);
        float sech2 = 1.0f - tnh * tnh;
        dtau_local += dstate[j] * (tnh - u * sech2);
        dstate[j] *= sech2;
        dstateT[j] *= sech2;
      }
      dtau_shared[i] = dtau_local;
      __syncthreads();
#pragma unroll
      for (int stride = C / 2; stride > 0; stride /= 2) {
        if (i < stride)
          dtau_shared[i] += dtau_shared[i + stride];
        __syncthreads();
      }
      if (i == 0) {
        dtau_[bb * num_chunks * H + chunk * H + hh] = dtau_shared[0];
      }
    }

    float dq_val = 0.f;
#pragma unroll
    for (int j = 0; j < C; ++j)
      dq_val += stateT[j] * dy[j];
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
    for (int j = 0; j < C; ++j)
      da += stateT[j] * dSb_shared[j];
    da_[ind] = to_bf(da);

#pragma unroll
    for (int j = 0; j < C; ++j) {
      dstate[j] = dstate[j] * w[j] + dSb * a[j];
      dstateT[j] = dstateT[j] * wi + ai * dSb_shared[j];
      if (t == 0)
        dh0_[dht_base + j] = dstate[j];
    }
  }
}

// RWKV-7-SANE 推理前向 CUDA kernel（无 mask）。
//
// Args 与 forward_inference_kernel_sane 相同，但不读 mask。
template <int C>
__launch_bounds__(C, 2) __global__
    void forward_inference_kernel_sane_no_mask(int T, int H, F_ w_, F_ q_,
                                               F_ k_, F_ v_, F_ a_, F_ b_,
                                               const float *__restrict__ tau_,
                                               bf *y_, float *s_, float *h0_) {
  int bb = blockIdx.y, hh = blockIdx.x, i = threadIdx.x;
  float state[C] = {0};
  __shared__ float q[C], k[C], w[C], a[C], b[C];

  int64_t h0_base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    state[j] = h0_[h0_base + j];

  const int num_chunks = T / _CHUNK_LEN_;

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

    if ((t + 1) % _CHUNK_LEN_ == 0) {
      int chunk = t / _CHUNK_LEN_;
      float tau = tau_[bb * num_chunks * H + chunk * H + hh];
#pragma unroll
      for (int j = 0; j < C; ++j) {
        state[j] = tau * tanhf(state[j] / tau);
      }
    }
  }

  int64_t base = ((int64_t)bb * H + hh) * C * C + i * C;
#pragma unroll
  for (int j = 0; j < C; ++j)
    s_[base + j] = state[j];
}

// Host wrapper for forward_kernel_sane_no_mask。
static ffi::Error
WKV7SaneFwdNoMaskHost(cudaStream_t stream, ffi::Buffer<ffi::BF16> w,
                      ffi::Buffer<ffi::BF16> q, ffi::Buffer<ffi::BF16> k,
                      ffi::Buffer<ffi::BF16> v, ffi::Buffer<ffi::BF16> a,
                      ffi::Buffer<ffi::BF16> b, ffi::Buffer<ffi::F32> tau,
                      ffi::Buffer<ffi::F32> h0, ffi::ResultBuffer<ffi::BF16> y,
                      ffi::ResultBuffer<ffi::F32> s,
                      ffi::ResultBuffer<ffi::F32> sa) {
  constexpr int C = _C_;
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  dim3 block(C);
  dim3 grid(H, B);

  forward_kernel_sane_no_mask<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      reinterpret_cast<bf *>(y->typed_data()), s->typed_data(),
      sa->typed_data(), h0.typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA forward_kernel_sane_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// Host wrapper for backward_kernel_sane_no_mask。
static ffi::Error WKV7SaneBwdNoMaskHost(
    cudaStream_t stream, ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::BF16> dy,
    ffi::Buffer<ffi::F32> s, ffi::Buffer<ffi::F32> sa,
    ffi::Buffer<ffi::F32> dht, ffi::ResultBuffer<ffi::F32> dh0,
    ffi::ResultBuffer<ffi::F32> dtau, ffi::ResultBuffer<ffi::BF16> dw,
    ffi::ResultBuffer<ffi::BF16> dq, ffi::ResultBuffer<ffi::BF16> dk,
    ffi::ResultBuffer<ffi::BF16> dv, ffi::ResultBuffer<ffi::BF16> da,
    ffi::ResultBuffer<ffi::BF16> db) {
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  constexpr int C = _C_;
  dim3 block(C);
  dim3 grid(H, B);

  backward_kernel_sane_no_mask<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      reinterpret_cast<bf *>(dy.typed_data()), s.typed_data(), sa.typed_data(),
      dht.typed_data(), dh0->typed_data(), dtau->typed_data(),
      reinterpret_cast<bf *>(dw->typed_data()),
      reinterpret_cast<bf *>(dq->typed_data()),
      reinterpret_cast<bf *>(dk->typed_data()),
      reinterpret_cast<bf *>(dv->typed_data()),
      reinterpret_cast<bf *>(da->typed_data()),
      reinterpret_cast<bf *>(db->typed_data()));

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA backward_kernel_sane_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// Host wrapper for forward_inference_kernel_sane_no_mask。
static ffi::Error WKV7SaneInferenceNoMaskHost(
    cudaStream_t stream, ffi::Buffer<ffi::BF16> w, ffi::Buffer<ffi::BF16> q,
    ffi::Buffer<ffi::BF16> k, ffi::Buffer<ffi::BF16> v,
    ffi::Buffer<ffi::BF16> a, ffi::Buffer<ffi::BF16> b,
    ffi::Buffer<ffi::F32> tau, ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<ffi::BF16> y, ffi::ResultBuffer<ffi::F32> s) {
  constexpr int C = _C_;
  auto dims = w.dimensions();
  int B = dims[0], T = dims[1], H = dims[2];
  dim3 block(C);
  dim3 grid(H, B);

  forward_inference_kernel_sane_no_mask<_C_><<<grid, block, 0, stream>>>(
      T, H, reinterpret_cast<bf *>(w.typed_data()),
      reinterpret_cast<bf *>(q.typed_data()),
      reinterpret_cast<bf *>(k.typed_data()),
      reinterpret_cast<bf *>(v.typed_data()),
      reinterpret_cast<bf *>(a.typed_data()),
      reinterpret_cast<bf *>(b.typed_data()), tau.typed_data(),
      reinterpret_cast<bf *>(y->typed_data()), s->typed_data(),
      h0.typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("CUDA forward_inference_kernel_sane_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// XLA FFI handler 注册。
XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneFwd, WKV7SaneFwdHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneBwd, WKV7SaneBwdHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneInference, WKV7SaneInferenceHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneFwdNoMask, WKV7SaneFwdNoMaskHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneBwdNoMask, WKV7SaneBwdNoMaskHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>(),
                              {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(Wkv7SaneInferenceNoMask,
                              WKV7SaneInferenceNoMaskHost,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<cudaStream_t>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::BF16>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Arg<ffi::Buffer<ffi::F32>>()
                                  .Ret<ffi::Buffer<ffi::BF16>>()
                                  .Ret<ffi::Buffer<ffi::F32>>(),
                              {ffi::Traits::kCmdBufferCompatible});

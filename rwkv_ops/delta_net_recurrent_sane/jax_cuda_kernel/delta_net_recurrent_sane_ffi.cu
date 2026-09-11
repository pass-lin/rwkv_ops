// DeltaNet recurrent SANE JAX FFI CUDA 训练/推理 kernel。

#include <cstdint>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <string>
#include <xla/ffi/api/ffi.h>

namespace ffi = xla::ffi;

using bf = __nv_bfloat16;

__device__ inline float to_float(const bf &u) { return __bfloat162float(u); }

__device__ inline float to_float(const float &u) { return u; }

template <typename ET> __device__ inline ET from_float(const float &u);

template <> __device__ inline bf from_float<bf>(const float &u) {
  return __float2bfloat16_rn(u);
}

template <> __device__ inline float from_float<float>(const float &u) {
  return u;
}

// block 线程数需同时覆盖 K 维向量工作与 V 维 state 列工作。
constexpr int kBlockThreads = (((_K_) > (_V_) ? (_K_) : (_V_)) + 31) / 32 * 32;
constexpr int kNumWarps = kBlockThreads / 32;
// 反向跨线程归约时 j 维的分块大小，限制部分和缓冲的 shared 占用。
constexpr int kJBlock = (_K_) < 64 ? (_K_) : 64;

// DeltaNet recurrent SANE 训练前向 kernel。
//
// 每个 block 处理一个 (batch, head)，按时间步顺序扫描 T 步，在每个 chunk
// 末尾先写出 SANE 之前的 state checkpoint，再在寄存器内对 state 执行
// SANE blend（`state*(1-m) + tau*tanh(state/tau)*m`，blend 形式避免 warp
// 分支），并保存反向所需的 kv_mem 与 q/k 逆范数。
// 线程 v 持有 state 的一列 state[:, v]（_K_ 个 float 寄存器），
// kv_mem 与输出均为线程内 K 维点积，无需跨线程归约；tau/mask 对每个
// block 是标量，全 block 读取同一地址。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, N, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, N, T], float32, row-major。已在外部过 sigmoid。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值，必须 > 0。
//   mask_: [B, T//_CHUNK_LEN_], float32, row-major。>0 的 chunk 边界执行
//     SANE；无 mask 场景由封装层传全 1。
//   h0_: [B, N, K, V], float32, row-major。初始 state。
//   kv_mem_: [B, N, T, V], float32, row-major。每步的 `k_t @ state_{t-1}`。
//   chkp_: [B, N, T//_CHUNK_LEN_, K, V], float32, row-major。chunk 末尾
//     SANE 之前的 state 快照。
//   inv_q_, inv_k_: [B, N, T], float32, row-major。q/k 的 L2 逆范数。
//   ht_: [B, N, K, V], float32, row-major。最终 state（末尾 chunk 已 SANE）。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列，
//   线程 v < _K_ 同时承担 q/k 向量加载与归一化。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_sane_fwd_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ tau_,
    const float *__restrict__ mask_, const float *__restrict__ h0_,
    ET *__restrict__ o_, float *__restrict__ kv_mem_, float *__restrict__ chkp_,
    float *__restrict__ inv_q_, float *__restrict__ inv_k_,
    float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_];
  if (active) {
    const int64_t h0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      state[j] = h0_[h0_base + (int64_t)j * _V_];
  }

  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_red[2][_K_];
  __shared__ float sh_iq, sh_ik;

  for (int t = 0; t < T; t++) {
    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    const float v_val = active ? to_float(v_[vo_base + v]) : 0.f;
    const float beta_t = beta_[bh * T + t];
    __syncthreads();

    if (v < _K_) {
      sh_red[0][v] = sh_q[v] * sh_q[v];
      sh_red[1][v] = sh_k[v] * sh_k[v];
    }
    __syncthreads();
    if (v == 0) {
      float sq = 0.f, sk = 0.f;
      for (int j = 0; j < _K_; j++) {
        sq += sh_red[0][j];
        sk += sh_red[1][j];
      }
      sh_iq = rsqrtf(sq + 1e-6f);
      sh_ik = rsqrtf(sk + 1e-6f);
    }
    __syncthreads();

    const float iq = sh_iq, ik = sh_ik;
    if (v < _K_) {
      sh_khat[v] = sh_k[v] * ik;
      sh_qt[v] = sh_q[v] * iq * scale;
    }
    if (v == 0) {
      inv_q_[bh * T + t] = iq;
      inv_k_[bh * T + t] = ik;
    }
    __syncthreads();

    if (active) {
      float kvm = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        kvm += state[j] * sh_khat[j];
      }
      kv_mem_[vo_base + v] = kvm;

      const float delta = beta_t * (v_val - kvm);
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        state[j] += sh_khat[j] * delta;
        out += state[j] * sh_qt[j];
      }
      o_[vo_base + v] = from_float<ET>(out);

      if ((t + 1) % _CHUNK_LEN_ == 0) {
        const int c = t / _CHUNK_LEN_;
        const int64_t cbase = (bh * num_chunks + c) * _K_ * _V_ + v;
#pragma unroll
        for (int j = 0; j < _K_; j++)
          chkp_[cbase + (int64_t)j * _V_] = state[j];

        const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
        const float m = mask_[(int64_t)bb * num_chunks + c];
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          const float th = tanhf(state[j] / tau_safe);
          state[j] = state[j] * (1.f - m) + tau_safe * th * m;
        }
      }
    }
  }

  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      ht_[sbase + (int64_t)j * _V_] = state[j];
  }
}

// DeltaNet recurrent SANE 训练反向 kernel。
//
// 每个 block 处理一个 (batch, head)。从 dht 开始自 T-1 倒序递推到 0；
// 到达 chunk 边界（含 tp1==T 的初始边界）时从 checkpoint 重载 SANE 之前的
// S_{t+1}，避免长序列反向递推中长序列反向递推的数值稳定处理，同时应用
// SANE 反向：每个 state 元素计算 `u = state/tau_safe; th = tanh(u);
// sech2 = 1-th*th`，下游梯度乘 `(1-m) + m*sech2`，并用 blend 之前的梯度
// 归约出 `dtau = m * sum(carry * (th - u*sech2))`。dtau 需要沿 V 方向跨
// 线程求和：每个活跃线程先对所持列求部分和，warp 内 shuffle 归约后由
// lane 0 写部分和，最后由线程 0 串行汇总各 warp（边界次数少，串行足够）。
// 线程 v 持有 state 列与 dstate 列（各 _K_ 个 float 寄存器）。
// d_k_hat / d_q_tilde 需要沿 V 方向跨线程求和：每个 j 先由全部线程算出
// 本列贡献，warp 内 shuffle 归约后由 lane 0 写部分和，再由前 _K_ 个线程
// 跨 warp 汇总；部分和缓冲按 j 分块复用，占用与 V 无关，避免大 V 时
// shared memory 超过硬件上限。利用 dstate_new 与 state_decay 的线性分解，
// 只需归约原始 state 与 dstate 各一遍，外加四个标量级 block 归约。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, do_: [B, N, T, V], bfloat16 或 float32, row-major。do 为输出梯度。
//   beta_: [B, N, T], float32, row-major。
//   dht_: [B, N, K, V], float32, row-major。最终 state 梯度（无梯度时传零）。
//   kv_mem_: [B, N, T, V], float32, row-major。前向保存的 kv_mem。
//   inv_q_, inv_k_: [B, N, T], float32, row-major。前向保存的逆范数。
//   chkp_: [B, N, T//_CHUNK_LEN_, K, V], float32, row-major。SANE 之前的
//     state 快照，最后一个即 S_T，反向不重算前向。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值。
//   mask_: [B, T//_CHUNK_LEN_], float32, row-major。无 mask 场景为全 1。
//   dq_, dk_: [B, N, T, K], 与 q 同 dtype, row-major。输出梯度。
//   dv_: [B, N, T, V], float32, row-major。输出梯度。
//   dbeta_: [B, N, T], float32, row-major。输出梯度。
//   dh0_: [B, N, K, V], float32, row-major。初始 state 梯度。
//   dtau_: [B, N, T//_CHUNK_LEN_], float32, row-major。tau 梯度。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state/dstate 第 v 列，
//   线程 v < _K_ 承担跨 warp 部分和的汇总。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_sane_bwd_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const ET *__restrict__ do_,
    const float *__restrict__ dht_, const float *__restrict__ kv_mem_,
    const float *__restrict__ inv_q_, const float *__restrict__ inv_k_,
    const float *__restrict__ chkp_, const float *__restrict__ tau_,
    const float *__restrict__ mask_, ET *__restrict__ dq_, ET *__restrict__ dk_,
    float *__restrict__ dv_, float *__restrict__ dbeta_,
    float *__restrict__ dh0_, float *__restrict__ dtau_) {
  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_dkhat[_K_], sh_dqhat[_K_];
  __shared__ float sh_do[_V_], sh_delta[_V_], sh_dkv[_V_];
  __shared__ float sh_red[4][_V_];
  __shared__ float sh_scalar[6];
  // 跨线程归约的 warp 部分和缓冲，按 j 分块复用，占用与 V 无关。
  __shared__ float warp_part[3][kJBlock * kNumWarps];
  // dtau 跨 warp 部分和缓冲。
  __shared__ float dtau_part[kNumWarps];

  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int warp = v >> 5, lane = v & 31;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_], carry[_K_];

  if (active) {
    const int64_t dht_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      carry[j] = dht_[dht_base + (int64_t)j * _V_];
  }

  for (int t = T - 1; t >= 0; t--) {
    const int tp1 = t + 1;
    // chkp 最后一项即 S_T，chunk 边界一律从 checkpoint 重载 S_{t+1}。
    if (tp1 % _CHUNK_LEN_ == 0) {
      const int c = tp1 / _CHUNK_LEN_ - 1;
      const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
      const float m = mask_[(int64_t)bb * num_chunks + c];
      float dtau_p = 0.f;
      if (active) {
        const int64_t cbase = (bh * num_chunks + c) * _K_ * _V_ + v;
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          state[j] = chkp_[cbase + (int64_t)j * _V_];
          const float u = state[j] / tau_safe;
          const float th = tanhf(u);
          const float sech2 = 1.f - th * th;
          dtau_p += carry[j] * (th - u * sech2);
          carry[j] *= (1.f - m) + m * sech2;
        }
      }
      // 非活跃线程以 0 参与 shuffle，保证全 warp 归约正确。
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        dtau_p += __shfl_xor_sync(0xffffffffu, dtau_p, off);
      if (lane == 0)
        dtau_part[warp] = dtau_p;
      __syncthreads();
      if (v == 0) {
        float s = 0.f;
        for (int w2 = 0; w2 < kNumWarps; w2++)
          s += dtau_part[w2];
        dtau_[bh * num_chunks + c] = m * s;
      }
    }

    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;
    const float beta_t = beta_[bh * T + t];
    const float iq = inv_q_[bh * T + t];
    const float ik = inv_k_[bh * T + t];

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    float do_v = 0.f, v_val = 0.f, kvm = 0.f;
    if (active) {
      do_v = to_float(do_[vo_base + v]);
      v_val = to_float(v_[vo_base + v]);
      kvm = kv_mem_[vo_base + v];
    }
    __syncthreads();
    if (v < _K_) {
      sh_khat[v] = sh_k[v] * ik;
      sh_qt[v] = sh_q[v] * iq * scale;
    }
    __syncthreads();

    float delta_v = 0.f, dkv_v = 0.f;
    if (active) {
      delta_v = beta_t * (v_val - kvm);
      float d_delta = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++)
        d_delta += (carry[j] + sh_qt[j] * do_v) * sh_khat[j];
      dv_[vo_base + v] = beta_t * d_delta;
      dkv_v = -beta_t * d_delta;

#pragma unroll
      for (int j = 0; j < _K_; j++) {
        const float dnew = carry[j] + sh_qt[j] * do_v;
        const float sdec = state[j] - sh_khat[j] * delta_v;
      }

      sh_do[v] = do_v;
      sh_delta[v] = delta_v;
      sh_dkv[v] = dkv_v;
      sh_red[0][v] = (v_val - kvm) * d_delta;
      sh_red[2][v] = do_v * delta_v;
      sh_red[3][v] = delta_v * dkv_v;
    }
    __syncthreads();

    if (v < 4) {
      float s = 0.f;
      for (int j = 0; j < _V_; j++)
        s += sh_red[v][j];
      sh_scalar[v] = s;
    }

    // 沿 V 方向对 state/dstate 列加权求和得 t1/t2/t3，再由线性分解合成
    // d_k_hat：sum(dstate_new * delta) = t1 + q_tilde * sum(do * delta)，
    // sum(state_decay * d_kv_mem) = t2 - k_hat * sum(delta * d_kv_mem)。
    // 非活跃线程的贡献必须为 0，且不能读取未初始化的寄存器参与运算。
    float t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (int jb = 0; jb < _K_; jb += kJBlock) {
      const int jn = min(kJBlock, _K_ - jb);
      __syncthreads();
      for (int jj = 0; jj < jn; jj++) {
        const int j = jb + jj;
        float c1 = active ? carry[j] * delta_v : 0.f;
        float c2 = active ? state[j] * dkv_v : 0.f;
        float c3 = active ? state[j] * do_v : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          c1 += __shfl_xor_sync(0xffffffffu, c1, off);
          c2 += __shfl_xor_sync(0xffffffffu, c2, off);
          c3 += __shfl_xor_sync(0xffffffffu, c3, off);
        }
        if (lane == 0) {
          warp_part[0][jj * kNumWarps + warp] = c1;
          warp_part[1][jj * kNumWarps + warp] = c2;
          warp_part[2][jj * kNumWarps + warp] = c3;
        }
      }
      __syncthreads();
      if (v >= jb && v < jb + jn) {
        const int jj = v - jb;
        for (int w2 = 0; w2 < kNumWarps; w2++) {
          t1 += warp_part[0][jj * kNumWarps + w2];
          t2 += warp_part[1][jj * kNumWarps + w2];
          t3 += warp_part[2][jj * kNumWarps + w2];
        }
      }
    }
    __syncthreads();

    if (v < _K_) {
      sh_dkhat[v] =
          t1 + sh_qt[v] * sh_scalar[2] + t2 - sh_khat[v] * sh_scalar[3];
      sh_dqhat[v] = scale * t3;
    }
    __syncthreads();

    if (v == 0) {
      float qd = 0.f, kd = 0.f;
      for (int j = 0; j < _K_; j++) {
        qd += sh_q[j] * iq * sh_dqhat[j];
        kd += sh_khat[j] * sh_dkhat[j];
      }
      sh_scalar[4] = qd;
      sh_scalar[5] = kd;
    }
    __syncthreads();

    if (v < _K_) {
      dq_[qk_base + v] =
          from_float<ET>(iq * (sh_dqhat[v] - sh_q[v] * iq * sh_scalar[4]));
      dk_[qk_base + v] =
          from_float<ET>(ik * (sh_dkhat[v] - sh_khat[v] * sh_scalar[5]));
    }
    if (v == 0) {
      dbeta_[bh * T + t] = sh_scalar[0];
    }

    if (active) {
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        const float dnew = carry[j] + sh_qt[j] * do_v;
        const float sdec = state[j] - sh_khat[j] * delta_v;
        carry[j] = dnew + sh_khat[j] * dkv_v;
      }
    }
  }

  if (active) {
    const int64_t dh0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      dh0_[dh0_base + (int64_t)j * _V_] = carry[j];
  }
}

// DeltaNet recurrent SANE 推理前向 kernel。
//
// 与训练前向数学一致，但不输出 kv_mem / checkpoint / 逆范数，减少推理
// 显存占用。每个 block 处理一个 (batch, head)。支持任意 T：只在满 chunk
// 的边界执行 SANE，末尾不足 chunk 的余数步不执行。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, N, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, N, T], float32, row-major。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值。
//   mask_: [B, T//_CHUNK_LEN_], float32, row-major。无 mask 场景为全 1。
//   h0_: [B, N, K, V], float32, row-major。初始 state。
//   ht_: [B, N, K, V], float32, row-major。最终 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_sane_inference_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ tau_,
    const float *__restrict__ mask_, const float *__restrict__ h0_,
    ET *__restrict__ o_, float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_];
  if (active) {
    const int64_t h0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      state[j] = h0_[h0_base + (int64_t)j * _V_];
  }

  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_red[2][_K_];
  __shared__ float sh_iq, sh_ik;

  for (int t = 0; t < T; t++) {
    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    const float v_val = active ? to_float(v_[vo_base + v]) : 0.f;
    const float beta_t = beta_[bh * T + t];
    __syncthreads();

    if (v < _K_) {
      sh_red[0][v] = sh_q[v] * sh_q[v];
      sh_red[1][v] = sh_k[v] * sh_k[v];
    }
    __syncthreads();
    if (v == 0) {
      float sq = 0.f, sk = 0.f;
      for (int j = 0; j < _K_; j++) {
        sq += sh_red[0][j];
        sk += sh_red[1][j];
      }
      sh_iq = rsqrtf(sq + 1e-6f);
      sh_ik = rsqrtf(sk + 1e-6f);
    }
    __syncthreads();

    if (v < _K_) {
      sh_khat[v] = sh_k[v] * sh_ik;
      sh_qt[v] = sh_q[v] * sh_iq * scale;
    }
    __syncthreads();

    if (active) {
      float kvm = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        kvm += state[j] * sh_khat[j];
      }
      const float delta = beta_t * (v_val - kvm);
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        state[j] += sh_khat[j] * delta;
        out += state[j] * sh_qt[j];
      }
      o_[vo_base + v] = from_float<ET>(out);

      if ((t + 1) % _CHUNK_LEN_ == 0) {
        const int c = t / _CHUNK_LEN_;
        const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
        const float m = mask_[(int64_t)bb * num_chunks + c];
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          const float th = tanhf(state[j] / tau_safe);
          state[j] = state[j] * (1.f - m) + tau_safe * th * m;
        }
      }
    }
  }

  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      ht_[sbase + (int64_t)j * _V_] = state[j];
  }
}

// DeltaNet recurrent SANE 单步 RNN 前向 kernel。
//
// 每个 block 处理一个 (batch, head) 的一个时间步，用于 decode。
// 输入没有时间维。输出基于 SANE 之前的 state，写出的下一步 state 按
// do_sane blend 后落定。
//
// Args:
//   q_, k_: [B, N, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, N, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, N], float32, row-major。
//   tau_: [B, N], float32, row-major。SANE 阈值。
//   do_sane_: [B], float32, row-major。>0 在该步执行 SANE。
//   h0_: [B, N, K, V], float32, row-major。上一步 state。
//   ht_: [B, N, K, V], float32, row-major。下一步 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，单步 kernel 不使用，仅为编译标识一致。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_sane_single_step_kernel(
    int H, float scale, const ET *__restrict__ q_, const ET *__restrict__ k_,
    const ET *__restrict__ v_, const float *__restrict__ beta_,
    const float *__restrict__ tau_, const float *__restrict__ do_sane_,
    const float *__restrict__ h0_, ET *__restrict__ o_,
    float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;

  __shared__ float sh_q[_K_], sh_k[_K_];
  __shared__ float sh_red[2][_K_];
  __shared__ float sh_iq, sh_ik;

  if (v < _K_) {
    sh_q[v] = to_float(q_[bh * _K_ + v]);
    sh_k[v] = to_float(k_[bh * _K_ + v]);
  }
  const float v_val = active ? to_float(v_[bh * _V_ + v]) : 0.f;
  const float beta_t = beta_[bh];
  __syncthreads();

  if (v < _K_) {
    sh_red[0][v] = sh_q[v] * sh_q[v];
    sh_red[1][v] = sh_k[v] * sh_k[v];
  }
  __syncthreads();
  if (v == 0) {
    float sq = 0.f, sk = 0.f;
    for (int j = 0; j < _K_; j++) {
      sq += sh_red[0][j];
      sk += sh_red[1][j];
    }
    sh_iq = rsqrtf(sq + 1e-6f);
    sh_ik = rsqrtf(sk + 1e-6f);
  }
  __syncthreads();

  const float iq = sh_iq, ik = sh_ik;
  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
    float state[_K_];
    float kvm = 0.f;
#pragma unroll
    for (int j = 0; j < _K_; j++) {
      state[j] = h0_[sbase + (int64_t)j * _V_];
      kvm += state[j] * sh_k[j] * ik;
    }
    const float delta = beta_t * (v_val - kvm);
    float out = 0.f;
#pragma unroll
    for (int j = 0; j < _K_; j++) {
      state[j] += sh_k[j] * ik * delta;
      out += state[j] * sh_q[j] * iq * scale;
    }
    o_[bh * _V_ + v] = from_float<ET>(out);

    const float tau_safe = fmaxf(tau_[bh], 1e-6f);
    const float m = do_sane_[bb];
#pragma unroll
    for (int j = 0; j < _K_; j++) {
      const float th = tanhf(state[j] / tau_safe);
      state[j] = state[j] * (1.f - m) + tau_safe * th * m;
      ht_[sbase + (int64_t)j * _V_] = state[j];
    }
  }
}

// FFI host 启动函数

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneFwdHost(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> mask, ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<DT> o, ffi::ResultBuffer<ffi::F32> kv_mem,
    ffi::ResultBuffer<ffi::F32> chkp, ffi::ResultBuffer<ffi::F32> inv_q,
    ffi::ResultBuffer<ffi::F32> inv_k, ffi::ResultBuffer<ffi::F32> ht) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_fwd_kernel<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          tau.typed_data(), mask.typed_data(), h0.typed_data(),
          reinterpret_cast<ET *>(o->typed_data()), kv_mem->typed_data(),
          chkp->typed_data(), inv_q->typed_data(), inv_k->typed_data(),
          ht->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_fwd error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneBwdHost(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<DT> dy,
    ffi::Buffer<ffi::F32> dht, ffi::Buffer<ffi::F32> kv_mem,
    ffi::Buffer<ffi::F32> inv_q, ffi::Buffer<ffi::F32> inv_k,
    ffi::Buffer<ffi::F32> chkp, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> mask, ffi::ResultBuffer<DT> dq,
    ffi::ResultBuffer<DT> dk, ffi::ResultBuffer<ffi::F32> dv,
    ffi::ResultBuffer<ffi::F32> dbeta, ffi::ResultBuffer<ffi::F32> dh0,
    ffi::ResultBuffer<ffi::F32> dtau) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_bwd_kernel<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          reinterpret_cast<const ET *>(dy.typed_data()), dht.typed_data(),
          kv_mem.typed_data(), inv_q.typed_data(), inv_k.typed_data(),
          chkp.typed_data(), tau.typed_data(), mask.typed_data(),
          reinterpret_cast<ET *>(dq->typed_data()),
          reinterpret_cast<ET *>(dk->typed_data()), dv->typed_data(),
          dbeta->typed_data(), dh0->typed_data(), dtau->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_bwd error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneInferenceHost(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> mask, ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<DT> o, ffi::ResultBuffer<ffi::F32> ht) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_inference_kernel<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          tau.typed_data(), mask.typed_data(), h0.typed_data(),
          reinterpret_cast<ET *>(o->typed_data()), ht->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_inference error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneSingleStepHost(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> do_sane, ffi::Buffer<ffi::F32> h0,
    ffi::ResultBuffer<DT> o, ffi::ResultBuffer<ffi::F32> ht) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_single_step_kernel<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          tau.typed_data(), do_sane.typed_data(), h0.typed_data(),
          reinterpret_cast<ET *>(o->typed_data()), ht->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_single_step error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// FFI 注册（bf16 / f32 各一套）

#define DN_SANE_DEFINE_FWD_HANDLER(SYMBOL, ET, DT)                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(SYMBOL, (DeltaNetSaneFwdHost<ET, DT>),         \
                                ffi::Ffi::Bind()                               \
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()  \
                                    .Arg<ffi::Buffer<DT>>()       /* q */      \
                                    .Arg<ffi::Buffer<DT>>()       /* k */      \
                                    .Arg<ffi::Buffer<DT>>()       /* v */      \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* beta */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* tau */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* mask */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* h0 */     \
                                    .Ret<ffi::Buffer<DT>>()       /* o */      \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* kv_mem */ \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* chkp */   \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* inv_q */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* inv_k */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* ht */,    \
                                {ffi::Traits::kCmdBufferCompatible})

#define DN_SANE_DEFINE_BWD_HANDLER(SYMBOL, ET, DT)                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(SYMBOL, (DeltaNetSaneBwdHost<ET, DT>),         \
                                ffi::Ffi::Bind()                               \
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()  \
                                    .Arg<ffi::Buffer<DT>>()       /* q */      \
                                    .Arg<ffi::Buffer<DT>>()       /* k */      \
                                    .Arg<ffi::Buffer<DT>>()       /* v */      \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* beta */   \
                                    .Arg<ffi::Buffer<DT>>()       /* dy */     \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* dht */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* kv_mem */ \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* inv_q */  \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* inv_k */  \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* chkp */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* tau */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* mask */   \
                                    .Ret<ffi::Buffer<DT>>()       /* dq */     \
                                    .Ret<ffi::Buffer<DT>>()       /* dk */     \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dv */     \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dbeta */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dh0 */    \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dtau */,  \
                                {ffi::Traits::kCmdBufferCompatible})

#define DN_SANE_DEFINE_INF_HANDLER(SYMBOL, ET, DT)                             \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(SYMBOL, (DeltaNetSaneInferenceHost<ET, DT>),   \
                                ffi::Ffi::Bind()                               \
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()  \
                                    .Arg<ffi::Buffer<DT>>()       /* q */      \
                                    .Arg<ffi::Buffer<DT>>()       /* k */      \
                                    .Arg<ffi::Buffer<DT>>()       /* v */      \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* beta */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* tau */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* mask */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* h0 */     \
                                    .Ret<ffi::Buffer<DT>>()       /* o */      \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* ht */,    \
                                {ffi::Traits::kCmdBufferCompatible})

#define DN_SANE_DEFINE_SINGLE_HANDLER(SYMBOL, ET, DT)                          \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                               \
      SYMBOL, (DeltaNetSaneSingleStepHost<ET, DT>),                            \
      ffi::Ffi::Bind()                                                         \
          .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
          .Arg<ffi::Buffer<DT>>()       /* q */                                \
          .Arg<ffi::Buffer<DT>>()       /* k */                                \
          .Arg<ffi::Buffer<DT>>()       /* v */                                \
          .Arg<ffi::Buffer<ffi::F32>>() /* beta */                             \
          .Arg<ffi::Buffer<ffi::F32>>() /* tau */                              \
          .Arg<ffi::Buffer<ffi::F32>>() /* do_sane */                          \
          .Arg<ffi::Buffer<ffi::F32>>() /* h0 */                               \
          .Ret<ffi::Buffer<DT>>()       /* o */                                \
          .Ret<ffi::Buffer<ffi::F32>>() /* ht */,                              \
      {ffi::Traits::kCmdBufferCompatible})

DN_SANE_DEFINE_FWD_HANDLER(DeltaNetRecurrentSaneFwdBf16, bf, ffi::BF16);
DN_SANE_DEFINE_BWD_HANDLER(DeltaNetRecurrentSaneBwdBf16, bf, ffi::BF16);
DN_SANE_DEFINE_INF_HANDLER(DeltaNetRecurrentSaneInferenceBf16, bf, ffi::BF16);
DN_SANE_DEFINE_SINGLE_HANDLER(DeltaNetRecurrentSaneSingleStepBf16, bf,
                              ffi::BF16);

DN_SANE_DEFINE_FWD_HANDLER(DeltaNetRecurrentSaneFwdF32, float, ffi::F32);
DN_SANE_DEFINE_BWD_HANDLER(DeltaNetRecurrentSaneBwdF32, float, ffi::F32);
DN_SANE_DEFINE_INF_HANDLER(DeltaNetRecurrentSaneInferenceF32, float, ffi::F32);
DN_SANE_DEFINE_SINGLE_HANDLER(DeltaNetRecurrentSaneSingleStepF32, float,
                              ffi::F32);

// DeltaNet recurrent SANE 训练前向 kernel（无 mask）。
//
// 与带 mask 版本数学一致，但 chunk 边界无条件执行 SANE，全程不读取 mask。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, N, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, N, T], float32, row-major。已在外部过 sigmoid。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值，必须 > 0。
//   h0_: [B, N, K, V], float32, row-major。初始 state。
//   kv_mem_: [B, N, T, V], float32, row-major。每步的 `k_t @ state_{t-1}`。
//   chkp_: [B, N, T//_CHUNK_LEN_, K, V], float32, row-major。SANE 之前的快照。
//   inv_q_, inv_k_: [B, N, T], float32, row-major。q/k 的 L2 逆范数。
//   ht_: [B, N, K, V], float32, row-major。最终 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__ __launch_bounds__(kBlockThreads) void
delta_net_recurrent_sane_fwd_kernel_no_mask(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ tau_,
    const float *__restrict__ h0_, ET *__restrict__ o_,
    float *__restrict__ kv_mem_, float *__restrict__ chkp_,
    float *__restrict__ inv_q_, float *__restrict__ inv_k_,
    float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_];
  if (active) {
    const int64_t h0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      state[j] = h0_[h0_base + (int64_t)j * _V_];
  }

  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_red[2][_K_];
  __shared__ float sh_iq, sh_ik;

  for (int t = 0; t < T; t++) {
    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    const float v_val = active ? to_float(v_[vo_base + v]) : 0.f;
    const float beta_t = beta_[bh * T + t];
    __syncthreads();

    if (v < _K_) {
      sh_red[0][v] = sh_q[v] * sh_q[v];
      sh_red[1][v] = sh_k[v] * sh_k[v];
    }
    __syncthreads();
    if (v == 0) {
      float sq = 0.f, sk = 0.f;
      for (int j = 0; j < _K_; j++) {
        sq += sh_red[0][j];
        sk += sh_red[1][j];
      }
      sh_iq = rsqrtf(sq + 1e-6f);
      sh_ik = rsqrtf(sk + 1e-6f);
    }
    __syncthreads();

    const float iq = sh_iq, ik = sh_ik;
    if (v < _K_) {
      sh_khat[v] = sh_k[v] * ik;
      sh_qt[v] = sh_q[v] * iq * scale;
    }
    if (v == 0) {
      inv_q_[bh * T + t] = iq;
      inv_k_[bh * T + t] = ik;
    }
    __syncthreads();

    if (active) {
      float kvm = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        kvm += state[j] * sh_khat[j];
      }
      kv_mem_[vo_base + v] = kvm;

      const float delta = beta_t * (v_val - kvm);
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        state[j] += sh_khat[j] * delta;
        out += state[j] * sh_qt[j];
      }
      o_[vo_base + v] = from_float<ET>(out);

      if ((t + 1) % _CHUNK_LEN_ == 0) {
        const int c = t / _CHUNK_LEN_;
        const int64_t cbase = (bh * num_chunks + c) * _K_ * _V_ + v;
#pragma unroll
        for (int j = 0; j < _K_; j++)
          chkp_[cbase + (int64_t)j * _V_] = state[j];

        const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          const float th = tanhf(state[j] / tau_safe);
          state[j] = tau_safe * th;
        }
      }
    }
  }

  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      ht_[sbase + (int64_t)j * _V_] = state[j];
  }
}

// DeltaNet recurrent SANE 训练反向 kernel（无 mask）。
//
// 与带 mask 版本数学一致，但 chunk 边界无条件累加 dtau 并把下游梯度乘
// sech2，全程不读取 mask。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, do_: [B, N, T, V], bfloat16 或 float32, row-major。do 为输出梯度。
//   beta_: [B, N, T], float32, row-major。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值。
//   dht_: [B, N, K, V], float32, row-major。最终 state 梯度。
//   kv_mem_: [B, N, T, V], float32, row-major。前向保存的 kv_mem。
//   inv_q_, inv_k_: [B, N, T], float32, row-major。
//   chkp_: [B, N, T//_CHUNK_LEN_, K, V], float32, row-major。SANE 之前的快照。
//   dq_, dk_: [B, N, T, K]，输出梯度，与输入同 dtype。
//   dv_, dbeta_, dh0_, dtau_: 输出梯度，float32。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state/dstate 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__ __launch_bounds__(kBlockThreads) void
delta_net_recurrent_sane_bwd_kernel_no_mask(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const ET *__restrict__ do_,
    const float *__restrict__ dht_, const float *__restrict__ kv_mem_,
    const float *__restrict__ inv_q_, const float *__restrict__ inv_k_,
    const float *__restrict__ chkp_, const float *__restrict__ tau_,
    ET *__restrict__ dq_, ET *__restrict__ dk_, float *__restrict__ dv_,
    float *__restrict__ dbeta_, float *__restrict__ dh0_,
    float *__restrict__ dtau_) {
  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_dkhat[_K_], sh_dqhat[_K_];
  __shared__ float sh_do[_V_], sh_delta[_V_], sh_dkv[_V_];
  __shared__ float sh_red[4][_V_];
  __shared__ float sh_scalar[6];
  // 跨线程归约的 warp 部分和缓冲，按 j 分块复用，占用与 V 无关。
  __shared__ float warp_part[3][kJBlock * kNumWarps];
  // dtau 跨 warp 部分和缓冲。
  __shared__ float dtau_part[kNumWarps];

  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int warp = v >> 5, lane = v & 31;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_], carry[_K_];

  if (active) {
    const int64_t dht_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      carry[j] = dht_[dht_base + (int64_t)j * _V_];
  }

  for (int t = T - 1; t >= 0; t--) {
    const int tp1 = t + 1;
    // chkp 最后一项即 S_T，chunk 边界一律从 checkpoint 重载 S_{t+1}。
    if (tp1 % _CHUNK_LEN_ == 0) {
      const int c = tp1 / _CHUNK_LEN_ - 1;
      const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
      float dtau_p = 0.f;
      if (active) {
        const int64_t cbase = (bh * num_chunks + c) * _K_ * _V_ + v;
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          state[j] = chkp_[cbase + (int64_t)j * _V_];
          const float u = state[j] / tau_safe;
          const float th = tanhf(u);
          const float sech2 = 1.f - th * th;
          dtau_p += carry[j] * (th - u * sech2);
          carry[j] *= sech2;
        }
      }
      // 非活跃线程以 0 参与 shuffle，保证全 warp 归约正确。
#pragma unroll
      for (int off = 16; off > 0; off >>= 1)
        dtau_p += __shfl_xor_sync(0xffffffffu, dtau_p, off);
      if (lane == 0)
        dtau_part[warp] = dtau_p;
      __syncthreads();
      if (v == 0) {
        float s = 0.f;
        for (int w2 = 0; w2 < kNumWarps; w2++)
          s += dtau_part[w2];
        dtau_[bh * num_chunks + c] = s;
      }
    }

    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;
    const float beta_t = beta_[bh * T + t];
    const float iq = inv_q_[bh * T + t];
    const float ik = inv_k_[bh * T + t];

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    float do_v = 0.f, v_val = 0.f, kvm = 0.f;
    if (active) {
      do_v = to_float(do_[vo_base + v]);
      v_val = to_float(v_[vo_base + v]);
      kvm = kv_mem_[vo_base + v];
    }
    __syncthreads();
    if (v < _K_) {
      sh_khat[v] = sh_k[v] * ik;
      sh_qt[v] = sh_q[v] * iq * scale;
    }
    __syncthreads();

    float delta_v = 0.f, dkv_v = 0.f;
    if (active) {
      delta_v = beta_t * (v_val - kvm);
      float d_delta = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++)
        d_delta += (carry[j] + sh_qt[j] * do_v) * sh_khat[j];
      dv_[vo_base + v] = beta_t * d_delta;
      dkv_v = -beta_t * d_delta;

#pragma unroll
      for (int j = 0; j < _K_; j++) {
        const float dnew = carry[j] + sh_qt[j] * do_v;
        const float sdec = state[j] - sh_khat[j] * delta_v;
      }

      sh_do[v] = do_v;
      sh_delta[v] = delta_v;
      sh_dkv[v] = dkv_v;
      sh_red[0][v] = (v_val - kvm) * d_delta;
      sh_red[2][v] = do_v * delta_v;
      sh_red[3][v] = delta_v * dkv_v;
    }
    __syncthreads();

    if (v < 4) {
      float s = 0.f;
      for (int j = 0; j < _V_; j++)
        s += sh_red[v][j];
      sh_scalar[v] = s;
    }

    // 沿 V 方向对 state/dstate 列加权求和得 t1/t2/t3，再由线性分解合成
    // d_k_hat：sum(dstate_new * delta) = t1 + q_tilde * sum(do * delta)，
    // sum(state_decay * d_kv_mem) = t2 - k_hat * sum(delta * d_kv_mem)。
    // 非活跃线程的贡献必须为 0，且不能读取未初始化的寄存器参与运算。
    float t1 = 0.f, t2 = 0.f, t3 = 0.f;
    for (int jb = 0; jb < _K_; jb += kJBlock) {
      const int jn = min(kJBlock, _K_ - jb);
      __syncthreads();
      for (int jj = 0; jj < jn; jj++) {
        const int j = jb + jj;
        float c1 = active ? carry[j] * delta_v : 0.f;
        float c2 = active ? state[j] * dkv_v : 0.f;
        float c3 = active ? state[j] * do_v : 0.f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
          c1 += __shfl_xor_sync(0xffffffffu, c1, off);
          c2 += __shfl_xor_sync(0xffffffffu, c2, off);
          c3 += __shfl_xor_sync(0xffffffffu, c3, off);
        }
        if (lane == 0) {
          warp_part[0][jj * kNumWarps + warp] = c1;
          warp_part[1][jj * kNumWarps + warp] = c2;
          warp_part[2][jj * kNumWarps + warp] = c3;
        }
      }
      __syncthreads();
      if (v >= jb && v < jb + jn) {
        const int jj = v - jb;
        for (int w2 = 0; w2 < kNumWarps; w2++) {
          t1 += warp_part[0][jj * kNumWarps + w2];
          t2 += warp_part[1][jj * kNumWarps + w2];
          t3 += warp_part[2][jj * kNumWarps + w2];
        }
      }
    }
    __syncthreads();

    if (v < _K_) {
      sh_dkhat[v] =
          t1 + sh_qt[v] * sh_scalar[2] + t2 - sh_khat[v] * sh_scalar[3];
      sh_dqhat[v] = scale * t3;
    }
    __syncthreads();

    if (v == 0) {
      float qd = 0.f, kd = 0.f;
      for (int j = 0; j < _K_; j++) {
        qd += sh_q[j] * iq * sh_dqhat[j];
        kd += sh_khat[j] * sh_dkhat[j];
      }
      sh_scalar[4] = qd;
      sh_scalar[5] = kd;
    }
    __syncthreads();

    if (v < _K_) {
      dq_[qk_base + v] =
          from_float<ET>(iq * (sh_dqhat[v] - sh_q[v] * iq * sh_scalar[4]));
      dk_[qk_base + v] =
          from_float<ET>(ik * (sh_dkhat[v] - sh_khat[v] * sh_scalar[5]));
    }
    if (v == 0) {
      dbeta_[bh * T + t] = sh_scalar[0];
    }

    if (active) {
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        const float dnew = carry[j] + sh_qt[j] * do_v;
        const float sdec = state[j] - sh_khat[j] * delta_v;
        carry[j] = dnew + sh_khat[j] * dkv_v;
      }
    }
  }

  if (active) {
    const int64_t dh0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      dh0_[dh0_base + (int64_t)j * _V_] = carry[j];
  }
}

// DeltaNet recurrent SANE 推理前向 kernel（无 mask）。
//
// 与带 mask 版本数学一致，但 chunk 边界无条件执行 SANE，不读取 mask。
// 支持任意 T：只在满 chunk 的边界执行 SANE，末尾不足 chunk 的余数步不执行。
//
// Args:
//   q_, k_: [B, N, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, N, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, N, T], float32, row-major。
//   tau_: [B, N, T//_CHUNK_LEN_], float32, row-major。SANE 阈值。
//   h0_: [B, N, K, V], float32, row-major。初始 state。
//   ht_: [B, N, K, V], float32, row-major。最终 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (N, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16。
template <typename ET>
__global__ __launch_bounds__(kBlockThreads) void
delta_net_recurrent_sane_inference_kernel_no_mask(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ tau_,
    const float *__restrict__ h0_, ET *__restrict__ o_,
    float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_];
  if (active) {
    const int64_t h0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      state[j] = h0_[h0_base + (int64_t)j * _V_];
  }

  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_red[2][_K_];
  __shared__ float sh_iq, sh_ik;

  for (int t = 0; t < T; t++) {
    const int64_t qk_base = (bh * T + t) * _K_;
    const int64_t vo_base = (bh * T + t) * _V_;

    __syncthreads();
    if (v < _K_) {
      sh_q[v] = to_float(q_[qk_base + v]);
      sh_k[v] = to_float(k_[qk_base + v]);
    }
    const float v_val = active ? to_float(v_[vo_base + v]) : 0.f;
    const float beta_t = beta_[bh * T + t];
    __syncthreads();

    if (v < _K_) {
      sh_red[0][v] = sh_q[v] * sh_q[v];
      sh_red[1][v] = sh_k[v] * sh_k[v];
    }
    __syncthreads();
    if (v == 0) {
      float sq = 0.f, sk = 0.f;
      for (int j = 0; j < _K_; j++) {
        sq += sh_red[0][j];
        sk += sh_red[1][j];
      }
      sh_iq = rsqrtf(sq + 1e-6f);
      sh_ik = rsqrtf(sk + 1e-6f);
    }
    __syncthreads();

    if (v < _K_) {
      sh_khat[v] = sh_k[v] * sh_ik;
      sh_qt[v] = sh_q[v] * sh_iq * scale;
    }
    __syncthreads();

    if (active) {
      float kvm = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        kvm += state[j] * sh_khat[j];
      }
      const float delta = beta_t * (v_val - kvm);
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        state[j] += sh_khat[j] * delta;
        out += state[j] * sh_qt[j];
      }
      o_[vo_base + v] = from_float<ET>(out);

      if ((t + 1) % _CHUNK_LEN_ == 0) {
        const int c = t / _CHUNK_LEN_;
        const float tau_safe = fmaxf(tau_[bh * num_chunks + c], 1e-6f);
#pragma unroll
        for (int j = 0; j < _K_; j++) {
          const float th = tanhf(state[j] / tau_safe);
          state[j] = tau_safe * th;
        }
      }
    }
  }

  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      ht_[sbase + (int64_t)j * _V_] = state[j];
  }
}



// 无 mask 版本的 FFI host 启动函数

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneFwdHostNoMask(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> h0, ffi::ResultBuffer<DT> o,
    ffi::ResultBuffer<ffi::F32> kv_mem, ffi::ResultBuffer<ffi::F32> chkp,
    ffi::ResultBuffer<ffi::F32> inv_q, ffi::ResultBuffer<ffi::F32> inv_k,
    ffi::ResultBuffer<ffi::F32> ht) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_fwd_kernel_no_mask<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          tau.typed_data(), h0.typed_data(),
          reinterpret_cast<ET *>(o->typed_data()), kv_mem->typed_data(),
          chkp->typed_data(), inv_q->typed_data(), inv_k->typed_data(),
          ht->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_fwd_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneBwdHostNoMask(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<DT> dy,
    ffi::Buffer<ffi::F32> dht, ffi::Buffer<ffi::F32> kv_mem,
    ffi::Buffer<ffi::F32> inv_q, ffi::Buffer<ffi::F32> inv_k,
    ffi::Buffer<ffi::F32> chkp, ffi::Buffer<ffi::F32> tau,
    ffi::ResultBuffer<DT> dq, ffi::ResultBuffer<DT> dk,
    ffi::ResultBuffer<ffi::F32> dv, ffi::ResultBuffer<ffi::F32> dbeta,
    ffi::ResultBuffer<ffi::F32> dh0, ffi::ResultBuffer<ffi::F32> dtau) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_bwd_kernel_no_mask<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          reinterpret_cast<const ET *>(dy.typed_data()), dht.typed_data(),
          kv_mem.typed_data(), inv_q.typed_data(), inv_k.typed_data(),
          chkp.typed_data(), tau.typed_data(),
          reinterpret_cast<ET *>(dq->typed_data()),
          reinterpret_cast<ET *>(dk->typed_data()), dv->typed_data(),
          dbeta->typed_data(), dh0->typed_data(), dtau->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_bwd_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

template <typename ET, ffi::DataType DT>
static ffi::Error DeltaNetSaneInferenceHostNoMask(
    cudaStream_t stream, ffi::Buffer<DT> q, ffi::Buffer<DT> k,
    ffi::Buffer<DT> v, ffi::Buffer<ffi::F32> beta, ffi::Buffer<ffi::F32> tau,
    ffi::Buffer<ffi::F32> h0, ffi::ResultBuffer<DT> o,
    ffi::ResultBuffer<ffi::F32> ht) {
  auto dims = q.dimensions();
  int B = dims[0], H = dims[1], T = dims[2];
  const float scale = 1.0f / sqrtf((float)_K_);

  delta_net_recurrent_sane_inference_kernel_no_mask<ET>
      <<<dim3(H, B), kBlockThreads, 0, stream>>>(
          T, H, scale, reinterpret_cast<const ET *>(q.typed_data()),
          reinterpret_cast<const ET *>(k.typed_data()),
          reinterpret_cast<const ET *>(v.typed_data()), beta.typed_data(),
          tau.typed_data(), h0.typed_data(),
          reinterpret_cast<ET *>(o->typed_data()), ht->typed_data());

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
    return ffi::Error::Internal(
        std::string("delta_net_recurrent_sane_inference_no_mask error: ") +
        cudaGetErrorString(err));
  return ffi::Error::Success();
}

// 无 mask 版本的 FFI 注册宏与符号

#define DN_SANE_DEFINE_FWD_NO_MASK_HANDLER(SYMBOL, ET, DT)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(SYMBOL, (DeltaNetSaneFwdHostNoMask<ET, DT>),   \
                                ffi::Ffi::Bind()                               \
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()  \
                                    .Arg<ffi::Buffer<DT>>()       /* q */      \
                                    .Arg<ffi::Buffer<DT>>()       /* k */      \
                                    .Arg<ffi::Buffer<DT>>()       /* v */      \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* beta */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* tau */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* h0 */     \
                                    .Ret<ffi::Buffer<DT>>()       /* o */      \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* kv_mem */ \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* chkp */   \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* inv_q */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* inv_k */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* ht */,    \
                                {ffi::Traits::kCmdBufferCompatible})

#define DN_SANE_DEFINE_BWD_NO_MASK_HANDLER(SYMBOL, ET, DT)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(SYMBOL, (DeltaNetSaneBwdHostNoMask<ET, DT>),   \
                                ffi::Ffi::Bind()                               \
                                    .Ctx<ffi::PlatformStream<cudaStream_t>>()  \
                                    .Arg<ffi::Buffer<DT>>()       /* q */      \
                                    .Arg<ffi::Buffer<DT>>()       /* k */      \
                                    .Arg<ffi::Buffer<DT>>()       /* v */      \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* beta */   \
                                    .Arg<ffi::Buffer<DT>>()       /* dy */     \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* dht */    \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* kv_mem */ \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* inv_q */  \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* inv_k */  \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* chkp */   \
                                    .Arg<ffi::Buffer<ffi::F32>>() /* tau */    \
                                    .Ret<ffi::Buffer<DT>>()       /* dq */     \
                                    .Ret<ffi::Buffer<DT>>()       /* dk */     \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dv */     \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dbeta */  \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dh0 */    \
                                    .Ret<ffi::Buffer<ffi::F32>>() /* dtau */,  \
                                {ffi::Traits::kCmdBufferCompatible})

#define DN_SANE_DEFINE_INF_NO_MASK_HANDLER(SYMBOL, ET, DT)                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                               \
      SYMBOL, (DeltaNetSaneInferenceHostNoMask<ET, DT>),                       \
      ffi::Ffi::Bind()                                                         \
          .Ctx<ffi::PlatformStream<cudaStream_t>>()                            \
          .Arg<ffi::Buffer<DT>>()       /* q */                                \
          .Arg<ffi::Buffer<DT>>()       /* k */                                \
          .Arg<ffi::Buffer<DT>>()       /* v */                                \
          .Arg<ffi::Buffer<ffi::F32>>() /* beta */                             \
          .Arg<ffi::Buffer<ffi::F32>>() /* tau */                              \
          .Arg<ffi::Buffer<ffi::F32>>() /* h0 */                               \
          .Ret<ffi::Buffer<DT>>()       /* o */                                \
          .Ret<ffi::Buffer<ffi::F32>>() /* ht */,                              \
      {ffi::Traits::kCmdBufferCompatible})

DN_SANE_DEFINE_FWD_NO_MASK_HANDLER(DeltaNetRecurrentSaneFwdNoMaskBf16, bf,
                                   ffi::BF16);
DN_SANE_DEFINE_BWD_NO_MASK_HANDLER(DeltaNetRecurrentSaneBwdNoMaskBf16, bf,
                                   ffi::BF16);
DN_SANE_DEFINE_INF_NO_MASK_HANDLER(DeltaNetRecurrentSaneInferenceNoMaskBf16, bf,
                                   ffi::BF16);

DN_SANE_DEFINE_FWD_NO_MASK_HANDLER(DeltaNetRecurrentSaneFwdNoMaskF32, float,
                                   ffi::F32);
DN_SANE_DEFINE_BWD_NO_MASK_HANDLER(DeltaNetRecurrentSaneBwdNoMaskF32, float,
                                   ffi::F32);
DN_SANE_DEFINE_INF_NO_MASK_HANDLER(DeltaNetRecurrentSaneInferenceNoMaskF32,
                                   float, ffi::F32);


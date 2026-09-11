// DeltaNet recurrent PyTorch CUDA 训练/推理 kernel。

#include <cstdint>
#include <cuda_bf16.h>

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

// DeltaNet recurrent 训练前向 kernel。
//
// 每个 block 处理一个 (batch, head)，按时间步顺序扫描 T 步，在每个 chunk
// 末尾写出 state checkpoint，并保存反向所需的 kv_mem 与 q/k 逆范数。
// 线程 v 持有 state 的一列 state[:, v]（_K_ 个 float 寄存器），
// kv_mem 与输出均为线程内 K 维点积，无需跨线程归约。
// DeltaNet 无 decay 门：state 不做衰减，直接叠加 k_hat ⊗ delta。
//
// Args:
//   q_, k_: [B, H, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, H, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, H, T], float32, row-major。已在外部过 sigmoid。
//   h0_: [B, H, K, V], float32, row-major。初始 state。
//   kv_mem_: [B, H, T, V], float32, row-major。每步的 `k_t @ state_{t-1}`。
//   chkp_: [B, H, T//_CHUNK_LEN_, K, V], float32, row-major。chunk 末尾
//     state 快照。
//   inv_q_, inv_k_: [B, H, T], float32, row-major。q/k 的 L2 逆范数。
//   ht_: [B, H, K, V], float32, row-major。最终 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (H, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列，
//   线程 v < _K_ 同时承担 q/k 向量加载与归一化。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__ __launch_bounds__(kBlockThreads) void delta_net_recurrent_fwd_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ h0_,
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
      for (int j = 0; j < _K_; j++)
        kvm += state[j] * sh_khat[j];
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
        const int64_t cbase =
            (bh * num_chunks + t / _CHUNK_LEN_) * _K_ * _V_ + v;
#pragma unroll
        for (int j = 0; j < _K_; j++)
          chkp_[cbase + (int64_t)j * _V_] = state[j];
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

// DeltaNet recurrent 训练反向 kernel。
//
// 每个 block 处理一个 (batch, head)。先重新跑一遍前向得到 S_T，再从 T-1
// 倒序递推到 0；到达 chunk 边界时从 checkpoint 重载 S_{t+1}，抑制长序列
// 反向递推的浮点误差累积。
// 线程 v 持有 state 列与 dstate 列（各 _K_ 个 float 寄存器）。
// d_k_hat / d_q_tilde 需要沿 V 方向跨线程求和：每个 j 先由全部线程算出
// 本列贡献，warp 内 shuffle 归约后由 lane 0 写部分和，再由前 _K_ 个线程
// 跨 warp 汇总；部分和缓冲按 j 分块复用，占用与 V 无关，避免大 V 时
// shared memory 超过硬件上限。利用 dstate_new 与 state_old 的线性分解，
// 只需归约原始 state 与 dstate 各一遍，外加三个标量级 block 归约。
// DeltaNet 无 decay 门：S_{t-1} = S_t - k_hat ⊗ delta 直接恢复，
// dstate 回传不需要乘除 exp(g)。
//
// Args:
//   q_, k_: [B, H, T, K], bfloat16 或 float32, row-major。
//   v_, do_: [B, H, T, V], bfloat16 或 float32, row-major。do 为输出梯度。
//   beta_: [B, H, T], float32, row-major。
//   dht_: [B, H, K, V], float32, row-major。最终 state 梯度（无梯度时传零）。
//   kv_mem_: [B, H, T, V], float32, row-major。前向保存的 kv_mem。
//   inv_q_, inv_k_: [B, H, T], float32, row-major。前向保存的逆范数。
//   h0_: [B, H, K, V], float32, row-major。初始 state。
//   chkp_: [B, H, T//_CHUNK_LEN_, K, V], float32, row-major。state 快照。
//   dq_, dk_: [B, H, T, K], float32, row-major。输出梯度。
//   dv_: [B, H, T, V], float32, row-major。输出梯度。
//   dbeta_: [B, H, T], float32, row-major。输出梯度。
//   dh0_: [B, H, K, V], float32, row-major。初始 state 梯度。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (H, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state/dstate 第 v 列，
//   线程 v < _K_ 承担跨 warp 部分和的汇总。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，默认 16，T 必须被其整除。
template <typename ET>
__global__ __launch_bounds__(kBlockThreads) void delta_net_recurrent_bwd_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const ET *__restrict__ do_,
    const float *__restrict__ dht_, const float *__restrict__ kv_mem_,
    const float *__restrict__ inv_q_, const float *__restrict__ inv_k_,
    const float *__restrict__ h0_, const float *__restrict__ chkp_,
    float *__restrict__ dq_, float *__restrict__ dk_, float *__restrict__ dv_,
    float *__restrict__ dbeta_, float *__restrict__ dh0_) {
  __shared__ float sh_q[_K_], sh_k[_K_], sh_khat[_K_], sh_qt[_K_];
  __shared__ float sh_dkhat[_K_], sh_dqhat[_K_];
  __shared__ float sh_do[_V_], sh_delta[_V_], sh_dkv[_V_];
  __shared__ float sh_red[3][_V_];
  __shared__ float sh_scalar[5];
  // 跨线程归约的 warp 部分和缓冲，按 j 分块复用，占用与 V 无关。
  __shared__ float warp_part[3][kJBlock * kNumWarps];

  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;
  const int num_chunks = T / _CHUNK_LEN_;

  float state[_K_], carry[_K_];

  // 前向重算得到 S_T，只需 k 路径，逆范数直接读前向保存值。
  if (active) {
    const int64_t h0_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      state[j] = h0_[h0_base + (int64_t)j * _V_];
  }
  for (int t = 0; t < T; t++) {
    __syncthreads();
    if (v < _K_)
      sh_k[v] = to_float(k_[(bh * T + t) * _K_ + v]);
    const float v_val = active ? to_float(v_[(bh * T + t) * _V_ + v]) : 0.f;
    const float beta_t = beta_[bh * T + t];
    const float ik = inv_k_[bh * T + t];
    __syncthreads();
    if (v < _K_)
      sh_khat[v] = sh_k[v] * ik;
    __syncthreads();
    if (active) {
      float kvm = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++)
        kvm += state[j] * sh_khat[j];
      const float delta = beta_t * (v_val - kvm);
#pragma unroll
      for (int j = 0; j < _K_; j++)
        state[j] += sh_khat[j] * delta;
    }
  }

  if (active) {
    const int64_t dht_base = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      carry[j] = dht_[dht_base + (int64_t)j * _V_];
  }

  for (int t = T - 1; t >= 0; t--) {
    const int tp1 = t + 1;
    if (active && tp1 % _CHUNK_LEN_ == 0 && tp1 < T) {
      const int64_t cbase =
          (bh * num_chunks + tp1 / _CHUNK_LEN_ - 1) * _K_ * _V_ + v;
#pragma unroll
      for (int j = 0; j < _K_; j++)
        state[j] = chkp_[cbase + (int64_t)j * _V_];
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

      sh_do[v] = do_v;
      sh_delta[v] = delta_v;
      sh_dkv[v] = dkv_v;
      sh_red[0][v] = (v_val - kvm) * d_delta;
      sh_red[1][v] = do_v * delta_v;
      sh_red[2][v] = delta_v * dkv_v;
    }
    __syncthreads();

    if (v < 3) {
      float s = 0.f;
      for (int j = 0; j < _V_; j++)
        s += sh_red[v][j];
      sh_scalar[v] = s;
    }

    // 沿 V 方向对 state/dstate 列加权求和得 t1/t2/t3，再由线性分解合成
    // d_k_hat：sum(dstate_new * delta) = t1 + q_tilde * sum(do * delta)，
    // sum(state_old * d_kv_mem) = t2 - k_hat * sum(delta * d_kv_mem)。
    // 非活跃线程的贡献必须为 0，且不能读取未初始化的寄存器参与运算。
    const int warp = v >> 5, lane = v & 31;
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
          t1 + sh_qt[v] * sh_scalar[1] + t2 - sh_khat[v] * sh_scalar[2];
      sh_dqhat[v] = scale * t3;
    }
    __syncthreads();

    if (v == 0) {
      float qd = 0.f, kd = 0.f;
      for (int j = 0; j < _K_; j++) {
        qd += sh_q[j] * iq * sh_dqhat[j];
        kd += sh_khat[j] * sh_dkhat[j];
      }
      sh_scalar[3] = qd;
      sh_scalar[4] = kd;
    }
    __syncthreads();

    if (v < _K_) {
      dq_[qk_base + v] = iq * (sh_dqhat[v] - sh_q[v] * iq * sh_scalar[3]);
      dk_[qk_base + v] = ik * (sh_dkhat[v] - sh_khat[v] * sh_scalar[4]);
    }
    if (v == 0) {
      dbeta_[bh * T + t] = sh_scalar[0];
    }

    if (active) {
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        const float dnew = carry[j] + sh_qt[j] * do_v;
        state[j] = state[j] - sh_khat[j] * delta_v;
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

// DeltaNet recurrent 推理前向 kernel。
//
// 与训练前向数学一致，但不输出 kv_mem / checkpoint / 逆范数，减少推理
// 显存占用。每个 block 处理一个 (batch, head)。
//
// Args:
//   q_, k_: [B, H, T, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, H, T, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, H, T], float32, row-major。
//   h0_: [B, H, K, V], float32, row-major。初始 state。
//   ht_: [B, H, K, V], float32, row-major。最终 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (H, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，推理 kernel 不使用，仅为编译标识一致。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_inference_kernel(
    int T, int H, float scale, const ET *__restrict__ q_,
    const ET *__restrict__ k_, const ET *__restrict__ v_,
    const float *__restrict__ beta_, const float *__restrict__ h0_,
    ET *__restrict__ o_, float *__restrict__ ht_) {
  const int bb = blockIdx.y, hh = blockIdx.x, v = threadIdx.x;
  const bool active = v < _V_;
  const int64_t bh = (int64_t)bb * H + hh;

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
      sh_iq = 1.0f / sqrtf(sq + 1e-6f);
      sh_ik = 1.0f / sqrtf(sk + 1e-6f);
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
      for (int j = 0; j < _K_; j++)
        kvm += state[j] * sh_khat[j];
      const float delta = beta_t * (v_val - kvm);
      float out = 0.f;
#pragma unroll
      for (int j = 0; j < _K_; j++) {
        state[j] += sh_khat[j] * delta;
        out += state[j] * sh_qt[j];
      }
      o_[vo_base + v] = from_float<ET>(out);
    }
  }

  if (active) {
    const int64_t sbase = bh * _K_ * _V_ + v;
#pragma unroll
    for (int j = 0; j < _K_; j++)
      ht_[sbase + (int64_t)j * _V_] = state[j];
  }
}

// DeltaNet recurrent 单步 RNN 前向 kernel。
//
// 每个 block 处理一个 (batch, head) 的一个时间步，用于 decode。
// 输入没有时间维。
//
// Args:
//   q_, k_: [B, H, K], bfloat16 或 float32, row-major。
//   v_, o_: [B, H, V], bfloat16 或 float32, row-major。v 为输入，o 为输出。
//   beta_: [B, H], float32, row-major。
//   h0_: [B, H, K, V], float32, row-major。上一步 state。
//   ht_: [B, H, K, V], float32, row-major。下一步 state。
//   scale: query 缩放系数（1/sqrt(K)）。
//
// Grid / Block:
//   grid (H, B)，每个 block 对应一个 (head, batch)。
//   block (kBlockThreads,)，线程 v < _V_ 持有 state 第 v 列。
//
// 编译期宏:
//   _K_: key head size，不超过 1024。
//   _V_: value head size，不超过 1024。
//   _CHUNK_LEN_: chunk 长度，单步 kernel 不使用，仅为编译标识一致。
template <typename ET>
__global__
__launch_bounds__(kBlockThreads) void delta_net_recurrent_single_step_kernel(
    int H, float scale, const ET *__restrict__ q_, const ET *__restrict__ k_,
    const ET *__restrict__ v_, const float *__restrict__ beta_,
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
    sh_iq = 1.0f / sqrtf(sq + 1e-6f);
    sh_ik = 1.0f / sqrtf(sk + 1e-6f);
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
      ht_[sbase + (int64_t)j * _V_] = state[j];
      out += state[j] * sh_q[j] * iq * scale;
    }
    o_[bh * _V_ + v] = from_float<ET>(out);
  }
}

// C 接口启动函数

template <typename ET>
void cuda_dn_forward(int B, int T, int H, float scale, const ET *q, const ET *k,
                     const ET *v, const float *beta, const float *h0, ET *o,
                     float *kv_mem, float *chkp, float *inv_q, float *inv_k,
                     float *ht) {
  delta_net_recurrent_fwd_kernel<ET><<<dim3(H, B), kBlockThreads>>>(
      T, H, scale, q, k, v, beta, h0, o, kv_mem, chkp, inv_q, inv_k, ht);
}

template <typename ET>
void cuda_dn_backward(int B, int T, int H, float scale, const ET *q,
                      const ET *k, const ET *v, const float *beta,
                      const ET *dout, const float *dht, const float *kv_mem,
                      const float *inv_q, const float *inv_k, const float *h0,
                      const float *chkp, float *dq, float *dk, float *dv,
                      float *dbeta, float *dh0) {
  delta_net_recurrent_bwd_kernel<ET><<<dim3(H, B), kBlockThreads>>>(
      T, H, scale, q, k, v, beta, dout, dht, kv_mem, inv_q, inv_k, h0, chkp, dq,
      dk, dv, dbeta, dh0);
}

template <typename ET>
void cuda_dn_forward_inference(int B, int T, int H, float scale, const ET *q,
                               const ET *k, const ET *v, const float *beta,
                               const float *h0, ET *o, float *ht) {
  delta_net_recurrent_inference_kernel<ET>
      <<<dim3(H, B), kBlockThreads>>>(T, H, scale, q, k, v, beta, h0, o, ht);
}

template <typename ET>
void cuda_dn_single_step(int B, int H, float scale, const ET *q, const ET *k,
                         const ET *v, const float *beta, const float *h0, ET *o,
                         float *ht) {
  delta_net_recurrent_single_step_kernel<ET>
      <<<dim3(H, B), kBlockThreads>>>(H, scale, q, k, v, beta, h0, o, ht);
}

// 显式实例化 bfloat16 与 float32 两个版本，供 .cpp 侧按输入 dtype 分发。
template void cuda_dn_forward<bf>(int, int, int, float, const bf *, const bf *,
                                  const bf *, const float *, const float *,
                                  bf *, float *, float *, float *, float *,
                                  float *);
template void cuda_dn_forward<float>(int, int, int, float, const float *,
                                     const float *, const float *,
                                     const float *, const float *, float *,
                                     float *, float *, float *, float *,
                                     float *);
template void cuda_dn_backward<bf>(int, int, int, float, const bf *, const bf *,
                                   const bf *, const float *, const bf *,
                                   const float *, const float *, const float *,
                                   const float *, const float *, const float *,
                                   float *, float *, float *, float *, float *);
template void cuda_dn_backward<float>(int, int, int, float, const float *,
                                      const float *, const float *,
                                      const float *, const float *,
                                      const float *, const float *,
                                      const float *, const float *,
                                      const float *, const float *, float *,
                                      float *, float *, float *, float *);
template void cuda_dn_forward_inference<bf>(int, int, int, float, const bf *,
                                            const bf *, const bf *,
                                            const float *, const float *, bf *,
                                            float *);
template void cuda_dn_forward_inference<float>(int, int, int, float,
                                               const float *, const float *,
                                               const float *, const float *,
                                               const float *, float *, float *);
template void cuda_dn_single_step<bf>(int, int, float, const bf *, const bf *,
                                      const bf *, const float *, const float *,
                                      bf *, float *);
template void cuda_dn_single_step<float>(int, int, float, const float *,
                                         const float *, const float *,
                                         const float *, const float *, float *,
                                         float *);

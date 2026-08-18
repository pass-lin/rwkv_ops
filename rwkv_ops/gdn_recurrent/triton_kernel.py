"""Gated DeltaNet recurrent Triton kernel（共享实现）。"""

import triton
import triton.language as tl


@triton.jit
def gated_delta_net_recurrent_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    scale,
    B,
    H,
    T,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Gated DeltaNet recurrent 前向 Triton kernel。

    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。
    状态 state 在 kernel 内保持 [K, V] 布局，K 方向一次性加载，V 方向分块。

    Args:
        q, k: [B, H, T, K]，与输入同 dtype，row-major。
        v, o: [B, H, T, V]，v 与 o 同 dtype，row-major。
        g: [B, H, T]，log-space decay，row-major。
        beta: [B, H, T]，已在外部过 sigmoid 的写入强度，row-major。
        h0: [B, H, K, V]，float32，初始 state。
        ht: [B, H, K, V]，float32，最终 state（仅当 STORE_FINAL_STATE=True 时写入）。
        scale: float，query 缩放系数（通常为 1/sqrt(K)）。
        B, H, T: int，batch/head/sequence 大小。
        K, V: tl.constexpr，head 维度大小。
        BK, BV: tl.constexpr，block 大小（分别为 next_power_of_2(K) 与 V 分块大小）。

    指针算术全部使用 64 位整数，避免大 tensor 时 32 位偏移溢出。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    T_i64 = tl.cast(T, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64

    base_qk = bh * T_i64 * K_i64 + o_k
    base_vo = bh * T_i64 * V_i64 + o_v
    base_gb = bh * T_i64
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)

    for t in range(T):
        t_i64 = t.to(tl.int64)

        p_q = q + base_qk + t_i64 * K_i64
        p_k = k + base_qk + t_i64 * K_i64
        p_v = v + base_vo + t_i64 * V_i64
        p_g = g + base_gb + t_i64
        p_beta = beta + base_gb + t_i64
        p_o = o + base_vo + t_i64 * V_i64

        b_q = tl.load(p_q, mask=mask_k, other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0.0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0.0).to(tl.float32)
        b_g = tl.load(p_g).to(tl.float32)
        b_beta = tl.load(p_beta).to(tl.float32)

        # L2 归一化，与 native_keras_op 保持一致。
        b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
        b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
        b_q = b_q * scale

        # log-space decay。
        b_h = b_h * tl.exp(b_g)

        # kv_mem = sum_K(state * k)。
        kv_mem = tl.sum(b_h * b_k[:, None], axis=0)

        # delta = beta * (v - kv_mem)。
        b_v = b_beta * (b_v - kv_mem)

        # state += k^T delta。
        b_h = b_h + b_k[:, None] * b_v[None, :]

        # out = sum_K(state * q)。
        b_o = tl.sum(b_h * b_q[:, None], axis=0)
        tl.store(p_o, b_o.to(o.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        tl.store(ht + base_state, b_h.to(ht.dtype.element_ty), mask=mask_h)


# inference 前向与训练前向数学一致，直接复用。
gated_delta_net_recurrent_inference_fwd_kernel = gated_delta_net_recurrent_fwd_kernel


@triton.jit
def gated_delta_net_recurrent_single_step_fwd_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    h0,
    ht,
    scale,
    B,
    H,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """Gated DeltaNet recurrent 单步 RNN 前向 Triton kernel。

    每个 program 处理一个 (batch, head) 与 V 方向的一个 block。
    输入没有时间维：q/k 为 [B, H, K]，v/o 为 [B, H, V]，g/beta 为 [B, H]。

    Args:
        q, k: [B, H, K]，与输入同 dtype，row-major。
        v, o: [B, H, V]，v 与 o 同 dtype，row-major。
        g: [B, H]，log-space decay，row-major。
        beta: [B, H]，已在外部过 sigmoid 的写入强度，row-major。
        h0: [B, H, K, V]，float32，初始 state。
        ht: [B, H, K, V]，float32，下一步 state（仅当 STORE_FINAL_STATE=True 时写入）。
        scale: float，query 缩放系数。
        B, H: int，batch/head 大小。
        K, V: tl.constexpr，head 维度大小。
        BK, BV: tl.constexpr，block 大小。
    """
    pid = tl.program_id(0)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    i_b = i_nh // H
    i_h = i_nh % H

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)

    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    i_b_i64 = i_b.to(tl.int64)
    i_h_i64 = i_h.to(tl.int64)
    H_i64 = tl.cast(H, tl.int64)
    K_i64 = tl.cast(K, tl.int64)
    V_i64 = tl.cast(V, tl.int64)

    bh = i_b_i64 * H_i64 + i_h_i64

    base_qk = bh * K_i64 + o_k
    base_vo = bh * V_i64 + o_v
    base_state = bh * K_i64 * V_i64 + o_k[:, None] * V_i64 + o_v[None, :]

    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        b_h += tl.load(h0 + base_state, mask=mask_h, other=0.0).to(tl.float32)

    b_q = tl.load(q + base_qk, mask=mask_k, other=0.0).to(tl.float32)
    b_k = tl.load(k + base_qk, mask=mask_k, other=0.0).to(tl.float32)
    b_v = tl.load(v + base_vo, mask=mask_v, other=0.0).to(tl.float32)
    b_g = tl.load(g + bh).to(tl.float32)
    b_beta = tl.load(beta + bh).to(tl.float32)

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    b_h = b_h * tl.exp(b_g)
    kv_mem = tl.sum(b_h * b_k[:, None], axis=0)
    b_v = b_beta * (b_v - kv_mem)
    b_h = b_h + b_k[:, None] * b_v[None, :]
    b_o = tl.sum(b_h * b_q[:, None], axis=0)
    tl.store(o + base_vo, b_o.to(o.dtype.element_ty), mask=mask_v)

    if STORE_FINAL_STATE:
        tl.store(ht + base_state, b_h.to(ht.dtype.element_ty), mask=mask_h)

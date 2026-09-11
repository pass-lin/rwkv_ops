"""DeltaNet chunkwise SANE 状态递推 Triton kernel。

在每个 chunk 末尾对跨 chunk 传递的 state 执行 State Anomaly Neutralization：

    s_sane = tau * tanh(s / tau)
    s_next = mask * s_sane + (1 - mask) * s

输出 h 保存的是 SANE 后的状态（即下一个 chunk 的进入状态）。
"""

import torch
import triton
import triton.language as tl

from ...triton_utils import _sane_transform


@triton.autotune(
    configs=[
        triton.Config({"BV": 64}, num_warps=4, num_stages=2),
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _delta_net_chunk_fwd_h_sane_kernel(
    k_ptr,
    w_ptr,
    u_ptr,
    tau_ptr,
    mask_ptr,
    h0_ptr,
    B,
    H,
    T,
    K,
    V,
    h_ptr,
    v_new_ptr,
    ht_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    """chunk 间状态递推前向 SANE kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维顺序扫描。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        w_ptr: [B, H, T, K]，WY 表示 w。
        u_ptr: [B, H, T, V]，WY 表示 u。
        tau_ptr: [B, H, T//C]，SANE 阈值。
        mask_ptr: [B, T//C]，per-chunk mask；USE_MASK=False 时不读。
        h0_ptr: [B, H, K, V]，初始状态。
        B, H, T, K, V: 维度。
        h_ptr: [B, H, T//C, K, V]，chunk 级进入状态（SANE 后）。
        v_new_ptr: [B, H, T, V]，修正后的 value。
        ht_ptr: [B, H, K, V]，最终状态（最后一个 chunk SANE 后）。
        C: chunk 长度，编译期常量。
        BV: V 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    h = i_nh % H
    b = i_nh // H

    N = T // C

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    m_v = o_v < V
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)

    base_bh = (b * H + h).to(tl.int64)
    base_k = base_bh * T * K
    base_v = base_bh * T * V
    base_h = base_bh * N * K * V
    base_tau = base_bh * N
    base_mask = b * N

    # 加载初始状态
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    for i_t in range(N):
        i_t_int64 = i_t.to(tl.int64)

        # 保存进入当前 chunk 的状态 h（SANE 后的状态）
        p_h = h_ptr + base_h + i_t_int64 * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(h_ptr.dtype.element_ty), mask=m_h)

        # 加载 w, u, k 当前 chunk
        o_c = tl.arange(0, C)
        m_c = o_c < C

        p_w = w_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]
        p_u = u_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        p_k = k_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]

        b_w = tl.load(p_w, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )
        b_u = tl.load(p_u, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )

        # v_new = u - w @ h
        b_v_new = b_u - tl.dot(b_w, b_h.to(b_w.dtype), allow_tf32=False)

        # 保存 v_new
        p_v_new = (
            v_new_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        )
        tl.store(
            p_v_new,
            b_v_new.to(v_new_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新状态：h_raw = h + k^T @ v_new
        b_h_raw = b_h + tl.dot(
            tl.trans(b_k).to(b_v_new.dtype), b_v_new, allow_tf32=False
        )

        # 读取 tau
        p_tau = tau_ptr + base_tau + i_t_int64
        b_tau = tl.load(p_tau).to(tl.float32)
        b_tau = tl.maximum(b_tau, 1e-6)

        # SANE: h_sane = tau * tanh(h_raw / tau)
        b_h_sane = _sane_transform(b_h_raw, b_tau)

        if USE_MASK:
            # mask 形状 [B, N]，按 batch 索引
            p_mask = mask_ptr + base_mask + i_t_int64
            b_mask = tl.load(p_mask).to(tl.float32)
            b_mask = tl.where(b_mask > 0.0, 1.0, 0.0)
            b_h = b_h_raw * (1.0 - b_mask) + b_h_sane * b_mask
        else:
            b_h = b_h_sane

    if STORE_FINAL_STATE:
        p_ht = ht_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(ht_ptr.dtype.element_ty), mask=m_h)


@triton.autotune(
    configs=[
        triton.Config({"BV": 64}, num_warps=4, num_stages=2),
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _delta_net_chunk_fwd_h_sane_kernel_no_mask(
    k_ptr,
    w_ptr,
    u_ptr,
    tau_ptr,
    h0_ptr,
    B,
    H,
    T,
    K,
    V,
    h_ptr,
    v_new_ptr,
    ht_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """chunk 间状态递推前向 SANE 无 mask kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维顺序扫描。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        w_ptr: [B, H, T, K]，WY 表示 w。
        u_ptr: [B, H, T, V]，WY 表示 u。
        tau_ptr: [B, H, T//C]，SANE 阈值。
        h0_ptr: [B, H, K, V]，初始状态。
        B, H, T, K, V: 维度。
        h_ptr: [B, H, T//C, K, V]，chunk 级进入状态（SANE 后）。
        v_new_ptr: [B, H, T, V]，修正后的 value。
        ht_ptr: [B, H, K, V]，最终状态（最后一个 chunk SANE 后）。
        C: chunk 长度，编译期常量。
        BV: V 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_nh = pid // NV
    h = i_nh % H
    b = i_nh // H

    N = T // C

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    m_v = o_v < V
    m_h = (o_k[:, None] < K) & (o_v[None, :] < V)

    base_bh = (b * H + h).to(tl.int64)
    base_k = base_bh * T * K
    base_v = base_bh * T * V
    base_h = base_bh * N * K * V
    base_tau = base_bh * N

    # 加载初始状态
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    for i_t in range(N):
        i_t_int64 = i_t.to(tl.int64)

        # 保存进入当前 chunk 的状态 h（SANE 后的状态）
        p_h = h_ptr + base_h + i_t_int64 * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(h_ptr.dtype.element_ty), mask=m_h)

        # 加载 w, u, k 当前 chunk
        o_c = tl.arange(0, C)
        m_c = o_c < C

        p_w = w_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]
        p_u = u_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        p_k = k_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]

        b_w = tl.load(p_w, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )
        b_u = tl.load(p_u, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )

        # v_new = u - w @ h
        b_v_new = b_u - tl.dot(b_w, b_h.to(b_w.dtype), allow_tf32=False)

        # 保存 v_new
        p_v_new = (
            v_new_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        )
        tl.store(
            p_v_new,
            b_v_new.to(v_new_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新状态：h_raw = h + k^T @ v_new
        b_h_raw = b_h + tl.dot(
            tl.trans(b_k).to(b_v_new.dtype), b_v_new, allow_tf32=False
        )

        # 读取 tau
        p_tau = tau_ptr + base_tau + i_t_int64
        b_tau = tl.load(p_tau).to(tl.float32)
        b_tau = tl.maximum(b_tau, 1e-6)

        # SANE: h_sane = tau * tanh(h_raw / tau)
        b_h_sane = _sane_transform(b_h_raw, b_tau)

        b_h = b_h_sane

    if STORE_FINAL_STATE:
        p_ht = ht_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(ht_ptr.dtype.element_ty), mask=m_h)


def delta_net_chunk_fwd_h_sane_no_mask(
    k,
    w,
    u,
    tau,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
):
    """chunk 间状态递推 SANE 封装。

    Args:
        k: [B, H, T, K]，已归一化的 key。
        w: [B, H, T, K]，WY 表示 w。
        u: [B, H, T, V]，WY 表示 u。
        tau: [B, H, T//C]，SANE 阈值。
        mask: [B, T//C] 或 None，per-chunk mask。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，可选。
        output_final_state: bool。
        chunk_size: int。

    Returns:
        h: [B, H, T//chunk_size, K, V]，SANE 后进入状态。
        v_new: [B, H, T, V]。
        final_state: [B, H, K, V]，当 output_final_state=True。
    """
    B, H, T, K = k.shape
    V = u.shape[-1]
    C = chunk_size
    N = T // C

    h = torch.empty(B, H, N, K, V, dtype=torch.float32, device=k.device)
    v_new = torch.empty_like(u)

    if initial_state is not None:
        h0 = initial_state.to(torch.float32)
        if h0.shape[0] == 1 and B > 1:
            h0 = h0.expand(B, *h0.shape[1:]).contiguous()
    else:
        h0 = None

    ht = (
        torch.empty(B, H, K, V, dtype=torch.float32, device=k.device)
        if output_final_state
        else None
    )

    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (B * H * triton.cdiv(V, meta["BV"]),)

    _delta_net_chunk_fwd_h_sane_kernel[grid](
        k,
        w,
        u,
        tau,
        h0,
        B,
        H,
        T,
        K,
        V,
        h,
        v_new,
        ht,
        C=C,
        BK=BK,
        USE_INITIAL_STATE=(h0 is not None),
        STORE_FINAL_STATE=output_final_state,
    )
    return h, v_new, ht

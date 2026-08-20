"""Gated DeltaNet chunkwise chunk 间状态递推 Triton kernel。

计算每个 chunk 的输入状态 `h`、修正值 `v_new` 与最终状态。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2]
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _gdn_chunk_fwd_h_kernel(
    k_ptr,
    w_ptr,
    u_ptr,
    g_ptr,
    h_ptr,
    v_new_ptr,
    h0_ptr,
    ht_ptr,
    B,
    H,
    T,
    K,
    V,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
):
    """chunk 间状态递推前向 kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维顺序扫描。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        w_ptr: [B, H, T, K]，WY 表示 w。
        u_ptr: [B, H, T, V]，WY 表示 u。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        h_ptr: [B, H, T//C, K, V]，chunk 级输入状态 checkpoint。
        v_new_ptr: [B, H, T, V]，输出修正后的 value。
        h0_ptr: [B, H, K, V]，初始状态。
        ht_ptr: [B, H, K, V]，最终状态。
        B, H, T, K, V: 维度。
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

    base_k = ((b * H + h) * T) * K
    base_v = ((b * H + h) * T) * V
    base_gb = (b * H + h) * T
    base_h = ((b * H + h) * N) * K * V

    # 加载初始状态
    b_h = tl.zeros([BK, BV], dtype=tl.float32)
    if USE_INITIAL_STATE:
        p_h0 = h0_ptr + (b * H + h) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h += tl.load(p_h0, mask=m_h, other=0.0).to(tl.float32)

    for i_t in range(N):
        i_t_int64 = i_t.to(tl.int64)

        # 保存进入当前 chunk 的状态 h
        p_h = h_ptr + base_h + i_t_int64 * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_h, b_h.to(h_ptr.dtype.element_ty), mask=m_h)

        # 加载 w, u, k, g 当前 chunk
        o_c = tl.arange(0, C)
        m_c = o_c < C

        p_w = w_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]
        p_u = u_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        p_k = k_ptr + base_k + i_t_int64 * C * K + o_c[:, None] * K + o_k[None, :]
        p_g = g_ptr + base_gb + i_t_int64 * C + o_c

        b_w = tl.load(p_w, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )
        b_u = tl.load(p_u, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_c[:, None] & (o_k[None, :] < K), other=0.0).to(
            tl.float32
        )
        b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)

        # v_new = u - w @ h
        b_v_new = b_u - tl.dot(b_w, b_h.to(b_w.dtype), allow_tf32=False)

        # 应用 decay：v_new_scaled = v_new * exp(g_last - g)
        g_last = tl.load(g_ptr + base_gb + i_t_int64 * C + C - 1).to(tl.float32)
        decay_factor = tl.exp(g_last - b_g)
        b_v_new_scaled = b_v_new * decay_factor[:, None]

        # 保存 v_new
        p_v_new = (
            v_new_ptr + base_v + i_t_int64 * C * V + o_c[:, None] * V + o_v[None, :]
        )
        tl.store(
            p_v_new,
            b_v_new.to(v_new_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新状态：h = h * exp(g_last) + k^T @ v_new_scaled
        b_h = b_h * tl.exp(g_last)
        b_h += tl.dot(
            tl.trans(b_k).to(b_v_new_scaled.dtype), b_v_new_scaled, allow_tf32=False
        )

    if STORE_FINAL_STATE:
        p_ht = ht_ptr + (b * H + h) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_ht, b_h.to(ht_ptr.dtype.element_ty), mask=m_h)


def gdn_chunk_fwd_h(
    k, w, u, g, initial_state=None, output_final_state=False, chunk_size=64
):
    """chunk 间状态递推封装。

    Args:
        k: [B, H, T, K]，已归一化的 key。
        w: [B, H, T, K]，WY 表示 w。
        u: [B, H, T, V]，WY 表示 u。
        g: [B, H, T]，cumsum 后的 log-space decay。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，可选。
        output_final_state: bool。
        chunk_size: int。

    Returns:
        h: [B, H, T//chunk_size, K, V]，状态 checkpoint。
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

    _gdn_chunk_fwd_h_kernel[grid](
        k,
        w,
        u,
        g,
        h,
        v_new,
        h0,
        ht,
        B,
        H,
        T,
        K,
        V,
        C=C,
        BK=BK,
        USE_INITIAL_STATE=(h0 is not None),
        STORE_FINAL_STATE=output_final_state,
    )
    return h, v_new, ht

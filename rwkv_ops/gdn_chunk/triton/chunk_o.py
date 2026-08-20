"""Gated DeltaNet chunkwise 最终输出 Triton kernel。

计算 `o = scale * (q @ h + causal(q @ k^T * decay) @ v_new)`。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": BK, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for BV in [64, 128]
        for num_warps in [2, 4]
        for num_stages in [2]
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _gdn_chunk_fwd_o_kernel(
    q_ptr,
    k_ptr,
    v_new_ptr,
    h_ptr,
    g_ptr,
    o_ptr,
    scale,
    B,
    H,
    T,
    K,
    V,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """chunk 最终输出 kernel。

    每个 program 处理一个 (batch, head, chunk, V-block)。

    Args:
        q_ptr: [B, H, T, K]，已归一化的 query。
        k_ptr: [B, H, T, K]，已归一化的 key。
        v_new_ptr: [B, H, T, V]，修正后的 value。
        h_ptr: [B, H, T//C, K, V]，chunk 级输入状态 checkpoint。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        o_ptr: [B, H, T, V]，输出。
        scale: float，query 缩放系数 `1/sqrt(K)`。
        B, H, T, K, V: 维度。
        C: chunk 长度，编译期常量。
        BK, BV: K/V 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_t = (pid // NV) % (T // C)
    tmp = pid // (NV * (T // C))
    h = tmp % H
    b = tmp // H

    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, BK)
    o_c = tl.arange(0, C)
    m_v = o_v < V
    m_k = o_k < K
    m_c = o_c < C

    base = (b * H + h) * T
    base_k = base * K + i_t * C * K
    base_v = base * V + i_t * C * V
    base_gb = base + i_t * C
    base_h = ((b * H + h) * (T // C) + i_t) * K * V

    # 加载 v_new 块 [C, BV]
    p_v = v_new_ptr + base_v + o_c[:, None] * V + o_v[None, :]
    b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)

    # 加载 g [C]
    p_g = g_ptr + base_gb + o_c
    b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)

    # 计算 q @ h 和 q @ k^T，沿 K 维累加
    b_o = tl.zeros([C, BV], dtype=tl.float32)
    b_A = tl.zeros([C, C], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_q = q_ptr + base_k + o_c[:, None] * K + o_k[None, :]
        p_k = k_ptr + base_k + o_c[:, None] * K + o_k[None, :]
        p_h = h_ptr + base_h + o_k[:, None] * V + o_v[None, :]

        b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        # [C, K] @ [K, BV] -> [C, BV]
        b_o += tl.dot(b_q, b_h.to(b_q.dtype), allow_tf32=False)
        # [C, K] @ [K, C] -> [C, C]
        b_A += tl.dot(b_q, tl.trans(b_k), allow_tf32=False)

    # 对 inter 项应用 exp(g)
    b_o = b_o * tl.exp(b_g)[:, None]

    # 对 intra 项应用 causal decay（包含对角线）
    g_row = b_g[:, None]
    g_col = b_g[None, :]
    decay = tl.exp(g_row - g_col)
    mask_lower = o_c[:, None] >= o_c[None, :]
    b_A = tl.where(mask_lower & m_c[:, None] & m_c[None, :], b_A * decay, 0.0)

    # intra attention: [C, C] @ [C, BV] -> [C, BV]
    b_o += tl.dot(b_A.to(b_v.dtype), b_v, allow_tf32=False)

    # 缩放
    scale_f32 = tl.cast(scale, tl.float32)
    b_o = b_o * scale_f32

    # 保存输出
    p_o = o_ptr + base_v + o_c[:, None] * V + o_v[None, :]
    tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=m_c[:, None] & m_v[None, :])


def gdn_chunk_fwd_o(q, k, v_new, h, g, chunk_size=64):
    """chunk 最终输出封装。

    Args:
        q: [B, H, T, K]，已归一化的 query。
        k: [B, H, T, K]，已归一化的 key。
        v_new: [B, H, T, V]。
        h: [B, H, T//C, K, V]。
        g: [B, H, T]，cumsum 后的 log-space decay。
        chunk_size: int。

    Returns:
        o: [B, H, T, V]。
    """
    B, H, T, K = q.shape
    V = v_new.shape[-1]
    C = chunk_size

    o = torch.empty_like(v_new)
    scale = K**-0.5

    def grid(meta):
        return (B * H * (T // C) * triton.cdiv(V, meta["BV"]),)

    _gdn_chunk_fwd_o_kernel[grid](
        q,
        k,
        v_new,
        h,
        g,
        o,
        scale,
        B,
        H,
        T,
        K,
        V,
        C=C,
    )
    return o

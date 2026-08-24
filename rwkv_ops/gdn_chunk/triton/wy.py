"""Gated DeltaNet chunkwise WY 表示 Triton kernel。

由 `A = (I - L)^{-1}` 重算 `w` 与 `u`：
  `u = A @ (beta * v)`
  `w = A @ (beta * k * exp(g))`

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": 64, "BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [64, 128]
        for num_warps in [2, 4]
        for num_stages in [2, 3]
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _gdn_chunk_recompute_w_u_fwd_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    A_ptr,
    g_ptr,
    B,
    H,
    T,
    K,
    V,
    w_ptr,
    u_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """重算 WY 表示 `w, u`。

    每个 program 处理一个 (batch, head, chunk) 与 K/V 方向上的 block。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        v_ptr: [B, H, T, V]，value。
        beta_ptr: [B, H, T]，post-sigmoid beta。
        A_ptr: [B, H, T//C, C, C]，`(I - L)^{-1}`。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        B, H, T, K, V: 维度。
        w_ptr: [B, H, T, K]，输出 w。
        u_ptr: [B, H, T, V]，输出 u。
        C: chunk 长度，编译期常量。
        BK, BV: K/V 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    N = T // C
    n = pid % N
    tmp = pid // N
    h = tmp % H
    b = tmp // H

    base_k = ((b * H + h) * T + n * C) * K
    base_v = ((b * H + h) * T + n * C) * V
    base_gb = (b * H + h) * T + n * C
    base_A = ((b * H + h) * N + n) * C * C

    o_c = tl.arange(0, C)
    m_c = o_c < C

    p_beta = beta_ptr + base_gb + o_c
    p_g = g_ptr + base_gb + o_c
    b_beta = tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)
    b_g_exp = tl.exp(b_g)

    p_A = A_ptr + base_A + o_c[:, None] * C + o_c[None, :]
    b_A = tl.load(p_A, mask=m_c[:, None] & m_c[None, :], other=0.0).to(tl.float32)

    # 计算 u = A @ (beta * v)
    o_v = tl.arange(0, BV)
    m_v = o_v < V
    p_v = v_ptr + base_v + o_c[:, None] * V + o_v[None, :]
    b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    b_vb = b_v * b_beta[:, None]
    b_u = tl.dot(b_A, b_vb, allow_tf32=False)
    p_u = u_ptr + base_v + o_c[:, None] * V + o_v[None, :]
    tl.store(p_u, b_u.to(u_ptr.dtype.element_ty), mask=m_c[:, None] & m_v[None, :])

    # 计算 w = A @ (beta * k * exp(g))
    o_k = tl.arange(0, BK)
    m_k = o_k < K
    p_k = k_ptr + base_k + o_c[:, None] * K + o_k[None, :]
    b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_kb = b_k * b_beta[:, None] * b_g_exp[:, None]
    b_w = tl.dot(b_A, b_kb, allow_tf32=False)
    p_w = w_ptr + base_k + o_c[:, None] * K + o_k[None, :]
    tl.store(p_w, b_w.to(w_ptr.dtype.element_ty), mask=m_c[:, None] & m_k[None, :])


def gdn_chunk_recompute_w_u(k, v, beta, A, g, chunk_size=64):
    """重算 WY 表示 `w, u` 封装。

    Args:
        k: [B, H, T, K]，已归一化的 key。
        v: [B, H, T, V]。
        beta: [B, H, T]，post-sigmoid beta。
        A: [B, H, T//C, C, C]，`(I - L)^{-1}`。
        g: [B, H, T]，cumsum 后的 log-space decay。
        chunk_size: int。

    Returns:
        w: [B, H, T, K]。
        u: [B, H, T, V]。
    """
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    N = T // C

    w = torch.empty_like(k)
    u = torch.empty_like(v)

    grid = (B * H * N,)
    _gdn_chunk_recompute_w_u_fwd_kernel[grid](
        k,
        v,
        beta,
        A,
        g,
        B,
        H,
        T,
        K,
        V,
        w,
        u,
        C=C,
    )
    return w, u

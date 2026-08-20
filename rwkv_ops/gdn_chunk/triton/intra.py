"""Gated DeltaNet chunkwise intra-chunk 下三角 solve Triton kernel。

计算 `A = (I + L)^{-1}`，其中 `L_ij = beta_i * k_i^T k_j * exp(g_i - g_j)`（严格下三角）。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BK": BK}, num_warps=num_warps, num_stages=num_stages)
        for BK in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [1, 2]
    ],
    key=["K", "C"],
)
@triton.jit
def _gdn_chunk_fwd_intra_kernel(
    k_ptr,
    g_ptr,
    beta_ptr,
    A_ptr,
    B,
    H,
    T,
    K,
    C: tl.constexpr,
    BK: tl.constexpr,
):
    """Intra-chunk 下三角矩阵求逆 kernel。

    每个 program 处理一个 (batch, head, chunk)。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        beta_ptr: [B, H, T]，post-sigmoid beta。
        A_ptr: [B, H, T//C, C, C]，输出 `(I + L)^{-1}`。
        B, H, T, K: 维度。
        C: chunk 长度，编译期常量。
        BK: K 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    N = T // C
    n = pid % N
    tmp = pid // N
    h = tmp % H
    b = tmp // H

    base_k = ((b * H + h) * T + n * C) * K
    base_gb = (b * H + h) * T + n * C
    base_A = ((b * H + h) * N + n) * C * C

    o_c = tl.arange(0, C)
    m_c = o_c < C

    p_beta = beta_ptr + base_gb + o_c
    p_g = g_ptr + base_gb + o_c
    b_beta = tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32)
    b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)

    # k @ k^T，沿 K 维累加
    b_kkt = tl.zeros([C, C], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        p_k = k_ptr + base_k + o_c[:, None] * K + o_k[None, :]
        b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_kkt += tl.dot(b_k, tl.trans(b_k))

    # L = beta_i * k_i^T k_j * exp(g_i - g_j)，严格下三角
    decay = tl.exp(b_g[:, None] - b_g[None, :])
    b_L = b_kkt * decay * b_beta[:, None]
    mask_lower = o_c[:, None] > o_c[None, :]
    b_L = tl.where(mask_lower, b_L, 0.0)

    # A = (I + L)^{-1}，前向替换
    # A[i,j] = delta_ij - sum_m L[i,m] * A[m,j]
    b_A = tl.zeros([C, C], dtype=tl.float32)
    for i in range(C):
        # 用 mask 提取 L 的第 i 行，避免动态索引
        mask_i = o_c[:, None] == i
        row_L = tl.sum(tl.where(mask_i, b_L, 0.0), axis=0)
        row_i = -tl.sum(row_L[:, None] * b_A, axis=0)
        row_i = row_i + tl.where(o_c == i, 1.0, 0.0)
        row_i = tl.where(o_c <= i, row_i, 0.0)
        b_A = tl.where((o_c[:, None] == i) & (o_c[None, :] <= i), row_i[None, :], b_A)

    # 存储 A，只存下三角（含对角线）即可
    p_A = A_ptr + base_A + o_c[:, None] * C + o_c[None, :]
    tl.store(p_A, b_A.to(A_ptr.dtype.element_ty), mask=m_c[:, None] & m_c[None, :])


def gdn_chunk_fwd_intra(k, g, beta, chunk_size=64):
    """Intra-chunk 下三角矩阵求逆封装。

    Args:
        k: [B, H, T, K]，已归一化的 key。
        g: [B, H, T]，cumsum 后的 log-space decay。
        beta: [B, H, T]，post-sigmoid beta。
        chunk_size: int，chunk 长度。

    Returns:
        A: [B, H, T//chunk_size, chunk_size, chunk_size]，`(I - L)^{-1}`。
    """
    B, H, T, K = k.shape
    assert T % chunk_size == 0
    C = chunk_size
    N = T // C
    A = torch.empty(B, H, N, C, C, dtype=k.dtype, device=k.device)

    grid = (B * H * N,)
    _gdn_chunk_fwd_intra_kernel[grid](
        k,
        g,
        beta,
        A,
        B,
        H,
        T,
        K,
        C=C,
    )
    return A

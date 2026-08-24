"""Gated DeltaNet chunkwise 局部 dv Triton kernel。

从 `q, k, g, do` 计算每个 chunk 内的局部 `dv`：
  `dv_local = causal(k @ q^T * decay) @ do * scale`

chunk 间的状态梯度项 `k @ dh * decay_state` 在 `chunk_bwd_dhu` 中累加。

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
def _gdn_chunk_bwd_dv_local_kernel(
    q_ptr,
    k_ptr,
    g_ptr,
    do_ptr,
    B,
    H,
    T,
    K,
    V,
    scale,
    dv_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """局部 dv backward kernel。

    每个 program 处理一个 (batch, head, chunk, V-block)。

    Args:
        q_ptr, k_ptr: [B, H, T, K]，已归一化的 query/key。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        do_ptr: [B, H, T, V]，输出梯度。
        B, H, T, K, V: 维度。
        scale: float，query 缩放系数 `1/sqrt(K)`。
        C: chunk 长度，编译期常量。
        BK, BV: K/V 维 block 大小。
        dv_ptr: [B, H, T, V]，输出 dv_local。
    """
    pid = tl.program_id(0).to(tl.int64)
    NV = tl.cdiv(V, BV)
    i_v = pid % NV
    i_t = (pid // NV) % (T // C)
    tmp = pid // (NV * (T // C))
    h = tmp % H
    b = tmp // H

    o_v = i_v * BV + tl.arange(0, BV)
    o_c = tl.arange(0, C)
    m_v = o_v < V
    m_c = o_c < C

    base_bh = (b * H + h).to(tl.int64)
    base_qk = base_bh * T + i_t * C
    base_v = base_bh * T + i_t * C
    base_gb = base_bh * T + i_t * C

    b_dv = tl.zeros([C, BV], dtype=tl.float32)
    b_A = tl.zeros([C, C], dtype=tl.float32)

    # b_A = k @ q^T，沿 K 维累加
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        p_k = k_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
        p_q = q_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)

        b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
        b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)

        # [C, K] @ [K, C] -> [C, C]
        b_A += tl.dot(b_k, tl.trans(b_q), allow_tf32=False)

    # 加载 g
    p_g = g_ptr + base_gb + o_c
    b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)

    # 对 A 项应用 causal mask（包含对角线）与 transpose decay：exp(g_col - g_row)
    scale_f32 = tl.cast(scale, tl.float32)
    mask_upper = o_c[:, None] <= o_c[None, :]
    diff = b_g[None, :] - b_g[:, None]
    # 只在 mask 区域内计算 exp，避免上三角 exp(大正数) 溢出成 inf 再被 where 成 NaN
    decay = tl.exp(tl.where(mask_upper, diff, 0.0))
    b_A = tl.where(
        mask_upper & m_c[:, None] & m_c[None, :], b_A * decay * scale_f32, 0.0
    )

    # b_dv = b_A @ do
    p_do = do_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
    b_do = tl.load(p_do, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
    b_dv += tl.dot(b_A.to(b_do.dtype), b_do, allow_tf32=False)

    # 保存 dv_local
    p_dv = dv_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
    tl.store(p_dv, b_dv.to(dv_ptr.dtype.element_ty), mask=m_c[:, None] & m_v[None, :])


def gdn_chunk_bwd_dv_local(q, k, g, do, scale, chunk_size=64):
    """局部 dv backward 封装。

    Args:
        q: [B, H, T, K]，已归一化的 query。
        k: [B, H, T, K]，已归一化的 key。
        g: [B, H, T]，cumsum 后的 log-space decay。
        do: [B, H, T, V]，输出梯度。
        scale: float，`1/sqrt(K)`。
        chunk_size: int。

    Returns:
        dv_local: [B, H, T, V]。只包含 chunk 内 causal 项，后续需要与 dh 项累加。
    """
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size

    dv = torch.empty_like(do)

    def grid(meta):
        return (B * H * (T // C) * triton.cdiv(V, meta["BV"]),)

    _gdn_chunk_bwd_dv_local_kernel[grid](
        q,
        k,
        g,
        do,
        B,
        H,
        T,
        K,
        V,
        scale,
        dv,
        C=C,
    )
    return dv

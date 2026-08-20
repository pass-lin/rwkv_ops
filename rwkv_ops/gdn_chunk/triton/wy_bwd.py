"""Gated DeltaNet chunkwise WY 表示反向 Triton kernel。

从 `dw, du` 反推 `dk, dv, db, dg`。

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
        for num_stages in [2, 3]
    ],
    key=["K", "V", "C"],
)
@triton.jit
def _gdn_chunk_prepare_wy_repr_bwd_kernel(
    k_ptr,
    v_ptr,
    beta_ptr,
    g_ptr,
    A_ptr,
    dw_ptr,
    du_ptr,
    dk_ptr,
    dv_ptr,
    db_ptr,
    dg_ptr,
    B,
    H,
    T,
    K,
    V,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """WY 表示反向 kernel。

    每个 program 处理一个 (batch, head, chunk)。

    Args:
        k_ptr: [B, H, T, K]，已归一化的 key。
        v_ptr: [B, H, T, V]。
        beta_ptr: [B, H, T]，post-sigmoid beta。
        g_ptr: [B, H, T]，cumsum 后的 log-space decay。
        A_ptr: [B, H, T//C, C, C]，`(I + L)^{-1}`。
        dw_ptr, du_ptr: [B, H, T, K], [B, H, T, V]。
        dk_ptr, dv_ptr, db_ptr, dg_ptr: 输出。
        B, H, T, K, V: 维度。
        C: chunk 长度，编译期常量。
        BK, BV: K/V 维 block 大小。
    """
    pid = tl.program_id(0).to(tl.int64)
    N = T // C
    n = pid % N
    h = (pid // N) % H
    b = pid // (N * H)

    base_bh = (b * H + h).to(tl.int64)
    base_k = base_bh * T + n * C
    base_v = base_bh * T + n * C
    base_gb = base_bh * T + n * C
    base_A = base_bh * N + n

    o_c = tl.arange(0, C)
    m_c = o_c < C

    # 加载 beta, A
    p_beta = beta_ptr + base_gb + o_c
    p_db = db_ptr + base_gb + o_c
    b_beta = tl.load(p_beta, mask=m_c, other=0.0).to(tl.float32)
    b_db = tl.zeros([C], dtype=tl.float32)

    p_A = A_ptr + base_A * C * C + o_c[:, None] * C + o_c[None, :]
    b_A = tl.load(p_A, mask=m_c[:, None] & m_c[None, :], other=0.0).to(tl.float32)
    b_dA = tl.zeros([C, C], dtype=tl.float32)

    p_g = g_ptr + base_gb + o_c
    b_g = tl.load(p_g, mask=m_c, other=0.0).to(tl.float32)
    b_g_exp = tl.exp(b_g)
    b_dg = tl.zeros([C], dtype=tl.float32)

    # K 维：处理 dw -> dk
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_c[:, None] & (o_k[None, :] < K)

        p_k = k_ptr + (base_k * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
        p_dk = dk_ptr + (base_k * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
        p_dw = dw_ptr + (base_k * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)

        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
        b_kbg = b_k * (b_beta * b_g_exp)[:, None]
        b_dw = tl.load(p_dw, mask=m_k, other=0.0).to(tl.float32)

        b_dA += tl.dot(b_dw, tl.trans(b_kbg).to(b_dw.dtype), allow_tf32=False)
        b_dkbg = tl.dot(b_A, b_dw, allow_tf32=False)

        b_dk = b_dkbg * (b_g_exp * b_beta)[:, None]
        b_db += tl.sum(b_dkbg * b_k * b_g_exp[:, None], axis=1)
        b_dg += tl.sum(b_dkbg * b_kbg, axis=1)

        tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_k)

    # V 维：处理 du -> dv
    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = m_c[:, None] & (o_v[None, :] < V)

        p_v = v_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
        p_dv = dv_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
        p_du = du_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)

        b_v = tl.load(p_v, mask=m_v, other=0.0).to(tl.float32)
        b_vb = (b_v * b_beta[:, None]).to(b_v.dtype)
        b_du = tl.load(p_du, mask=m_v, other=0.0).to(tl.float32)

        b_dA += tl.dot(b_du, tl.trans(b_vb), allow_tf32=False)
        b_dvb = tl.dot(b_A, b_du, allow_tf32=False)
        b_dv = b_dvb * b_beta[:, None]
        b_db += tl.sum(b_dvb * b_v, axis=1)

        tl.store(p_dv, b_dv.to(dv_ptr.dtype.element_ty), mask=m_v)

    # dA = A @ dA @ A，注意符号与 decay
    mask_lower = o_c[:, None] > o_c[None, :]
    b_dA = tl.where(mask_lower & m_c[:, None] & m_c[None, :], b_dA, 0.0)
    b_dA = tl.dot(b_dA.to(b_A.dtype), b_A, allow_tf32=False)
    b_dA = tl.dot(b_A, b_dA.to(b_A.dtype), allow_tf32=False)
    # 只在严格下三角计算 exp，避免上三角 exp(大正数) 溢出成 NaN
    diff = b_g[:, None] - b_g[None, :]
    decay = tl.exp(tl.where(mask_lower, diff, 0.0))
    b_dA *= decay
    b_dA = tl.where(mask_lower & m_c[:, None] & m_c[None, :], -b_dA, 0.0).to(
        k_ptr.dtype.element_ty
    )

    # 用 dA 重新计算 dk 和 db
    tl.debug_barrier()
    b_A = tl.zeros([C, C], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = m_c[:, None] & (o_k[None, :] < K)

        p_k = k_ptr + (base_k * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
        p_dk = dk_ptr + (base_k * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)

        b_k = tl.load(p_k, mask=m_k, other=0.0).to(tl.float32)
        b_kt = tl.trans(b_k)
        b_kb = b_k * b_beta[:, None]

        b_A += tl.dot(b_k, b_kt, allow_tf32=False)
        b_dkb = tl.dot(b_dA, b_k, allow_tf32=False)
        b_db += tl.sum(b_dkb * b_k, axis=1)
        b_dk = b_dkb * b_beta[:, None] + tl.trans(
            tl.dot(tl.trans(b_kb).to(b_dA.dtype), b_dA, allow_tf32=False)
        )
        b_dk += tl.load(p_dk, mask=m_k, other=0.0).to(tl.float32)

        tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_k)

    tl.store(p_db, b_db.to(db_ptr.dtype.element_ty), mask=m_c)

    # dg 贡献
    b_A *= b_beta[:, None]
    b_AdA = b_dA * b_A
    p_dg = dg_ptr + base_gb + o_c
    b_dg += tl.sum(b_AdA, axis=1) - tl.sum(b_AdA, axis=0)
    tl.store(p_dg, b_dg.to(dg_ptr.dtype.element_ty), mask=m_c)


def gdn_chunk_prepare_wy_repr_bwd(k, v, beta, g, A, dw, du, chunk_size=64):
    """WY 表示反向封装。

    Args:
        k: [B, H, T, K]。
        v: [B, H, T, V]。
        beta: [B, H, T]。
        g: [B, H, T]，cumsum 后的 log-space decay。
        A: [B, H, T//C, C, C]。
        dw: [B, H, T, K]。
        du: [B, H, T, V]。
        chunk_size: int。

    Returns:
        dk: [B, H, T, K]。
        dv: [B, H, T, V]。
        db: [B, H, T]。
        dg: [B, H, T]。
    """
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk_size

    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    db = torch.empty(B, H, T, dtype=torch.float32, device=k.device)
    dg = torch.empty(B, H, T, dtype=torch.float32, device=k.device)

    grid = (B * H * (T // C),)
    _gdn_chunk_prepare_wy_repr_bwd_kernel[grid](
        k,
        v,
        beta,
        g,
        A,
        dw,
        du,
        dk,
        dv,
        db,
        dg,
        B,
        H,
        T,
        K,
        V,
        C=C,
    )
    return dk, dv, db, dg

"""DeltaNet chunkwise dq/dk/dw Triton kernel。

从 `do, dv, dh` 计算 `dq, dk, dw`。

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
def _delta_net_chunk_bwd_dqk_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    w_ptr,
    h_ptr,
    dh_ptr,
    do_ptr,
    dv_ptr,
    B,
    H,
    T,
    K,
    V,
    scale,
    dq_ptr,
    dk_ptr,
    dw_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """dq/dk/dw backward kernel。

    每个 program 处理一个 (batch, head, chunk, K-block)，沿 V 维累加。

    Args:
        q_ptr, k_ptr, w_ptr: [B, H, T, K]。
        v_ptr: [B, H, T, V]，即 v_new。
        h_ptr: [B, H, T//C, K, V]，chunk 级状态。
        dh_ptr: [B, H, T//C, K, V]，chunk 级状态梯度。
        do_ptr: [B, H, T, V]。
        dv_ptr: [B, H, T, V]。
        B, H, T, K, V: 维度。
        scale: float，`1/sqrt(K)`。
        C: chunk 长度，编译期常量。
        BK, BV: K/V 维 block 大小。
        dq_ptr, dk_ptr, dw_ptr: 输出。
    """
    pid = tl.program_id(0).to(tl.int64)
    NK = tl.cdiv(K, BK)
    i_k = pid % NK
    i_t = (pid // NK) % (T // C)
    tmp = pid // (NK * (T // C))
    h = tmp % H
    b = tmp // H

    base_bh = (b * H + h).to(tl.int64)
    base_qk = base_bh * T + i_t * C
    base_v = base_bh * T + i_t * C
    base_dh = base_bh * (T // C) + i_t

    o_k = i_k * BK + tl.arange(0, BK)
    o_c = tl.arange(0, C)
    m_k = o_k < K
    m_c = o_c < C

    b_dq = tl.zeros([C, BK], dtype=tl.float32)
    b_dk = tl.zeros([C, BK], dtype=tl.float32)
    b_ds = tl.zeros([C, C], dtype=tl.float32)
    b_dw = tl.zeros([C, BK], dtype=tl.float32)

    for i_v in range(tl.cdiv(V, BV)):
        o_v = i_v * BV + tl.arange(0, BV)
        m_v = o_v < V

        # 加载 v, do
        p_v = v_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
        p_do = do_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
        b_v = tl.load(p_v, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_do = tl.load(p_do, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        # 加载 h, dh [K, V]
        p_h = h_ptr + (base_dh * K * V + o_k[:, None] * V + o_v[None, :]).to(tl.int64)
        p_dh = dh_ptr + (base_dh * K * V + o_k[:, None] * V + o_v[None, :]).to(tl.int64)
        b_h = tl.load(p_h, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_dh = tl.load(p_dh, mask=m_k[:, None] & m_v[None, :], other=0.0).to(tl.float32)

        # b_ds += do @ v^T
        b_ds += tl.dot(b_do, tl.trans(b_v), allow_tf32=False)

        # b_dq += do @ h^T
        b_dq += tl.dot(b_do, tl.trans(b_h).to(b_do.dtype), allow_tf32=False)

        # b_dk += v @ dh^T
        b_dk += tl.dot(b_v, tl.trans(b_dh).to(b_v.dtype), allow_tf32=False)

        # b_dw -= dv @ h^T
        p_dv = dv_ptr + (base_v * V + o_c[:, None] * V + o_v[None, :]).to(tl.int64)
        b_dv = tl.load(p_dv, mask=m_c[:, None] & m_v[None, :], other=0.0).to(tl.float32)
        b_dw -= tl.dot(b_dv, tl.trans(b_h).to(b_dv.dtype), allow_tf32=False)

    # 加载 q, k
    p_q = q_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
    p_k = k_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
    b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)
    b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(tl.float32)

    scale_f32 = tl.cast(scale, tl.float32)

    # 对 b_dq 应用 inter 项的 scale
    b_dq *= scale_f32

    # 对 b_ds 应用 causal mask
    mask_upper = o_c[:, None] >= o_c[None, :]
    b_ds = tl.where(mask_upper & m_c[:, None] & m_c[None, :], b_ds * scale_f32, 0.0)

    # b_dq += b_ds @ k
    # b_dk += b_ds^T @ q
    b_ds = b_ds.to(b_k.dtype)
    b_dq += tl.dot(b_ds, b_k, allow_tf32=False)
    b_dk += tl.dot(tl.trans(b_ds), b_q, allow_tf32=False)

    # 保存
    p_dq = dq_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
    p_dk = dk_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)
    p_dw = dw_ptr + (base_qk * K + o_c[:, None] * K + o_k[None, :]).to(tl.int64)

    tl.store(p_dq, b_dq.to(dq_ptr.dtype.element_ty), mask=m_c[:, None] & m_k[None, :])
    tl.store(p_dk, b_dk.to(dk_ptr.dtype.element_ty), mask=m_c[:, None] & m_k[None, :])
    tl.store(p_dw, b_dw.to(dw_ptr.dtype.element_ty), mask=m_c[:, None] & m_k[None, :])


def delta_net_chunk_bwd_dqk(q, k, v_new, w, h, dh, do, dv, scale, chunk_size=64):
    """dq/dk/dw backward 封装。

    Args:
        q, k, w: [B, H, T, K]。
        v_new: [B, H, T, V]。
        h, dh: [B, H, T//C, K, V]。
        do, dv: [B, H, T, V]。
        scale: float。
        chunk_size: int。

    Returns:
        dq, dk, dw: [B, H, T, K]。
    """
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size

    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dw = torch.empty_like(w)

    def grid(meta):
        return (B * H * (T // C) * triton.cdiv(K, meta["BK"]),)

    _delta_net_chunk_bwd_dqk_kernel[grid](
        q,
        k,
        v_new,
        w,
        h,
        dh,
        do,
        dv,
        B,
        H,
        T,
        K,
        V,
        scale,
        dq,
        dk,
        dw,
        C=C,
    )
    return dq, dk, dw

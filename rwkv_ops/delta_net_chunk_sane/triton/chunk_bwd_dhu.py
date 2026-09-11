"""DeltaNet chunkwise SANE 状态反向扫描 Triton kernel。

对每个 chunk 边界反向传播 SANE 的 tanh 导数，并输出 dtau。

关键优化：不额外保存 SANE 前状态 s_raw，而是在反向时从 h[i]（SANE 后进入状态）
和 v_new 重算 s_raw = h[i] + k^T @ v_new。
"""

import torch
import triton
import triton.language as tl

from ...triton_utils import _sane_backward_factor, _sane_dtau_factor


def _bwd_autotune_pre_hook(nargs, reset_only=False):
    """反向 kernel 在 autotune benchmark 前清零 dtau，避免 atomic_add 累加脏值。"""
    for name in ("dtau_ptr", "dtau"):
        buf = nargs.get(name)
        if buf is None:
            continue
        if hasattr(buf, "zero_"):
            buf.zero_()
        else:
            try:
                import jax.numpy as jnp

                nargs[name] = jnp.zeros_like(buf)
            except Exception:
                pass


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2]
    ],
    key=["K", "V", "C"],
    pre_hook=_bwd_autotune_pre_hook,
)
@triton.jit
def _delta_net_chunk_bwd_dhu_sane_kernel(
    q_ptr,
    k_ptr,
    w_ptr,
    h_ptr,
    v_new_ptr,
    tau_ptr,
    mask_ptr,
    do_ptr,
    dv_local_ptr,
    dht_ptr,
    B,
    H,
    T,
    K,
    V,
    scale,
    dh_ptr,
    dh0_ptr,
    dv_out_ptr,
    dtau_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_MASK: tl.constexpr,
):
    """状态反向扫描 SANE kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维反向扫描。

    Args:
        q_ptr, k_ptr, w_ptr: [B, H, T, K]。
        h_ptr: [B, H, T//C, K, V]，SANE 后进入状态。
        v_new_ptr: [B, H, T, V]。
        tau_ptr: [B, H, T//C]，SANE 阈值。
        mask_ptr: [B, T//C]，per-chunk mask；USE_MASK=False 时不读。
        do_ptr: [B, H, T, V]。
        dv_local_ptr: [B, H, T, V]，来自 chunk_bwd_dv_local 的局部 dv。
        dht_ptr: [B, H, K, V]，最终状态梯度；未提供时为全零。
        B, H, T, K, V: 维度。
        scale: float，`1/sqrt(K)`。
        C: chunk 长度，编译期常量。
        BV: V 维 block 大小。
        dh_ptr: [B, H, T//C, K, V]，输出 chunk 级状态梯度（对 SANE 后状态）。
        dh0_ptr: [B, H, K, V]，输出初始状态梯度。
        dv_out_ptr: [B, H, T, V]，输出累加后的 dv。
        dtau_ptr: [B, H, T//C]，输出 tau 梯度。
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
    base_qk = base_bh * T
    base_v = base_bh * T
    base_dh = base_bh * N
    base_h = base_bh * N
    base_tau = base_bh * N
    base_mask = b * N

    # 初始化 dh；未提供 dht 时传零张量，结果等价于零初始化。
    p_dht = dht_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    b_dh = tl.load(p_dht, mask=m_h, other=0.0).to(tl.float32)

    scale_f32 = tl.cast(scale, tl.float32)

    for i_t in range(N - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)

        # 保存 dh_i（这是对 SANE 后状态 h[i+1] 的梯度）
        p_dh = dh_ptr + (base_dh + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_dh.to(dh_ptr.dtype.element_ty), mask=m_h)

        # 重算 SANE 前状态 s_raw = h[i] + k^T @ v_new
        p_h = h_ptr + (base_h + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_h, other=0.0).to(tl.float32)

        o_c = tl.arange(0, C)
        m_c = o_c < C

        base_chunk = base_qk + i_t_int64 * C
        base_chunk_v = base_v + i_t_int64 * C

        b_s_raw = b_h

        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_k = k_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_v_new = v_new_ptr + (
                base_chunk_v * V + o_c[:, None] * V + o_v[None, :]
            ).to(tl.int64)
            b_v_new = tl.load(p_v_new, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
                tl.float32
            )

            b_s_raw += tl.dot(
                tl.trans(b_k).to(b_v_new.dtype), b_v_new, allow_tf32=False
            )

        # 读取 tau
        p_tau = tau_ptr + base_tau + i_t_int64
        b_tau = tl.load(p_tau).to(tl.float32)
        b_tau = tl.maximum(b_tau, 1e-6)

        # SANE 反向：d(s_raw) = d(h[i+1]) * d_sane_factor
        b_sech2 = _sane_backward_factor(b_s_raw, b_tau)

        if USE_MASK:
            p_mask = mask_ptr + base_mask + i_t_int64
            b_mask = tl.load(p_mask).to(tl.float32)
            b_mask = tl.where(b_mask > 0.0, 1.0, 0.0)
            b_dh_raw = b_dh * ((1.0 - b_mask) + b_mask * b_sech2)
        else:
            b_mask = 1.0
            b_dh_raw = b_dh * b_sech2

        # dtau = sum(d(h[i+1]) * mask * (tanh(u) - u * sech2))
        b_dtau_local = _sane_dtau_factor(b_dh * b_mask, b_s_raw, b_tau)
        b_dtau = tl.sum(b_dtau_local)
        p_dtau = dtau_ptr + base_tau + i_t_int64
        tl.atomic_add(p_dtau, b_dtau.to(dtau_ptr.dtype.element_ty))

        # 以下继续使用 b_dh_raw（对 s_raw 的梯度）反向传播过 chunk 更新
        b_dh = b_dh_raw

        # dv = k @ dh，沿 K 维累加
        b_dv = tl.zeros([C, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_k = k_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_dh_block = (
                dh_ptr
                + (base_dh + i_t_int64) * K * V
                + o_k_chunk[:, None] * V
                + o_v[None, :]
            )
            b_dh_block = tl.load(
                p_dh_block, mask=m_k[:, None] & m_v[None, :], other=0.0
            ).to(tl.float32)

            b_dv += tl.dot(b_k, b_dh_block.to(b_k.dtype), allow_tf32=False)

        # 加上 dv_local 并保存最终 dv
        p_dv_local = dv_local_ptr + (
            base_chunk_v * V + o_c[:, None] * V + o_v[None, :]
        ).to(tl.int64)
        b_dv += tl.load(p_dv_local, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
            tl.float32
        )
        p_dv_out = dv_out_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
            tl.int64
        )
        tl.store(
            p_dv_out,
            b_dv.to(dv_out_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新 dh：dh = dh + q @ do * scale - w @ dv
        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_q = q_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_do = do_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
                tl.int64
            )
            b_do = tl.load(p_do, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]
            b_dh += (
                tl.dot(tl.trans(b_q).to(b_do.dtype), b_do, allow_tf32=False) * scale_f32
            )

            p_w = w_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_w = tl.load(p_w, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]
            b_dh -= tl.dot(tl.trans(b_w).to(b_dv.dtype), b_dv, allow_tf32=False)

    # 保存 dh0
    p_dh0 = dh0_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_h)


@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for BV in [32, 64]
        for num_warps in [2, 4]
        for num_stages in [2]
    ],
    key=["K", "V", "C"],
    pre_hook=_bwd_autotune_pre_hook,
)
@triton.jit
def _delta_net_chunk_bwd_dhu_sane_kernel_no_mask(
    q_ptr,
    k_ptr,
    w_ptr,
    h_ptr,
    v_new_ptr,
    tau_ptr,
    do_ptr,
    dv_local_ptr,
    dht_ptr,
    B,
    H,
    T,
    K,
    V,
    scale,
    dh_ptr,
    dh0_ptr,
    dv_out_ptr,
    dtau_ptr,
    C: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
):
    """状态反向扫描 SANE 无 mask kernel。

    每个 program 处理一个 (batch, head, V-block)，沿 chunk 维反向扫描。

    Args:
        q_ptr, k_ptr, w_ptr: [B, H, T, K]。
        h_ptr: [B, H, T//C, K, V]，SANE 后进入状态。
        v_new_ptr: [B, H, T, V]。
        tau_ptr: [B, H, T//C]，SANE 阈值。
        do_ptr: [B, H, T, V]。
        dv_local_ptr: [B, H, T, V]，来自 chunk_bwd_dv_local 的局部 dv。
        dht_ptr: [B, H, K, V]，最终状态梯度；未提供时为全零。
        B, H, T, K, V: 维度。
        scale: float，`1/sqrt(K)`。
        C: chunk 长度，编译期常量。
        BV: V 维 block 大小。
        dh_ptr: [B, H, T//C, K, V]，输出 chunk 级状态梯度（对 SANE 后状态）。
        dh0_ptr: [B, H, K, V]，输出初始状态梯度。
        dv_out_ptr: [B, H, T, V]，输出累加后的 dv。
        dtau_ptr: [B, H, T//C]，输出 tau 梯度。
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
    base_qk = base_bh * T
    base_v = base_bh * T
    base_dh = base_bh * N
    base_h = base_bh * N
    base_tau = base_bh * N

    # 初始化 dh；未提供 dht 时传零张量，结果等价于零初始化。
    p_dht = dht_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    b_dh = tl.load(p_dht, mask=m_h, other=0.0).to(tl.float32)

    scale_f32 = tl.cast(scale, tl.float32)

    for i_t in range(N - 1, -1, -1):
        i_t_int64 = i_t.to(tl.int64)

        # 保存 dh_i（这是对 SANE 后状态 h[i+1] 的梯度）
        p_dh = dh_ptr + (base_dh + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
        tl.store(p_dh, b_dh.to(dh_ptr.dtype.element_ty), mask=m_h)

        # 重算 SANE 前状态 s_raw = h[i] + k^T @ v_new
        p_h = h_ptr + (base_h + i_t_int64) * K * V + o_k[:, None] * V + o_v[None, :]
        b_h = tl.load(p_h, mask=m_h, other=0.0).to(tl.float32)

        o_c = tl.arange(0, C)
        m_c = o_c < C

        base_chunk = base_qk + i_t_int64 * C
        base_chunk_v = base_v + i_t_int64 * C

        b_s_raw = b_h

        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_k = k_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_v_new = v_new_ptr + (
                base_chunk_v * V + o_c[:, None] * V + o_v[None, :]
            ).to(tl.int64)
            b_v_new = tl.load(p_v_new, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
                tl.float32
            )

            b_s_raw += tl.dot(
                tl.trans(b_k).to(b_v_new.dtype), b_v_new, allow_tf32=False
            )

        # 读取 tau
        p_tau = tau_ptr + base_tau + i_t_int64
        b_tau = tl.load(p_tau).to(tl.float32)
        b_tau = tl.maximum(b_tau, 1e-6)

        # SANE 反向：d(s_raw) = d(h[i+1]) * d_sane_factor
        b_sech2 = _sane_backward_factor(b_s_raw, b_tau)

        b_dh_raw = b_dh * b_sech2

        # dtau = sum(d(h[i+1]) * mask * (tanh(u) - u * sech2))
        b_dtau_local = _sane_dtau_factor(b_dh, b_s_raw, b_tau)
        b_dtau = tl.sum(b_dtau_local)
        p_dtau = dtau_ptr + base_tau + i_t_int64
        tl.atomic_add(p_dtau, b_dtau.to(dtau_ptr.dtype.element_ty))

        # 以下继续使用 b_dh_raw（对 s_raw 的梯度）反向传播过 chunk 更新
        b_dh = b_dh_raw

        # dv = k @ dh，沿 K 维累加
        b_dv = tl.zeros([C, BV], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_k = k_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_k = tl.load(p_k, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_dh_block = (
                dh_ptr
                + (base_dh + i_t_int64) * K * V
                + o_k_chunk[:, None] * V
                + o_v[None, :]
            )
            b_dh_block = tl.load(
                p_dh_block, mask=m_k[:, None] & m_v[None, :], other=0.0
            ).to(tl.float32)

            b_dv += tl.dot(b_k, b_dh_block.to(b_k.dtype), allow_tf32=False)

        # 加上 dv_local 并保存最终 dv
        p_dv_local = dv_local_ptr + (
            base_chunk_v * V + o_c[:, None] * V + o_v[None, :]
        ).to(tl.int64)
        b_dv += tl.load(p_dv_local, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
            tl.float32
        )
        p_dv_out = dv_out_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
            tl.int64
        )
        tl.store(
            p_dv_out,
            b_dv.to(dv_out_ptr.dtype.element_ty),
            mask=m_c[:, None] & m_v[None, :],
        )

        # 更新 dh：dh = dh + q @ do * scale - w @ dv
        for i_k in range(tl.cdiv(K, BK)):
            o_k_chunk = i_k * BK + tl.arange(0, BK)
            m_k = o_k_chunk < K

            p_q = q_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_q = tl.load(p_q, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            p_do = do_ptr + (base_chunk_v * V + o_c[:, None] * V + o_v[None, :]).to(
                tl.int64
            )
            b_do = tl.load(p_do, mask=m_c[:, None] & m_v[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]
            b_dh += (
                tl.dot(tl.trans(b_q).to(b_do.dtype), b_do, allow_tf32=False) * scale_f32
            )

            p_w = w_ptr + (base_chunk * K + o_c[:, None] * K + o_k_chunk[None, :]).to(
                tl.int64
            )
            b_w = tl.load(p_w, mask=m_c[:, None] & m_k[None, :], other=0.0).to(
                tl.float32
            )

            # [C, K]^T @ [C, V] -> [K, V]
            b_dh -= tl.dot(tl.trans(b_w).to(b_dv.dtype), b_dv, allow_tf32=False)

    # 保存 dh0
    p_dh0 = dh0_ptr + base_bh * K * V + o_k[:, None] * V + o_v[None, :]
    tl.store(p_dh0, b_dh.to(dh0_ptr.dtype.element_ty), mask=m_h)


def delta_net_chunk_bwd_dhu_sane_no_mask(
    q,
    k,
    w,
    h,
    v_new,
    tau,
    do,
    dv_local,
    dht=None,
    scale=1.0,
    chunk_size=64,
):
    """状态反向扫描 SANE 封装。

    Args:
        q: [B, H, T, K]。
        k: [B, H, T, K]。
        w: [B, H, T, K]。
        h: [B, H, T//C, K, V]，SANE 后进入状态。
        v_new: [B, H, T, V]。
        tau: [B, H, T//C]，SANE 阈值。
        do: [B, H, T, V]。
        dv_local: [B, H, T, V]。
        dht: [B, H, K, V]，可选。
        scale: float。
        chunk_size: int。

    Returns:
        dh: [B, H, T//C, K, V]。
        dh0: [B, H, K, V]。
        dv: [B, H, T, V]。
        dtau: [B, H, T//C]。
    """
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    N = T // C

    dh = torch.empty(B, H, N, K, V, dtype=torch.float32, device=q.device)
    dv = torch.empty_like(do)
    dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
    # dtau 由 kernel 内 atomic_add 累加，必须零初始化。
    dtau = torch.zeros(B, H, N, dtype=torch.float32, device=q.device)

    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (B * H * triton.cdiv(V, meta["BV"]),)

    if dht is None:
        dht = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    _delta_net_chunk_bwd_dhu_sane_kernel[grid](
        q,
        k,
        w,
        h,
        v_new,
        tau,
        do,
        dv_local,
        dht,
        B,
        H,
        T,
        K,
        V,
        scale,
        dh,
        dh0,
        dv,
        dtau,
        C=C,
        BK=BK,
    )
    return dh, dh0, dv, dtau

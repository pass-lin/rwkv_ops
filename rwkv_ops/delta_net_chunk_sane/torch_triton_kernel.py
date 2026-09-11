"""PyTorch 版 DeltaNet chunkwise SANE Triton kernel 封装。"""

import warnings

import torch

from .triton import (
    delta_net_chunk_bwd_dhu,
    delta_net_chunk_bwd_dqk,
    delta_net_chunk_bwd_dv_local,
    delta_net_chunk_fwd_h,
    delta_net_chunk_fwd_intra,
    delta_net_chunk_fwd_o,
    delta_net_chunk_l2norm_bwd,
    delta_net_chunk_l2norm_fwd,
    delta_net_chunk_prepare_wy_repr_bwd,
    delta_net_chunk_recompute_w_u,
)
from .triton.chunk_bwd_dhu import _delta_net_chunk_bwd_dhu_sane_kernel
from .triton.chunk_h import _delta_net_chunk_fwd_h_sane_kernel
from ..delta_net_chunk.triton.chunk_bwd_dqk import _delta_net_chunk_bwd_dqk_kernel
from ..delta_net_chunk.triton.chunk_bwd_dv import _delta_net_chunk_bwd_dv_local_kernel
from ..delta_net_chunk.triton.chunk_o import _delta_net_chunk_fwd_o_kernel
from ..delta_net_chunk.triton.intra import _delta_net_chunk_fwd_intra_kernel
from ..delta_net_chunk.triton.l2norm import (
    _delta_net_chunk_l2norm_bwd_kernel,
    _delta_net_chunk_l2norm_fwd_kernel,
)
from ..delta_net_chunk.triton.wy import _delta_net_chunk_recompute_w_u_fwd_kernel
from ..delta_net_chunk.triton.wy_bwd import _delta_net_chunk_prepare_wy_repr_bwd_kernel


def _clear_delta_net_chunk_sane_autotune_cache():
    """清空 delta_net_chunk_sane 所有 Triton kernel 的 autotune cache。"""
    for kernel in (
        _delta_net_chunk_fwd_h_sane_kernel,
        _delta_net_chunk_fwd_o_kernel,
        _delta_net_chunk_fwd_intra_kernel,
        _delta_net_chunk_recompute_w_u_fwd_kernel,
        _delta_net_chunk_l2norm_fwd_kernel,
        _delta_net_chunk_l2norm_bwd_kernel,
        _delta_net_chunk_bwd_dhu_sane_kernel,
        _delta_net_chunk_bwd_dqk_kernel,
        _delta_net_chunk_bwd_dv_local_kernel,
        _delta_net_chunk_prepare_wy_repr_bwd_kernel,
    ):
        if hasattr(kernel, "cache"):
            kernel.cache.clear()


def _normalize_inputs(q, k, v, beta):
    """把输入从 [B, T, H, *] 转成 [B, H, T, *] 并保证连续。"""
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    return q, k, v, beta


def _transpose_tau(tau):
    """tau 公共接口 [B, T//C, H] -> 内部 [B, H, T//C]。"""
    return tau.transpose(1, 2).contiguous()


class DeltaNetChunkSaneTritonFunction(torch.autograd.Function):
    """DeltaNet chunkwise SANE 训练前向 Triton 封装。"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        beta,
        tau,
        mask,
        initial_state,
        output_final_state,
        chunk_size,
        use_mask,
    ):
        _clear_delta_net_chunk_sane_autotune_cache()
        q, k, v, beta = _normalize_inputs(q, k, v, beta)
        tau = _transpose_tau(tau)
        if use_mask:
            mask = mask.contiguous()

        B, H, T, K = q.shape

        if T % chunk_size != 0:
            raise ValueError(
                f"Triton SANE kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
            )
        if chunk_size < 16:
            raise ValueError(
                f"Triton kernel requires chunk_size >= 16, got {chunk_size}"
            )

        C = T // chunk_size
        if tau.shape != (B, H, C):
            raise ValueError(
                f"tau shape {tau.shape} does not match expected (B={B}, H={H}, T//chunk_size={C})"
            )
        if use_mask and mask.shape != (B, C):
            raise ValueError(
                f"mask shape {mask.shape} must match (B, T//chunk_size) = ({B}, {C})"
            )

        # 保存原始 q/k 供 L2 norm 反向使用
        q_orig = q.clone()
        k_orig = k.clone()

        # L2 归一化
        q, inv_norm_q = delta_net_chunk_l2norm_fwd(q)
        k, inv_norm_k = delta_net_chunk_l2norm_fwd(k)

        # intra-chunk: A = (I + L)^{-1}
        A = delta_net_chunk_fwd_intra(k, beta, chunk_size=chunk_size)

        # WY 表示: w, u
        w, u = delta_net_chunk_recompute_w_u(k, v, beta, A, chunk_size=chunk_size)

        # chunk 间状态递推（SANE）
        h, v_new, final_state = delta_net_chunk_fwd_h(
            k,
            w,
            u,
            tau,
            mask if use_mask else None,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
            use_mask=use_mask,
        )

        # 最终输出
        o = delta_net_chunk_fwd_o(q, k, v_new, h, chunk_size=chunk_size)

        ctx.save_for_backward(
            q,
            k,
            q_orig,
            k_orig,
            v,
            beta,
            tau,
            mask,
            inv_norm_q,
            inv_norm_k,
            A,
            w,
            u,
            h,
            v_new,
            initial_state,
        )
        ctx.chunk_size = chunk_size
        ctx.output_final_state = output_final_state
        ctx.use_mask = use_mask

        o = o.transpose(1, 2)
        if output_final_state:
            return o, final_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        """反向传播：计算 dq, dk, dv, db, dtau, dh0。"""
        (
            q,
            k,
            q_orig,
            k_orig,
            v,
            beta,
            tau,
            mask,
            inv_norm_q,
            inv_norm_k,
            A,
            w,
            u,
            h,
            v_new,
            initial_state,
        ) = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        use_mask = ctx.use_mask
        B, H, T, K = q.shape
        scale = K**-0.5

        # do 从外部 layout [B, T, H, V] 转成内部 [B, H, T, V]
        do = do.transpose(1, 2).contiguous()

        # 1. 局部 dv（只含 chunk 内 causal 项）
        dv_local = delta_net_chunk_bwd_dv_local(q, k, do, scale, chunk_size=chunk_size)

        # 2. 状态反向扫描（SANE）
        dh, dh0, dv, dtau = delta_net_chunk_bwd_dhu(
            q,
            k,
            w,
            h,
            v_new,
            tau,
            mask if use_mask else None,
            do,
            dv_local,
            dht=dht,
            scale=scale,
            chunk_size=chunk_size,
            use_mask=use_mask,
        )

        # 3. dq / dk / dw
        dq, dk, dw = delta_net_chunk_bwd_dqk(
            q,
            k,
            v_new,
            w,
            h,
            dh,
            do,
            dv,
            scale,
            chunk_size=chunk_size,
        )

        # 4. WY 表示反向
        dk2, dv2, db = delta_net_chunk_prepare_wy_repr_bwd(
            k,
            v,
            beta,
            A,
            dw,
            dv,
            chunk_size=chunk_size,
        )
        dk += dk2
        dv += dv2

        # 5. L2 norm 反向（需要原始输入而非归一化结果）
        dq = delta_net_chunk_l2norm_bwd(q_orig, inv_norm_q, dq)
        dk = delta_net_chunk_l2norm_bwd(k_orig, inv_norm_k, dk)

        # 6. 转回外部 layout [B, T, H, *]
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
        db = db.transpose(1, 2)
        dtau = dtau.transpose(1, 2)

        # 7. 匹配 forward 输入的梯度位置
        return (
            dq,
            dk,
            dv,
            db,
            dtau,  # tau
            None,  # mask
            dh0,  # initial_state
            None,  # output_final_state
            None,  # chunk_size
            None,  # use_mask
        )


def delta_net_chunk_sane(
    q,
    k,
    v,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """DeltaNet chunkwise SANE Triton 实现（训练前向）。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        beta: [B, T, H]，写入强度门控，已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，SANE 阈值。
        mask: [B, T//chunk_size]，float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；
            output_final_state=False 时不返回；
            output_final_state=True 且 mask=None 时为 None 并发出 UserWarning。

    Raises:
        ValueError: T 不被 chunk_size 整除，chunk_size < 16，或 tau/mask 形状不匹配。
    """
    B, T, H, K = q.shape
    C = T // chunk_size

    if tau.shape != (B, C, H):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, T//chunk_size={C}, H={H})"
        )

    use_mask = output_final_state and mask is not None
    if use_mask:
        if mask.shape != (B, C):
            raise ValueError(
                f"mask shape {mask.shape} must match (B, T//chunk_size) = ({B}, {C})"
            )
        mask_tensor = mask
    else:
        mask_tensor = None

    dtype = v.dtype
    q = q.to(dtype)
    k = k.to(dtype)

    out, final_state = DeltaNetChunkSaneTritonFunction.apply(
        q,
        k,
        v,
        beta,
        tau.to(torch.float32),
        mask_tensor.to(torch.float32) if mask_tensor is not None else None,
        initial_state,
        output_final_state,
        chunk_size,
        use_mask,
    )

    out = out.to(dtype)
    if not output_final_state:
        return out, None

    if mask is None:
        warnings.warn(
            "[delta_net_chunk_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[delta_net_chunk_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return out, final_state

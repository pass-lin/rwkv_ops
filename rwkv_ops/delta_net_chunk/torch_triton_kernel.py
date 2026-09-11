"""PyTorch 版 DeltaNet chunkwise Triton kernel 封装。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

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
from .triton.chunk_bwd_dhu import _delta_net_chunk_bwd_dhu_kernel
from .triton.chunk_bwd_dqk import _delta_net_chunk_bwd_dqk_kernel
from .triton.chunk_bwd_dv import _delta_net_chunk_bwd_dv_local_kernel
from .triton.chunk_h import _delta_net_chunk_fwd_h_kernel
from .triton.chunk_o import _delta_net_chunk_fwd_o_kernel
from .triton.intra import _delta_net_chunk_fwd_intra_kernel
from .triton.l2norm import (
    _delta_net_chunk_l2norm_bwd_kernel,
    _delta_net_chunk_l2norm_fwd_kernel,
)
from .triton.wy import _delta_net_chunk_recompute_w_u_fwd_kernel
from .triton.wy_bwd import _delta_net_chunk_prepare_wy_repr_bwd_kernel


def _clear_delta_net_chunk_autotune_cache():
    """清空 delta_net_chunk 所有 Triton kernel 的 autotune cache。

    不同调用路径（例如 USE_INITIAL_STATE=True/False）共享同一个 kernel 对象，
    autotune 缓存可能把为一条路径选出的 config 复用到另一条路径，导致 bf16 下
    输出 NaN。在每次前向调用前清空缓存可作为临时 workaround。
    """
    for kernel in (
        _delta_net_chunk_fwd_h_kernel,
        _delta_net_chunk_fwd_o_kernel,
        _delta_net_chunk_fwd_intra_kernel,
        _delta_net_chunk_recompute_w_u_fwd_kernel,
        _delta_net_chunk_l2norm_fwd_kernel,
        _delta_net_chunk_l2norm_bwd_kernel,
        _delta_net_chunk_bwd_dhu_kernel,
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


class DeltaNetChunkTritonFunction(torch.autograd.Function):
    """DeltaNet chunkwise 训练前向 Triton 封装。"""

    @staticmethod
    def forward(ctx, q, k, v, beta, initial_state, output_final_state, chunk_size):
        _clear_delta_net_chunk_autotune_cache()
        q, k, v, beta = _normalize_inputs(q, k, v, beta)

        B, H, T, K = q.shape

        if T % chunk_size != 0:
            raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")
        if chunk_size < 16:
            raise ValueError(
                f"Triton kernel requires chunk_size >= 16, got {chunk_size}"
            )

        # 保存原始 q/k 供 L2 norm 反向使用。
        q_orig = q.clone()
        k_orig = k.clone()

        # L2 归一化（在 Triton 内完成，与 delta_net_recurrent 一致）
        q, inv_norm_q = delta_net_chunk_l2norm_fwd(q)
        k, inv_norm_k = delta_net_chunk_l2norm_fwd(k)

        # intra-chunk: A = (I + L)^{-1}
        A = delta_net_chunk_fwd_intra(k, beta, chunk_size=chunk_size)

        # WY 表示: w, u
        w, u = delta_net_chunk_recompute_w_u(k, v, beta, A, chunk_size=chunk_size)

        # chunk 间状态递推
        h, v_new, final_state = delta_net_chunk_fwd_h(
            k,
            w,
            u,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
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

        o = o.transpose(1, 2)
        if output_final_state:
            return o, final_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        """反向传播：计算 dq, dk, dv, db, dh0。"""
        (
            q,
            k,
            q_orig,
            k_orig,
            v,
            beta,
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
        B, H, T, K = q.shape
        scale = K**-0.5

        # do 从外部 layout [B, T, H, V] 转成内部 [B, H, T, V]
        do = do.transpose(1, 2).contiguous()

        # 局部 dv（只含 chunk 内 causal 项）
        dv_local = delta_net_chunk_bwd_dv_local(q, k, do, scale, chunk_size=chunk_size)

        # 状态反向扫描
        dh, dh0, dv = delta_net_chunk_bwd_dhu(
            q,
            k,
            w,
            do,
            dv_local,
            dht=dht,
            scale=scale,
            chunk_size=chunk_size,
        )

        # dq / dk / dw
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

        # WY 表示反向
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

        # L2 norm 反向（需要原始输入而非归一化结果）
        dq = delta_net_chunk_l2norm_bwd(q_orig, inv_norm_q, dq)
        dk = delta_net_chunk_l2norm_bwd(k_orig, inv_norm_k, dk)

        # 转回外部 layout [B, T, H, *]
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
        db = db.transpose(1, 2)

        # 匹配 forward 输入的梯度位置
        return (
            dq,
            dk,
            dv,
            db,
            dh0,  # initial_state
            None,  # output_final_state
            None,  # chunk_size
        )


def delta_net_chunk(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """DeltaNet chunkwise Triton 实现（训练前向）。

    接口与 `rwkv_ops.delta_net_chunk.native_keras_op.delta_net_chunk` 一致。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        beta: [B, T, H]，写入强度门控，已在外部过 sigmoid。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 chunk_size < 16。
    """
    return DeltaNetChunkTritonFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        chunk_size,
    )

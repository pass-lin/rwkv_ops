"""PyTorch 版 Gated DeltaNet chunkwise Triton kernel 封装。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch

from .triton import (
    chunk_local_cumsum,
    gdn_chunk_bwd_dhu,
    gdn_chunk_bwd_dqkwg,
    gdn_chunk_bwd_dv_local,
    gdn_chunk_fwd_h,
    gdn_chunk_fwd_intra,
    gdn_chunk_fwd_o,
    gdn_chunk_l2norm_bwd,
    gdn_chunk_l2norm_fwd,
    gdn_chunk_prepare_wy_repr_bwd,
    gdn_chunk_recompute_w_u,
)


def _normalize_inputs(q, k, v, g, beta):
    """把输入从 [B, T, H, *] 转成 [B, H, T, *] 并保证连续。"""
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()
    v = v.transpose(1, 2).contiguous()
    g = g.transpose(1, 2).contiguous()
    beta = beta.transpose(1, 2).contiguous()
    return q, k, v, g, beta


class GatedDeltaNetChunkTritonFunction(torch.autograd.Function):
    """Gated DeltaNet chunkwise 训练前向 Triton 封装。"""

    @staticmethod
    def forward(ctx, q, k, v, g, beta, initial_state, output_final_state, chunk_size):
        q, k, v, g, beta = _normalize_inputs(q, k, v, g, beta)

        B, H, T, K = q.shape

        if T % chunk_size != 0:
            raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")

        # L2 归一化（在 Triton 内完成，与 gdn_recurrent 一致）
        q, inv_norm_q = gdn_chunk_l2norm_fwd(q)
        k, inv_norm_k = gdn_chunk_l2norm_fwd(k)

        # gate chunk 内 cumsum
        g = chunk_local_cumsum(g, chunk_size=chunk_size)

        # intra-chunk: A = (I - L)^{-1}
        A = gdn_chunk_fwd_intra(k, g, beta, chunk_size=chunk_size)

        # WY 表示: w, u
        w, u = gdn_chunk_recompute_w_u(k, v, beta, A, g, chunk_size=chunk_size)

        # chunk 间状态递推
        h, v_new, final_state = gdn_chunk_fwd_h(
            k,
            w,
            u,
            g,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )

        # 最终输出
        o = gdn_chunk_fwd_o(q, k, v_new, h, g, chunk_size=chunk_size)

        ctx.save_for_backward(
            q,
            k,
            v,
            g,
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
        """反向传播：计算 dq, dk, dv, dg, db, dh0。"""
        (
            q,
            k,
            v,
            g,
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

        # 1. 局部 dv（只含 chunk 内 causal 项）
        dv_local = gdn_chunk_bwd_dv_local(q, k, g, do, scale, chunk_size=chunk_size)

        # 2. 状态反向扫描
        dh, dh0, dv = gdn_chunk_bwd_dhu(
            q,
            k,
            w,
            g,
            do,
            dv_local,
            dht=dht,
            scale=scale,
            chunk_size=chunk_size,
        )

        # 3. dq / dk / dw / chunk 内 dg
        dq, dk, dw, dg = gdn_chunk_bwd_dqkwg(
            q,
            k,
            v_new,
            w,
            g,
            h,
            dh,
            do,
            dv,
            scale,
            chunk_size=chunk_size,
        )

        # 4. WY 表示反向
        dk2, dv2, db, dg2 = gdn_chunk_prepare_wy_repr_bwd(
            k,
            v,
            beta,
            g,
            A,
            dw,
            dv,
            chunk_size=chunk_size,
        )
        dk += dk2
        dv += dv2
        dg += dg2

        # 5. g 的 reverse cumsum（因为 forward 对 g 做过 cumsum）
        dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True)

        # 6. L2 norm 反向
        dq = gdn_chunk_l2norm_bwd(q, inv_norm_q, dq)
        dk = gdn_chunk_l2norm_bwd(k, inv_norm_k, dk)

        # 7. 转回外部 layout [B, T, H, *]
        dq = dq.transpose(1, 2)
        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)
        dg = dg.transpose(1, 2)
        db = db.transpose(1, 2)

        # 8. 匹配 forward 输入的梯度位置
        return (
            dq,
            dk,
            dv,
            dg,
            db,
            dh0,  # initial_state
            None,  # output_final_state
            None,  # chunk_size
        )


def gated_delta_net_chunk(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    chunk_size=64,
):
    """Gated DeltaNet chunkwise Triton 实现（训练前向）。

    接口与 `rwkv_ops.gdn_chunk.native_keras_op.gated_delta_net_chunk` 一致。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，已在外部过 sigmoid。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 64。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    return GatedDeltaNetChunkTritonFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        chunk_size,
    )

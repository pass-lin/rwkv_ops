# -*- coding: utf-8 -*-
"""
This file implements the forward and backward pass of a chunked delta rule attention mechanism,
optimized with Triton kernels for GPU acceleration. It includes functions for forward propagation,
backward gradient computation, and integration with PyTorch's autograd system.

该文件实现了分块 Delta Rule 注意力机制的前向与反向传播，
使用 Triton 内核进行 GPU 加速优化。包括前向传播、梯度反向传播函数，
并集成了 PyTorch 的自动求导系统。
"""

import warnings
from typing import Optional

import torch
import triton

# 导入内核实现模块 / Import kernel implementation modules
from .torch_kernel.chunk_A_bwd import chunk_dplr_bwd_dqk_intra
from .torch_kernel.chunk_A_fwd import chunk_dplr_fwd_intra
from .torch_kernel.chunk_h_bwd import chunk_dplr_bwd_dhu
from .torch_kernel.chunk_h_fwd import chunk_dplr_fwd_h
from .torch_kernel.chunk_o_bwd import (
    chunk_dplr_bwd_dAu,
    chunk_dplr_bwd_dv,
    chunk_dplr_bwd_o,
)
from .torch_kernel.chunk_o_fwd import chunk_dplr_fwd_o
from .torch_kernel.wy_fast_bwd import chunk_dplr_bwd_wy
from .torch_kernel.wy_fast_fwd import prepare_wy_repr_fwd
from .torch_kernel.cumsum import chunk_rwkv6_fwd_cumsum
from .get_torch_devices_info import (
    autocast_custom_bwd,
    autocast_custom_fwd,
    input_guard,
)


def cast(x, dtype):
    """
    Cast tensor x to specified dtype if not already in that format.

    如果张量 x 不是目标数据类型，则将其转换为目标类型。
    """
    if x is None or x.dtype == dtype:
        return x
    return x.to(dtype)


def chunk_dplr_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: float = 1,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    chunk_size: int = 16,
):
    """
    Forward pass of chunked delta rule attention.

    分块 Delta Rule 注意力机制的前向传播。

    Args:
        q (torch.Tensor): Queries tensor [B, T, H, K]
        k (torch.Tensor): Keys tensor [B, T, H, K]
        v (torch.Tensor): Values tensor [B, T, H, V]
        a (torch.Tensor): Activations tensor [B, T, H, K]
        b (torch.Tensor): Betas tensor [B, T, H, K]
        gk (torch.Tensor): Log decay tensor [B, T, H, K]
        scale (float): Scale factor for attention scores
        initial_state (Optional[torch.Tensor]): Initial state for recurrent processing
        output_final_state (bool): Whether to return final state
        chunk_size (int): Chunk size for processing

    Returns:
        o (torch.Tensor): Output tensor [B, T, H, V]
        final_state (Optional[torch.Tensor]): Final state if requested
    """
    T = q.shape[1]
    BT = min(chunk_size, max(triton.next_power_of_2(T), 16))
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT)

    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        chunk_size=BT,
    )

    del ge

    # Compute WY representation
    w, u, _ = prepare_wy_repr_fwd(ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, chunk_size=BT)

    del A_ab, A_ak

    h, v_new, final_state = chunk_dplr_fwd_h(
        kg=kg,
        bg=bg,
        v=v,
        w=w,
        u=u,
        gk=gi,
        initial_state=initial_state,
        output_final_state=output_final_state,
        chunk_size=BT,
    )

    del u, kg, bg, gi

    o = chunk_dplr_fwd_o(
        qg=qg, v=v, v_new=v_new, A_qk=A_qk, A_qb=A_qb, h=h, chunk_size=BT
    )
    del v_new, h, A_qk, A_qb

    return o, final_state


def chunk_dplr_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    initial_state: torch.Tensor,
    scale,
    do,
    dht,
    BT: int = 16,
):
    """
    Backward pass of chunked delta rule attention.

    分块 Delta Rule 注意力机制的反向传播。

    Args:
        q (torch.Tensor): Queries tensor [B, T, H, K]
        k (torch.Tensor): Keys tensor [B, T, H, K]
        v (torch.Tensor): Values tensor [B, T, H, V]
        a (torch.Tensor): Activations tensor [B, T, H, K]
        b (torch.Tensor): Betas tensor [B, T, H, K]
        gk (torch.Tensor): Log decay tensor [B, T, H, K]
        initial_state (torch.Tensor): Initial state for recurrent processing
        scale (float): Scale factor for attention scores
        do (torch.Tensor): Gradient of outputs
        dht (torch.Tensor): Gradient of final hidden state
        BT (int): Chunk size for processing

    Returns:
        dq (torch.Tensor): Gradient of queries
        dk (torch.Tensor): Gradient of keys
        dv (torch.Tensor): Gradient of values
        da (torch.Tensor): Gradient of activations
        db (torch.Tensor): Gradient of betas
        dgk (torch.Tensor): Gradient of log decays
        dh0 (torch.Tensor): Gradient of initial state
    """
    gi, ge = chunk_rwkv6_fwd_cumsum(gk, BT)
    A_ab, A_qk, A_ak, A_qb, qg, kg, ag, bg = chunk_dplr_fwd_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        scale=scale,
        chunk_size=BT,
    )
    w, u, A_ab_inv = prepare_wy_repr_fwd(
        ag=ag, A_ab=A_ab, A_ak=A_ak, v=v, chunk_size=BT
    )
    del A_ab
    h, v_new, _ = chunk_dplr_fwd_h(
        kg=kg, bg=bg, v=v, w=w, u=u, gk=gi, initial_state=initial_state, chunk_size=BT
    )
    del u

    dv_new_intra, dA_qk, dA_qb = chunk_dplr_bwd_dAu(
        v=v, v_new=v_new, do=do, A_qb=A_qb, scale=scale, chunk_size=BT
    )

    dh, dh0, dv_new = chunk_dplr_bwd_dhu(
        qg=qg,
        bg=bg,
        w=w,
        gk=gi,
        h0=initial_state,
        dht=dht,
        do=do,
        dv=dv_new_intra,
        chunk_size=BT,
    )

    dv = chunk_dplr_bwd_dv(A_qk=A_qk, kg=kg, do=do, dh=dh, chunk_size=BT)
    del A_qk

    dqg, dkg, dw, dbg, dgk_last = chunk_dplr_bwd_o(
        k=kg,
        b=bg,
        v=v,
        v_new=v_new,
        do=do,
        h=h,
        dh=dh,
        dv=dv,
        w=w,
        gk=gi,
        chunk_size=BT,
        scale=scale,
    )
    del v_new

    dA_ab, dA_ak, dv, dag = chunk_dplr_bwd_wy(
        A_ab_inv=A_ab_inv,
        A_ak=A_ak,
        v=v,
        ag=ag,
        dw=dw,
        du=dv_new,
        dv0=dv,
        chunk_size=BT,
    )
    del A_ak

    dq, dk, da, db, dgk = chunk_dplr_bwd_dqk_intra(
        q=q,
        k=k,
        a=a,
        b=b,
        gi=gi,
        ge=ge,
        dAqk=dA_qk,
        dAqb=dA_qb,
        dAak=dA_ak,
        dAab=dA_ab,
        dgk_last=dgk_last,
        dqg=dqg,
        dkg=dkg,
        dag=dag,
        dbg=dbg,
        chunk_size=BT,
        scale=scale,
    )

    return (
        dq.to(q),
        dk.to(k),
        dv.to(v),
        da.to(a),
        db.to(b),
        dgk.to(gk),
        None,
        dh0,
        None,
        None,
    )


class ChunkDPLRDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        gk: torch.Tensor,
        scale: float = 1,
        initial_state: torch.Tensor = None,
        output_final_state: bool = True,
        cu_seqlens: Optional[torch.LongTensor] = None,
    ):
        """
        Forward function with autograd support.

        支持自动求导的前向函数。
        """
        chunk_size = 16
        o, final_state = chunk_dplr_fwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )
        ctx.save_for_backward(q, k, v, a, b, gk, initial_state)
        ctx.cu_seqlens = cu_seqlens
        ctx.scale = scale
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(ctx, do: torch.Tensor, dht: torch.Tensor):
        """
        Backward function with autograd support.

        支持自动求导的反向函数。
        """
        q, k, v, a, b, gk, initial_state = ctx.saved_tensors
        BT = ctx.chunk_size
        cu_seqlens = ctx.cu_seqlens
        scale = ctx.scale

        return chunk_dplr_bwd(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            gk=gk,
            scale=scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            BT=BT,
        )


@torch.compiler.disable
def chunk_dplr_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    gk: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    r"""
    Main interface function for chunked delta rule attention.

    分块 Delta Rule 注意力机制的主要接口函数。

    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`
        v (torch.Tensor):
            values of shape `[B, T, H, V]`
        a (torch.Tensor):
            activations of shape `[B, T, H, K]`
        b (torch.Tensor):
            betas of shape `[B, T, H, K]`
        gk (torch.Tensor):
            gk of shape `[B, T, H, K]`  decay term in log space!
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format, which is not supported for variable-length inputs.
            Default: `False`.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]` if `head_first=False` else `[B, H, T, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
    """
    if q.dtype == torch.float32:
        warnings.warn(
            """ChunkDeltaRuleFunction does not support float32 on some platforms. Please use bfloat16/float16.
            If you want to use float32, please solve the issue by yourself.""",
            category=RuntimeWarning,
            stacklevel=2,
        )
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    scale = k.shape[-1] ** -0.5 if scale is None else scale
    o, final_state = ChunkDPLRDeltaRuleFunction.apply(
        q,
        k,
        v,
        a,
        b,
        gk,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
    )
    return o, final_state


def chunk_rwkv7(
    r: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    w: torch.Tensor = None,
    log_w: torch.Tensor = None,
    scale: float = 1.0,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
):
    """
    Interface function for RWKV-7 attention.

    RWKV-7 注意力机制的接口函数。
    """
    if w is not None:
        log_w = -torch.exp(w)
    else:
        assert log_w is not None, "Either w or log_w must be provided!"

    return chunk_dplr_delta_rule(
        q=r,
        k=k,
        v=v,
        a=a,
        b=b,
        gk=log_w,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def transpose_head(x, head_first):
    """
    Transpose between head-first and time-first formats.

    在 head-first 和 time-first 格式之间转置。
    """
    if head_first:
        x = torch.permute(x, dims=(0, 2, 1, 3))
    out = cast(x, torch.bfloat16).contiguous()
    return out


def generalized_delta_rule(
    r: torch.Tensor,
    w: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    initial_state: torch.Tensor = None,
    output_final_state: bool = True,
    head_first: bool = False,
):
    """
    Generalized delta rule attention interface.

    泛化 Delta Rule 注意力机制接口。
    """
    dtype = r.dtype
    r = transpose_head(r, head_first)
    k = transpose_head(k, head_first)
    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    if w.device.type == "cuda":
        out, state = chunk_rwkv7(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    else:
        from .native_keras_op import generalized_delta_rule

        out, state = generalized_delta_rule(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=initial_state,
            output_final_state=output_final_state,
        )
    out = transpose_head(out, head_first)
    if output_final_state:
        return out, cast(state, dtype)
    else:
        return out

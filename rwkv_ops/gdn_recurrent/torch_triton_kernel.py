"""PyTorch 版 Gated DeltaNet recurrent Triton kernel 封装。"""

import torch
import triton

from .triton_kernel import (
    gated_delta_net_recurrent_fwd_kernel,
    gated_delta_net_recurrent_bwd_kernel,
    gated_delta_net_recurrent_inference_fwd_kernel,
    gated_delta_net_recurrent_single_step_fwd_kernel,
)


def _normalize_inputs(q, k, v, g, beta, head_first):
    """把输入统一转成 [B, H, T, *] 并保证连续。"""
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        g = g.transpose(1, 2)
        beta = beta.transpose(1, 2)
    return [x.contiguous() for x in [q, k, v, g, beta]]


def _prepare_initial_state(initial_state, B, H, K, V, device):
    """准备 float32 连续初始 state，支持 [1, H, K, V] 广播。"""
    if initial_state is None:
        return torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
    h0 = initial_state.to(torch.float32).contiguous()
    if h0.shape[0] == 1 and B > 1:
        h0 = h0.expand(B, *h0.shape[1:]).contiguous()
    return h0


def _make_recurrent_grid(B, H, V, BV):
    """recurrent kernel 的启动 grid。"""
    return (triton.cdiv(V, BV) * B * H,)


def _make_gated_delta_net_recurrent_triton_function(chunk_size):
    """按 chunk_size 构造 Gated DeltaNet recurrent 训练 Triton 封装类。"""

    class GatedDeltaNetRecurrentTritonFunction(torch.autograd.Function):
        """Gated DeltaNet recurrent 训练前向 Triton 封装。"""

        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            output_final_state,
            head_first,
        ):
            q, k, v, g, beta = _normalize_inputs(q, k, v, g, beta, head_first)

            B, H, T, K = q.shape
            V = v.shape[-1]
            scale = K**-0.5

            if T % chunk_size != 0:
                raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")

            o = torch.empty_like(v)
            final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
            kv_mem_out = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
            inv_norm_q = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
            inv_norm_k = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
            num_chunks = T // chunk_size
            state_chkp = torch.empty(
                B, H, num_chunks, K, V, dtype=torch.float32, device=q.device
            )
            h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

            BK = triton.next_power_of_2(K)
            BV = min(128, triton.next_power_of_2(V))
            grid = _make_recurrent_grid(B, H, V, BV)

            gated_delta_net_recurrent_fwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                o=o,
                kv_mem_out=kv_mem_out,
                state_chkp=state_chkp,
                inv_norm_q=inv_norm_q,
                inv_norm_k=inv_norm_k,
                h0=h0,
                ht=final_state,
                scale=scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                CHUNK_LEN=chunk_size,
                USE_INITIAL_STATE=True,
                STORE_FINAL_STATE=True,
            )

            ctx.save_for_backward(
                q, k, v, g, beta, h0, kv_mem_out, state_chkp, inv_norm_q, inv_norm_k
            )
            ctx.head_first = head_first
            ctx.output_final_state = output_final_state
            ctx.chunk_size = chunk_size

            if not head_first:
                o = o.transpose(1, 2)

            if output_final_state:
                return o, final_state
            return o, None

        @staticmethod
        def backward(ctx, do, dht):
            q, k, v, g, beta, h0, kv_mem_out, state_chkp, inv_norm_q, inv_norm_k = (
                ctx.saved_tensors
            )

            if not ctx.head_first:
                do = do.transpose(1, 2)
            do = do.contiguous()

            B, H, T, K = q.shape
            V = v.shape[-1]
            scale = K**-0.5
            CHUNK_LEN = ctx.chunk_size

            # 梯度缓冲区用 float32 累加，避免 atomic_add 与低精度问题。
            dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
            dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
            dv = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
            dg = torch.zeros(B, H, T, dtype=torch.float32, device=q.device)
            dbeta = torch.zeros(B, H, T, dtype=torch.float32, device=q.device)
            dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

            use_final_state_gradient = dht is not None
            if use_final_state_gradient:
                dht = dht.to(torch.float32).contiguous()
            else:
                dht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

            BK = triton.next_power_of_2(K)
            BV = min(128, triton.next_power_of_2(V))
            grid = _make_recurrent_grid(B, H, V, BV)

            gated_delta_net_recurrent_bwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                do=do,
                dht=dht,
                kv_mem_out=kv_mem_out,
                inv_norm_q=inv_norm_q,
                inv_norm_k=inv_norm_k,
                h0=h0,
                state_chkp=state_chkp,
                dq=dq,
                dk=dk,
                dv=dv,
                dg=dg,
                dbeta=dbeta,
                dh0=dh0,
                scale=scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                CHUNK_LEN=CHUNK_LEN,
                USE_FINAL_STATE_GRADIENT=use_final_state_gradient,
            )

            if not ctx.head_first:
                dq = dq.transpose(1, 2)
                dk = dk.transpose(1, 2)
                dv = dv.transpose(1, 2)
                dg = dg.transpose(1, 2)
                dbeta = dbeta.transpose(1, 2)

            input_dtype = q.dtype
            dq = dq.to(input_dtype)
            dk = dk.to(input_dtype)
            dv = dv.to(input_dtype)
            dg = dg.to(input_dtype)
            dbeta = dbeta.to(input_dtype)
            dh0 = dh0.to(torch.float32)

            return dq, dk, dv, dg, dbeta, dh0, None, None

    return GatedDeltaNetRecurrentTritonFunction


def _make_gated_delta_net_recurrent_inference_triton_function(chunk_size):
    """按 chunk_size 构造 Gated DeltaNet recurrent 推理 Triton 封装类。"""

    class GatedDeltaNetRecurrentInferenceTritonFunction(torch.autograd.Function):
        """Gated DeltaNet recurrent 推理前向 Triton 封装（无反向）。"""

        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            output_final_state,
            head_first,
        ):
            q, k, v, g, beta = _normalize_inputs(q, k, v, g, beta, head_first)

            B, H, T, K = q.shape
            V = v.shape[-1]
            scale = K**-0.5

            if T % chunk_size != 0:
                raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")

            o = torch.empty_like(v)
            final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
            h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

            BK = triton.next_power_of_2(K)
            BV = min(128, triton.next_power_of_2(V))
            grid = _make_recurrent_grid(B, H, V, BV)

            gated_delta_net_recurrent_inference_fwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                o=o,
                h0=h0,
                ht=final_state,
                scale=scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                CHUNK_LEN=chunk_size,
                USE_INITIAL_STATE=True,
                STORE_FINAL_STATE=True,
            )

            if not head_first:
                o = o.transpose(1, 2)

            if output_final_state:
                return o, final_state
            return o, None

        @staticmethod
        def backward(ctx, do, dht):
            raise NotImplementedError(
                "Gated DeltaNet recurrent inference does not support backward."
            )

    return GatedDeltaNetRecurrentInferenceTritonFunction


class GatedDeltaNetRecurrentSingleStepTritonFunction(torch.autograd.Function):
    """Gated DeltaNet recurrent 单步 RNN 前向 Triton 封装（无反向）。"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        head_first,
    ):
        if not head_first:
            raise NotImplementedError(
                "gated_delta_net_recurrent_single_step currently only supports head_first=True."
            )

        q, k, v, g, beta = [x.contiguous() for x in [q, k, v, g, beta]]

        B, H, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        o = torch.empty_like(v)
        next_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
        h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

        BK = triton.next_power_of_2(K)
        BV = min(128, triton.next_power_of_2(V))
        grid = (triton.cdiv(V, BV) * B * H,)

        gated_delta_net_recurrent_single_step_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=o,
            h0=h0,
            ht=next_state,
            scale=scale,
            B=B,
            H=H,
            K=K,
            V=V,
            BK=BK,
            BV=BV,
            USE_INITIAL_STATE=True,
            STORE_FINAL_STATE=True,
        )

        if output_final_state:
            return o, next_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "Gated DeltaNet recurrent single step does not support backward."
        )


def gated_delta_net_recurrent(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    chunk_size=16,
):
    """Gated DeltaNet recurrent 训练算子（PyTorch Triton 实现）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        RuntimeError: 输入不在 CUDA 设备上。
        ValueError: T 不被 chunk_size 整除。
    """
    if q.device.type != "cuda":
        raise RuntimeError("Gated DeltaNet Triton kernel only supports CUDA devices.")

    Op = _make_gated_delta_net_recurrent_triton_function(chunk_size)
    return Op.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        head_first,
    )


def gated_delta_net_recurrent_inference(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """Gated DeltaNet recurrent 推理算子（PyTorch Triton 实现，无梯度）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        RuntimeError: 输入不在 CUDA 设备上。
        ValueError: T 不被 chunk_size 整除。
    """
    if q.device.type != "cuda":
        raise RuntimeError("Gated DeltaNet Triton kernel only supports CUDA devices.")

    Op = _make_gated_delta_net_recurrent_inference_triton_function(chunk_size)
    return Op.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        head_first,
    )


def gated_delta_net_recurrent_single_step(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=True,
    head_first=True,
    chunk_size=16,
):
    """Gated DeltaNet recurrent 单步 RNN 算子（PyTorch Triton 实现）。

    Args:
        q, k: [B, H, K]，查询与键。
        v: [B, H, V]，值。
        g: [B, H]，log-space decay。
        beta: [B, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回下一步 state。
        head_first: bool，输入输出是否 head 维优先。单步默认 True（[B, H, *]）。
        chunk_size: int，chunk 长度，默认 16。单步 kernel 忽略该值。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        RuntimeError: 输入不在 CUDA 设备上。
        NotImplementedError: head_first=False 尚未支持。
    """
    if q.device.type != "cuda":
        raise RuntimeError("Gated DeltaNet Triton kernel only supports CUDA devices.")

    return GatedDeltaNetRecurrentSingleStepTritonFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        output_final_state,
        head_first,
    )

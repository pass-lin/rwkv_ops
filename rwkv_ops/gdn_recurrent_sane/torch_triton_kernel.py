"""PyTorch 版 Gated DeltaNet recurrent SANE Triton kernel 封装。"""

import warnings

import torch
import triton

from .triton_kernel import (
    gated_delta_net_recurrent_sane_bwd_kernel,
    gated_delta_net_recurrent_sane_fwd_kernel,
    gated_delta_net_recurrent_sane_inference_fwd_kernel,
    gated_delta_net_recurrent_sane_single_step_fwd_kernel,
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


def _normalize_tau_mask(tau, mask, B, H, T, chunk_size, device):
    """把 tau/mask 转成内部布局 [B, H, T//chunk_size] / [B, T//chunk_size]。"""
    num_chunks = T // chunk_size
    use_mask = mask is not None
    if num_chunks > 0:
        tau = tau.transpose(1, 2).contiguous().to(device, torch.float32)
        if use_mask:
            mask = mask.contiguous().to(device, torch.float32)
        else:
            mask = torch.ones(B, num_chunks, dtype=torch.float32, device=device)
        return tau, mask, use_mask
    # T < chunk_size 时创建一个不会被读取的占位符。
    tau_dummy = torch.zeros(B, H, 1, dtype=torch.float32, device=device)
    mask_dummy = torch.ones(B, 1, dtype=torch.float32, device=device)
    return tau_dummy, mask_dummy, use_mask


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


def _make_gated_delta_net_recurrent_sane_triton_function(chunk_size):
    """按 chunk_size 构造 Gated DeltaNet recurrent SANE 训练 Triton 封装类。"""

    class GatedDeltaNetRecurrentSaneTritonFunction(torch.autograd.Function):
        """Gated DeltaNet recurrent SANE 训练前向 Triton 封装。"""

        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            tau,
            mask,
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

            mask_is_none = mask is None
            tau, mask, use_mask = _normalize_tau_mask(
                tau, mask, B, H, T, chunk_size, q.device
            )

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

            gated_delta_net_recurrent_sane_fwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                tau=tau,
                mask=mask,
                h0=h0,
                o=o,
                kv_mem_out=kv_mem_out,
                state_chkp=state_chkp,
                inv_norm_q=inv_norm_q,
                inv_norm_k=inv_norm_k,
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
                USE_MASK=use_mask,
            )

            ctx.save_for_backward(
                q,
                k,
                v,
                g,
                beta,
                tau,
                mask,
                h0,
                kv_mem_out,
                state_chkp,
                inv_norm_q,
                inv_norm_k,
            )
            ctx.head_first = head_first
            ctx.output_final_state = output_final_state
            ctx.chunk_size = chunk_size
            ctx.use_mask = use_mask
            ctx.num_chunks = num_chunks

            if not head_first:
                o = o.transpose(1, 2)

            if not output_final_state:
                return o, None

            if mask_is_none:
                warnings.warn(
                    "[gdn_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
                    "由于未提供 padding mask，返回的 final_state 可能被污染，"
                    "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
                    "[gdn_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
                    "The returned final_state is set to None because padding chunks "
                    "may contaminate the state. Provide an explicit mask to obtain final_state.",
                    UserWarning,
                    stacklevel=2,
                )
                return o, None

            return o, final_state

        @staticmethod
        def backward(ctx, do, dht):
            (
                q,
                k,
                v,
                g,
                beta,
                tau,
                mask,
                h0,
                kv_mem_out,
                state_chkp,
                inv_norm_q,
                inv_norm_k,
            ) = ctx.saved_tensors

            if not ctx.head_first:
                do = do.transpose(1, 2)
            do = do.contiguous()

            B, H, T, K = q.shape
            V = v.shape[-1]
            scale = K**-0.5
            CHUNK_LEN = ctx.chunk_size
            num_chunks = ctx.num_chunks
            use_mask = ctx.use_mask

            dq = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
            dk = torch.zeros(B, H, T, K, dtype=torch.float32, device=q.device)
            dv = torch.zeros(B, H, T, V, dtype=torch.float32, device=q.device)
            dg = torch.zeros(B, H, T, dtype=torch.float32, device=q.device)
            dbeta = torch.zeros(B, H, T, dtype=torch.float32, device=q.device)
            dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

            if num_chunks > 0:
                dtau = torch.zeros(
                    B, H, num_chunks, dtype=torch.float32, device=q.device
                )
            else:
                dtau = torch.empty(B, H, 0, dtype=torch.float32, device=q.device)

            use_final_state_gradient = dht is not None
            if use_final_state_gradient:
                dht = dht.to(torch.float32).contiguous()
            else:
                dht = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

            BK = triton.next_power_of_2(K)
            BV = min(128, triton.next_power_of_2(V))
            grid = _make_recurrent_grid(B, H, V, BV)

            gated_delta_net_recurrent_sane_bwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                tau=tau,
                mask=mask,
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
                dtau=dtau,
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
                USE_MASK=use_mask,
            )

            if not ctx.head_first:
                dq = dq.transpose(1, 2)
                dk = dk.transpose(1, 2)
                dv = dv.transpose(1, 2)
                dg = dg.transpose(1, 2)
                dbeta = dbeta.transpose(1, 2)

            # dtau 内部布局为 [B, H, T//CHUNK_LEN]，需转回外部 [B, T//CHUNK_LEN, H]。
            if num_chunks > 0:
                dtau = dtau.transpose(1, 2)

            input_dtype = q.dtype
            dq = dq.to(input_dtype)
            dk = dk.to(input_dtype)
            dv = dv.to(input_dtype)
            dg = dg.to(input_dtype)
            dbeta = dbeta.to(input_dtype)
            dh0 = dh0.to(torch.float32)

            return (
                dq,
                dk,
                dv,
                dg,
                dbeta,
                dtau,
                None,
                dh0,
                None,
                None,
            )

    return GatedDeltaNetRecurrentSaneTritonFunction


def _make_gated_delta_net_recurrent_sane_inference_triton_function(chunk_size):
    """按 chunk_size 构造 Gated DeltaNet recurrent SANE 推理 Triton 封装类。"""

    class GatedDeltaNetRecurrentSaneInferenceTritonFunction(torch.autograd.Function):
        """Gated DeltaNet recurrent SANE 推理前向 Triton 封装（无反向）。"""

        @staticmethod
        def forward(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            tau,
            mask,
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

            mask_is_none = mask is None
            tau, mask, use_mask = _normalize_tau_mask(
                tau, mask, B, H, T, chunk_size, q.device
            )

            o = torch.empty_like(v)
            final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
            h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

            BK = triton.next_power_of_2(K)
            BV = min(128, triton.next_power_of_2(V))
            grid = _make_recurrent_grid(B, H, V, BV)

            gated_delta_net_recurrent_sane_inference_fwd_kernel[grid](
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                tau=tau,
                mask=mask,
                h0=h0,
                o=o,
                ht=final_state,
                scale=scale,
                B=B,
                H=H,
                T=T,
                K=K,
                V=V,
                BK=BK,
                BV=BV,
                USE_INITIAL_STATE=True,
                STORE_FINAL_STATE=True,
                USE_MASK=use_mask,
                CHUNK_LEN=chunk_size,
            )

            if not head_first:
                o = o.transpose(1, 2)

            if not output_final_state:
                return o, None

            if mask_is_none:
                warnings.warn(
                    "[gdn_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
                    "由于未提供 padding mask，返回的 final_state 可能被污染，"
                    "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
                    "[gdn_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
                    "The returned final_state is set to None because padding chunks "
                    "may contaminate the state. Provide an explicit mask to obtain final_state.",
                    UserWarning,
                    stacklevel=2,
                )
                return o, None

            return o, final_state

        @staticmethod
        def backward(ctx, do, dht):
            raise NotImplementedError(
                "Gated DeltaNet recurrent SANE inference does not support backward."
            )

    return GatedDeltaNetRecurrentSaneInferenceTritonFunction


class GatedDeltaNetRecurrentSaneSingleStepTritonFunction(torch.autograd.Function):
    """Gated DeltaNet recurrent SANE 单步 RNN 前向 Triton 封装（无反向）。"""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        tau,
        do_sane,
        initial_state,
        output_final_state,
        head_first,
    ):
        if not head_first:
            raise NotImplementedError(
                "gated_delta_net_recurrent_sane_single_step currently only supports "
                "head_first=True."
            )

        q, k, v, g, beta = [x.contiguous() for x in [q, k, v, g, beta]]
        tau = tau.contiguous().to(torch.float32)
        do_sane = do_sane.contiguous().to(torch.float32)

        B, H, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        o = torch.empty_like(v)
        next_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
        h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

        BK = triton.next_power_of_2(K)
        BV = min(128, triton.next_power_of_2(V))
        grid = (triton.cdiv(V, BV) * B * H,)

        gated_delta_net_recurrent_sane_single_step_fwd_kernel[grid](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            tau=tau,
            do_sane=do_sane,
            h0=h0,
            o=o,
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
            "Gated DeltaNet recurrent SANE single step does not support backward."
        )


def gated_delta_net_recurrent_sane(
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    chunk_size=16,
):
    """Gated DeltaNet recurrent SANE 训练算子（PyTorch Triton 实现）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        tau: [B, T//chunk_size, H]，float32。阈值，必须 > 0。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE。
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
        raise RuntimeError(
            "Gated DeltaNet SANE Triton kernel only supports CUDA devices."
        )

    Op = _make_gated_delta_net_recurrent_sane_triton_function(chunk_size)
    return Op.apply(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        initial_state,
        output_final_state,
        head_first,
    )


def gated_delta_net_recurrent_sane_inference(
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """Gated DeltaNet recurrent SANE 推理算子（PyTorch Triton 实现，无梯度）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        tau: [B, T//chunk_size, H]，float32。
        mask: [B, T//chunk_size]，float32 或 None。
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
        raise RuntimeError(
            "Gated DeltaNet SANE Triton kernel only supports CUDA devices."
        )

    Op = _make_gated_delta_net_recurrent_sane_inference_triton_function(chunk_size)
    return Op.apply(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        initial_state,
        output_final_state,
        head_first,
    )


def gated_delta_net_recurrent_sane_single_step(
    q,
    k,
    v,
    g,
    beta,
    tau,
    do_sane,
    initial_state=None,
    output_final_state=True,
    head_first=True,
    chunk_size=16,
):
    """Gated DeltaNet recurrent SANE 单步 RNN 算子（PyTorch Triton 实现）。

    Args:
        q, k: [B, H, K]，查询与键。
        v: [B, H, V]，值。
        g: [B, H]，log-space decay。
        beta: [B, H]，写入强度，必须已在外部过 sigmoid 并落在 (0,1)。
        tau: [B, H]，float32。
        do_sane: [B]，float32。>0 执行 SANE。
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
        raise RuntimeError(
            "Gated DeltaNet SANE Triton kernel only supports CUDA devices."
        )

    return GatedDeltaNetRecurrentSaneSingleStepTritonFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        tau,
        do_sane,
        initial_state,
        output_final_state,
        head_first,
    )

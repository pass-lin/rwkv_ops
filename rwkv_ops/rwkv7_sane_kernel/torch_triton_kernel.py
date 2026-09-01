"""PyTorch 版 RWKV7-SANE Triton kernel 封装。"""

import warnings

import torch
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.numpy import transpose

from .triton_kernel import (
    rwkv7_sane_bwd_kernel,
    rwkv7_sane_bwd_kernel_with_mask,
    rwkv7_sane_fwd_kernel,
    rwkv7_sane_fwd_kernel_with_mask,
)


def transpose_head(x, head_first):
    """统一输入布局为 [B, N, T, H]。"""
    if head_first:
        return transpose(x, (0, 2, 1, 3))
    return x


def _apply_sane_to_final_state(state, tau, mask=None):
    """对最终 state 应用 State Anomaly Neutralization。

    Args:
        state: [B, N, H, H], float32。
        tau: [B, N, C], float32。
        mask: [B, C], float32（可选）。

    Returns:
        [B, N, H, H], float32。
    """
    B, N, H, _ = state.shape
    last_tau = tau[:, :, -1].view(B, N, 1, 1)
    tau_safe = torch.clamp(last_tau, min=1e-6)
    sane_state = last_tau * torch.tanh(state / tau_safe)
    if mask is None:
        return sane_state
    last_mask = mask[:, -1].view(B, 1, 1, 1)
    return torch.where(last_mask > 0, sane_state, state)


def _make_sane_triton_op_with_mask(chunk_size: int):
    class TritonSnWindBacksteppingWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, mask, h0):
            B, N, T, H = w.shape
            DTYPE = q.dtype

            if T % chunk_size != 0:
                raise ValueError(
                    f"Sequence length T={T} must be divisible by {chunk_size}"
                )
            if chunk_size < 16:
                raise ValueError(
                    f"Triton kernel requires chunk_size >= 16, got {chunk_size}"
                )
            if H != 64:
                raise ValueError(
                    f"Triton SANE kernel currently only supports Head Size = 64, got {H}"
                )

            out = torch.empty_like(v)
            sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
            state_chkp = torch.empty(
                B, N, T // chunk_size, H, H, dtype=torch.float32, device=w.device
            )

            def grid(META):
                return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

            rwkv7_sane_fwd_kernel_with_mask[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                TAU=tau,
                MASK=mask,
                H0=h0,
                B_BATCH=B,
                N_HEAD=N,
                T_LEN=T,
                OUT=out,
                SA_OUT=sa_out,
                STATE_CHKP=state_chkp,
                H_SIZE=H,
                CHUNK_LEN=chunk_size,
            )

            ctx.save_for_backward(w, q, k, v, a, b, tau, mask, state_chkp, sa_out)

            last_state = state_chkp[:, :, -1, :, :].clone()
            last_state = _apply_sane_to_final_state(last_state, tau, mask=mask)
            return out.to(DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            w, q, k, v, a, b, tau, mask, state_chkp, sa_out = ctx.saved_tensors
            B, N, T, H = w.shape
            DTYPE = q.dtype

            dw = torch.empty_like(w)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            dtau = torch.empty(
                B, N, T // chunk_size, dtype=torch.float32, device=w.device
            )
            dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

            dy = dy.contiguous()
            if dht is None:
                dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
            else:
                dht = dht.contiguous().to(torch.float32)

            def grid(META):
                return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

            rwkv7_sane_bwd_kernel_with_mask[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                SA=sa_out,
                STATE_CHKP=state_chkp,
                TAU=tau,
                MASK=mask,
                B_BATCH=B,
                N_HEAD=N,
                T_LEN=T,
                DY=dy,
                DHT=dht,
                DR=dq,
                DW=dw,
                DK=dk,
                DV=dv,
                DA=da,
                DB=db,
                DTAU=dtau,
                DH0=dh0,
                H_SIZE=H,
                CHUNK_LEN=chunk_size,
            )

            return (
                dw.to(DTYPE),
                dq.to(DTYPE),
                dk.to(DTYPE),
                dv.to(DTYPE),
                da.to(DTYPE),
                db.to(DTYPE),
                dtau,
                None,  # mask has no gradient
                dh0,
            )

    return TritonSnWindBacksteppingWithMask


def _make_sane_triton_op_no_mask(chunk_size: int):
    class TritonSnWindBacksteppingNoMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, h0):
            B, N, T, H = w.shape
            DTYPE = q.dtype

            if T % chunk_size != 0:
                raise ValueError(
                    f"Sequence length T={T} must be divisible by {chunk_size}"
                )
            if chunk_size < 16:
                raise ValueError(
                    f"Triton kernel requires chunk_size >= 16, got {chunk_size}"
                )
            if H != 64:
                raise ValueError(
                    f"Triton SANE kernel currently only supports Head Size = 64, got {H}"
                )

            out = torch.empty_like(v)
            sa_out = torch.empty(B, N, T, H, dtype=torch.float32, device=w.device)
            state_chkp = torch.empty(
                B, N, T // chunk_size, H, H, dtype=torch.float32, device=w.device
            )

            def grid(META):
                return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

            rwkv7_sane_fwd_kernel[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                TAU=tau,
                H0=h0,
                B_BATCH=B,
                N_HEAD=N,
                T_LEN=T,
                OUT=out,
                SA_OUT=sa_out,
                STATE_CHKP=state_chkp,
                H_SIZE=H,
                CHUNK_LEN=chunk_size,
            )

            ctx.save_for_backward(w, q, k, v, a, b, tau, state_chkp, sa_out)

            last_state = state_chkp[:, :, -1, :, :].clone()
            last_state = _apply_sane_to_final_state(last_state, tau, mask=None)
            return out.to(DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            w, q, k, v, a, b, tau, state_chkp, sa_out = ctx.saved_tensors
            B, N, T, H = w.shape
            DTYPE = q.dtype

            dw = torch.empty_like(w)
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            da = torch.empty_like(a)
            db = torch.empty_like(b)
            dtau = torch.empty(
                B, N, T // chunk_size, dtype=torch.float32, device=w.device
            )
            dh0 = torch.empty(B, N, H, H, dtype=torch.float32, device=w.device)

            dy = dy.contiguous()
            if dht is None:
                dht = torch.zeros(B, N, H, H, dtype=torch.float32, device=w.device)
            else:
                dht = dht.contiguous().to(torch.float32)

            def grid(META):
                return ((B + META["MINI_BSZ"] - 1) // META["MINI_BSZ"], N)

            rwkv7_sane_bwd_kernel[grid](
                R=q,
                W=w,
                K=k,
                V=v,
                A=a,
                B_param=b,
                SA=sa_out,
                STATE_CHKP=state_chkp,
                TAU=tau,
                B_BATCH=B,
                N_HEAD=N,
                T_LEN=T,
                DY=dy,
                DHT=dht,
                DR=dq,
                DW=dw,
                DK=dk,
                DV=dv,
                DA=da,
                DB=db,
                DTAU=dtau,
                DH0=dh0,
                H_SIZE=H,
                CHUNK_LEN=chunk_size,
            )

            return (
                dw.to(DTYPE),
                dq.to(DTYPE),
                dk.to(DTYPE),
                dv.to(DTYPE),
                da.to(DTYPE),
                db.to(DTYPE),
                dtau,
                dh0,
            )

    return TritonSnWindBacksteppingNoMask


def generalized_delta_rule_sane(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（Torch-Triton chunkwise 训练版）。

    Args:
        r, w, k, v, a, b: [B, T, H, K]（head_first=False）或 [B, H, T, K]（head_first=True），bfloat16。
            T 必须被 chunk_size 整除；当前 Triton kernel 仅支持 K=64。
        tau: [B, T//chunk_size, H]，float32。阈值，必须严格 > 1。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先。
        chunk_size: int，chunk 长度，必须整除序列长度。

    Returns:
        out: [B, T, H, K]，与输入同 dtype。
        final_state: [B, H, K, K]，float32。
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 K 不等于 64，或 tau/mask 形状不匹配。
    """
    if w.device.type != "cuda":
        from .native_keras_op import generalized_delta_rule_sane

        return generalized_delta_rule_sane(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            mask=mask,
            initial_state=initial_state,
            output_final_state=output_final_state,
            head_first=head_first,
            chunk_size=chunk_size,
        )

    # 统一转换为 head-first [B, N, T, H]。
    if not head_first:
        r = r.transpose(1, 2)
        w = w.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        a = a.transpose(1, 2)
        b = b.transpose(1, 2)

    r, w, k, v, a, b = [x.contiguous() for x in [r, w, k, v, a, b]]

    B, N, T, H = w.shape

    if T % chunk_size != 0:
        raise ValueError(
            f"Triton SANE kernel requires sequence length T={T} to be divisible by {chunk_size}"
        )

    # tau 公共接口为 [B, T//chunk_size, N]；转成 [B, N, T//chunk_size]。
    tau = cast(tau, "float32").contiguous()
    if tau.shape != (B, T // chunk_size, N):
        raise ValueError(
            f"tau shape {tuple(tau.shape)} does not match expected "
            f"(B={B}, T//chunk_size={T // chunk_size}, N={N})"
        )
    tau = tau.transpose(1, 2).contiguous()

    if initial_state is None:
        initial_state = torch.zeros(B, N, H, H, dtype=torch.float32, device=r.device)
    else:
        initial_state = initial_state.to(torch.float32).contiguous()

    use_mask = output_final_state and mask is not None

    if use_mask:
        mask = cast(mask, "float32").contiguous()
        if mask.shape != (B, T // chunk_size):
            raise ValueError(
                f"mask shape {tuple(mask.shape)} must match (B, T//chunk_size) = ({B}, {T // chunk_size})"
            )
        Op = _make_sane_triton_op_with_mask(chunk_size)
        out, state = Op.apply(w, r, k, v, a, b, tau, mask, initial_state)
    else:
        Op = _make_sane_triton_op_no_mask(chunk_size)
        out, state = Op.apply(w, r, k, v, a, b, tau, initial_state)

    if not head_first:
        out = out.transpose(1, 2)

    if not output_final_state:
        return out

    if use_mask:
        return out, state

    warnings.warn(
        "[rwkv7_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
        "由于未提供 padding mask，返回的 final_state 可能被污染，"
        "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
        "[rwkv7_sane] mask is None: using unconditional State Anomaly Neutralization. "
        "The returned final_state is set to None because padding chunks "
        "may contaminate the state. Provide an explicit mask to obtain final_state.",
        UserWarning,
        stacklevel=2,
    )
    return out, None


def generalized_delta_rule_sane_inference(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """Triton 版本推理入口：直接复用训练 kernel，T 仍需被 chunk_size 整除。"""
    return generalized_delta_rule_sane(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        mask=mask,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
        chunk_size=chunk_size,
    )


def get_torch_generalized_delta_rule_sane(HEAD_SIZE=64, chunk_size: int = 16):
    """返回绑定 chunk_size 的 Torch-Triton 后端 (训练算子, 推理算子)。"""
    import functools

    return [
        functools.partial(generalized_delta_rule_sane, chunk_size=chunk_size),
        functools.partial(generalized_delta_rule_sane_inference, chunk_size=chunk_size),
    ]

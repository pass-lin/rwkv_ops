"""PyTorch 版 RWKV-7 State Anomaly Neutralization CUDA kernel 封装。"""

import os
import warnings
import torch
from torch.utils.cpp_extension import load
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.numpy import transpose


def transpose_head(x, head_first):
    if head_first:
        return transpose(x, (0, 2, 1, 3))
    return x


def get_torch_generalized_delta_rule_sane(HEAD_SIZE=64, chunk_size: int = 16):
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={chunk_size}",
        f"-DTORCH_LIBRARY_NAME=wind_backstepping_sane_{HEAD_SIZE}_{chunk_size}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_name = f"wind_backstepping_sane_{HEAD_SIZE}_{chunk_size}"
    load(
        name=lib_name,
        sources=[
            os.path.join(current_dir, "wkv7_sane_cuda.cu"),
            os.path.join(current_dir, "wkv7_sane_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cflags=[f"-DTORCH_LIBRARY_NAME={lib_name}"],
        extra_cuda_cflags=flags,
    )

    ops = getattr(torch.ops, lib_name)

    class WindBacksteppingSANE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()
            mask = cast(mask, "float32").contiguous()

            if T % chunk_size != 0:
                raise ValueError(
                    "RWKV-SANE inputs sequence length must be divisible by 16"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // chunk_size, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            ops.forward_sane(w, q, k, v, a, b, tau, mask, y, s, sa, h0)

            ctx.save_for_backward(w, q, k, v, a, b, tau, mask, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))

            # s[:, :, -1] 是 SANE 之前的 checkpoint，需要再应用一次 SANE 得到 final_state
            last_tau = tau[:, -1].view(B, H, 1, 1)
            last_mask = mask[:, -1].view(B, 1, 1, 1)
            tau_safe = torch.clamp(last_tau, min=1e-6)
            sane_state = last_tau * torch.tanh(last_state / tau_safe)
            last_state = torch.where(last_mask > 0, sane_state, last_state)

            return cast(y, DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            DTYPE = dy.dtype
            dy = cast(dy, torch.bfloat16).contiguous()
            dht = cast(dht, "float32").contiguous()
            w, q, k, v, a, b, tau, mask, s, sa = ctx.saved_tensors

            dh0 = torch.empty(dht.shape, dtype=dht.dtype, device=dht.device)
            dtau = torch.empty(tau.shape, dtype=tau.dtype, device=tau.device)
            dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]

            ops.backward_sane(
                w,
                q,
                k,
                v,
                a,
                b,
                tau,
                mask,
                dy,
                s,
                sa,
                dht,
                dh0,
                dtau,
                dw,
                dq,
                dk,
                dv,
                da,
                db,
            )
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(da, DTYPE),
                cast(db, DTYPE),
                dtau,
                None,  # mask has no gradient
                dh0,
            )

    class WindBacksteppingSANENoMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()

            if T % chunk_size != 0:
                raise ValueError(
                    "RWKV-SANE inputs sequence length must be divisible by 16"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // chunk_size, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            ops.forward_sane_no_mask(w, q, k, v, a, b, tau, y, s, sa, h0)

            ctx.save_for_backward(w, q, k, v, a, b, tau, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))

            # 无条件 SANE：直接应用
            last_tau = tau[:, -1].view(B, H, 1, 1)
            tau_safe = torch.clamp(last_tau, min=1e-6)
            last_state = last_tau * torch.tanh(last_state / tau_safe)

            return cast(y, DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            DTYPE = dy.dtype
            dy = cast(dy, torch.bfloat16).contiguous()
            dht = cast(dht, "float32").contiguous()
            w, q, k, v, a, b, tau, s, sa = ctx.saved_tensors

            dh0 = torch.empty(dht.shape, dtype=dht.dtype, device=dht.device)
            dtau = torch.empty(tau.shape, dtype=tau.dtype, device=tau.device)
            dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]

            ops.backward_sane_no_mask(
                w,
                q,
                k,
                v,
                a,
                b,
                tau,
                dy,
                s,
                sa,
                dht,
                dh0,
                dtau,
                dw,
                dq,
                dk,
                dv,
                da,
                db,
            )
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(da, DTYPE),
                cast(db, DTYPE),
                dtau,
                dh0,
            )

    # 纯推理 / prefill：不保存反向 checkpoint，只输出 y 与最终 state。
    # 因 tau/mask 按 chunk 读取，T 仍需被 16 整除；任意长度请用单步 RNN 接口。
    class Wkv7SaneInference(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()
            mask = cast(mask, "float32").contiguous()

            y = torch.empty_like(v)
            s = torch.empty(B, H, N, N, dtype=torch.float32, device=w.device)
            ops.forward_inference_sane(w, q, k, v, a, b, tau, mask, y, s, h0)
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError("inference kernel does not support backward")

    class Wkv7SaneInferenceNoMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()

            y = torch.empty_like(v)
            s = torch.empty(B, H, N, N, dtype=torch.float32, device=w.device)
            ops.forward_inference_sane_no_mask(w, q, k, v, a, b, tau, y, s, h0)
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError("inference kernel does not support backward")

    _compiled_chunk_size = chunk_size

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
        chunk_size: int = _compiled_chunk_size,
    ):
        """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（训练版）。

        非 CUDA 设备自动回退到 native 实现。
        当 mask=None 且 output_final_state=True 时，会发出 UserWarning 并将
        final_state 设为 None，避免 padding chunk 污染 state。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。T 必须被 16 整除。
            tau: [B, T//16, H], float32。阈值，必须严格 > 1。
            mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE。
            initial_state: [B, H, K, K], float32, 可选。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 或 mask=None 时不返回。

        Raises:
            ValueError: T 不被 16 整除，或 tau/mask 形状不匹配。
        """
        if chunk_size != _compiled_chunk_size:
            raise ValueError(
                f"CUDA kernel was compiled for chunk_size={_compiled_chunk_size}, "
                f"got {chunk_size}"
            )

        if w.device.type != "cuda":
            from ..native_keras_op import generalized_delta_rule_sane

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

        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)

        B, T, H, N = w.shape
        if T % chunk_size != 0:
            raise ValueError(
                f"RWKV-SANE training/prefill requires T divisible by {chunk_size}, "
                f"but got T={T}."
            )

        tau = cast(tau, "float32").contiguous()
        if tau.shape != (B, T // chunk_size, H):
            raise ValueError(
                f"tau shape {tuple(tau.shape)} does not match expected "
                f"(B={B}, T//16={T // chunk_size}, H={H})"
            )

        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        # 当且仅当需要 final_state 且显式提供 mask 时才使用带 mask 算子。
        use_mask = output_final_state and mask is not None

        if use_mask:
            mask = cast(mask, "float32").contiguous()
            if mask.shape != (B, T // chunk_size):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // chunk_size})"
                )
            out, state = WindBacksteppingSANE.apply(
                w, r, k, v, a, b, tau, mask, initial_state
            )
            return (out, state) if output_final_state else out

        out, state = WindBacksteppingSANENoMask.apply(
            w, r, k, v, a, b, tau, initial_state
        )
        if not output_final_state:
            return out

        # mask is None 且 output_final_state=True：警告并返回 None state。
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
        chunk_size: int = _compiled_chunk_size,
    ):
        """带 State Anomaly Neutralization 的 RWKV-7 推理入口（无梯度）。

        仅支持 CUDA；不保存反向 checkpoint，显存占用低于训练版。
        tau/mask 按 chunk 读取，T 不必被 16 整除。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。
            tau: [B, T//16, H], float32。
            mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE。
            initial_state: [B, H, K, K], float32, 可选。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 或 mask=None 时不返回。

        Raises:
            NotImplementedError: 非 CUDA 设备。
            ValueError: tau/mask 形状不匹配。
        """
        if chunk_size != _compiled_chunk_size:
            raise ValueError(
                f"CUDA kernel was compiled for chunk_size={_compiled_chunk_size}, "
                f"got {chunk_size}"
            )

        if w.device.type != "cuda":
            raise NotImplementedError("Inference kernel only supports CUDA")

        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)

        B, T, H, N = w.shape

        tau = cast(tau, "float32").contiguous()
        if tau.shape != (B, T // chunk_size, H):
            raise ValueError(
                f"tau shape {tuple(tau.shape)} does not match expected "
                f"(B={B}, T//16={T // chunk_size}, H={H})"
            )

        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        use_mask = output_final_state and mask is not None

        if use_mask:
            mask = cast(mask, "float32").contiguous()
            if mask.shape != (B, T // chunk_size):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // chunk_size})"
                )
            out, state = Wkv7SaneInference.apply(
                w, r, k, v, a, b, tau, mask, initial_state
            )
            return (out, state) if output_final_state else out

        out, state = Wkv7SaneInferenceNoMask.apply(w, r, k, v, a, b, tau, initial_state)
        if not output_final_state:
            return out

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

    return [generalized_delta_rule_sane, generalized_delta_rule_sane_inference]

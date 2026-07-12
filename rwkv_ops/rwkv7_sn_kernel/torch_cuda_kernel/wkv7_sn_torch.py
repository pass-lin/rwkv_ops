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


def get_torch_generalized_delta_rule_sn(HEAD_SIZE=64):
    CHUNK_LEN = 16
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={CHUNK_LEN}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]

    current_dir = os.path.dirname(os.path.abspath(__file__))
    load(
        name="wind_backstepping_sn",
        sources=[
            os.path.join(current_dir, "wkv7_sn_cuda.cu"),
            os.path.join(current_dir, "wkv7_sn_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    class WindBacksteppingSN(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()
            mask = cast(mask, "float32").contiguous()

            if T % CHUNK_LEN != 0:
                raise ValueError(
                    "RWKV-SN inputs sequence length must be divisible by 16"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            torch.ops.wind_backstepping_sn.forward_sn(
                w, q, k, v, a, b, tau, mask, y, s, sa, h0
            )

            ctx.save_for_backward(w, q, k, v, a, b, tau, mask, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))

            # s[:, :, -1] 是 SN 之前的 checkpoint，需要再应用一次 SN 得到 final_state
            last_tau = tau[:, -1].view(B, H, 1, 1)
            last_mask = mask[:, -1].view(B, 1, 1, 1)
            tau_safe = torch.clamp(last_tau, min=1e-6)
            sn_state = last_tau * torch.tanh(last_state / tau_safe)
            last_state = torch.where(last_mask > 0, sn_state, last_state)

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

            torch.ops.wind_backstepping_sn.backward_sn(
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

    class WindBacksteppingSNNoMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            tau = cast(tau, "float32").contiguous()

            if T % CHUNK_LEN != 0:
                raise ValueError(
                    "RWKV-SN inputs sequence length must be divisible by 16"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            torch.ops.wind_backstepping_sn.forward_sn_no_mask(
                w, q, k, v, a, b, tau, y, s, sa, h0
            )

            ctx.save_for_backward(w, q, k, v, a, b, tau, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))

            # 无条件 SN：直接应用
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

            torch.ops.wind_backstepping_sn.backward_sn_no_mask(
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
    class Wkv7SnInference(torch.autograd.Function):
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
            torch.ops.wind_backstepping_sn.forward_inference_sn(
                w, q, k, v, a, b, tau, mask, y, s, h0
            )
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError("inference kernel does not support backward")

    class Wkv7SnInferenceNoMask(torch.autograd.Function):
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
            torch.ops.wind_backstepping_sn.forward_inference_sn_no_mask(
                w, q, k, v, a, b, tau, y, s, h0
            )
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError("inference kernel does not support backward")

    def generalized_delta_rule_sn(
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
    ):
        if w.device.type != "cuda":
            from ..native_keras_op import generalized_delta_rule_sn

            return generalized_delta_rule_sn(
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
            )

        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)

        B, T, H, N = w.shape
        if T % CHUNK_LEN != 0:
            raise ValueError(
                f"RWKV-SN training/prefill requires T divisible by {CHUNK_LEN}, "
                f"but got T={T}."
            )

        tau = cast(tau, "float32").contiguous()
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tuple(tau.shape)} does not match expected "
                f"(B={B}, T//16={T // CHUNK_LEN}, H={H})"
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
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )
            out, state = WindBacksteppingSN.apply(
                w, r, k, v, a, b, tau, mask, initial_state
            )
            return (out, state) if output_final_state else out

        out, state = WindBacksteppingSNNoMask.apply(
            w, r, k, v, a, b, tau, initial_state
        )
        if not output_final_state:
            return out

        # mask is None 且 output_final_state=True：警告并返回 None state。
        warnings.warn(
            "[rwkv7_sn] mask is None: 使用无条件 State Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[rwkv7_sn] mask is None: using unconditional State Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    def generalized_delta_rule_sn_inference(
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
    ):
        """
        State Neutralization 推理 / prefill 入口（无梯度）。

        与训练版本数值等价，但显存占用更低。推理 kernel 按 chunk 读取 tau/mask，
        因此 `tau` 长度只需与 `T // 16` 一致，T 不需要被 16 整除。
        """
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
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tuple(tau.shape)} does not match expected "
                f"(B={B}, T//16={T // CHUNK_LEN}, H={H})"
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
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )
            out, state = Wkv7SnInference.apply(
                w, r, k, v, a, b, tau, mask, initial_state
            )
            return (out, state) if output_final_state else out

        out, state = Wkv7SnInferenceNoMask.apply(w, r, k, v, a, b, tau, initial_state)
        if not output_final_state:
            return out

        warnings.warn(
            "[rwkv7_sn] mask is None: 使用无条件 State Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[rwkv7_sn] mask is None: using unconditional State Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return [generalized_delta_rule_sn, generalized_delta_rule_sn_inference]

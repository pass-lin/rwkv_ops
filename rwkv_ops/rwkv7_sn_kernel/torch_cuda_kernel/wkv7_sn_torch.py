import os
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
        if mask is None:
            mask = torch.ones(B, T // CHUNK_LEN, dtype=torch.float32, device=w.device)
        else:
            mask = cast(mask, "float32").contiguous()
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )

        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        out, state = WindBacksteppingSN.apply(
            w, r, k, v, a, b, tau, mask, initial_state
        )
        return (out, state) if output_final_state else out

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
        if w.device.type != "cuda":
            raise NotImplementedError("Inference kernel only supports CUDA")

        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)

        B, T, H, N = w.shape
        if T % CHUNK_LEN != 0:
            raise ValueError(
                f"RWKV-SN inference/prefill requires T divisible by {CHUNK_LEN}, "
                f"but got T={T}."
            )

        tau = cast(tau, "float32").contiguous()
        if mask is None:
            mask = torch.ones(B, T // CHUNK_LEN, dtype=torch.float32, device=w.device)
        else:
            mask = cast(mask, "float32").contiguous()
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {tuple(mask.shape)} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )

        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        out, state = Wkv7SnInference.apply(w, r, k, v, a, b, tau, mask, initial_state)
        return (out, state) if output_final_state else out

    return [generalized_delta_rule_sn, generalized_delta_rule_sn_inference]

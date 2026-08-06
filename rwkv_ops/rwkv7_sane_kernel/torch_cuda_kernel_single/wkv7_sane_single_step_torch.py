import os
import torch
from torch.utils.cpp_extension import load


def get_torch_generalized_delta_rule_sane_single_step(HEAD_SIZE=64):
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        "-D_CHUNK_LEN_=1",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]
    current_dir = os.path.dirname(os.path.abspath(__file__))
    load(
        name="wind_backstepping_sane_single_step",
        sources=[
            os.path.join(current_dir, "wkv7_sane_single_step_cuda.cu"),
            os.path.join(current_dir, "wkv7_sane_single_step_op.cpp"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=flags,
    )

    class WindBacksteppingSANESingleStep(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, tau, do_sane, h0):
            DTYPE = q.dtype
            w = w.contiguous().bfloat16()
            q = q.contiguous().bfloat16()
            k = k.contiguous().bfloat16()
            v = v.contiguous().bfloat16()
            a = a.contiguous().bfloat16()
            b = b.contiguous().bfloat16()
            tau = tau.contiguous().float()
            do_sane = do_sane.contiguous().to(torch.int8)
            h0 = h0.contiguous().float()
            y = torch.empty_like(v)
            h1 = torch.empty_like(h0)
            torch.ops.wind_backstepping_sane_single_step.forward_single_step_sane(
                w, q, k, v, a, b, tau, do_sane, h0, y, h1
            )
            return y.to(DTYPE), h1

        @staticmethod
        def backward(ctx, *grads):
            raise NotImplementedError("single-step kernel does not support backward")

    def run_single_step(w, q, k, v, a, b, tau, do_sane, h0):
        return WindBacksteppingSANESingleStep.apply(w, q, k, v, a, b, tau, do_sane, h0)

    def generalized_delta_rule_sane_single_step(
        r,
        w,
        k,
        v,
        a,
        b,
        tau,
        do_sane,
        initial_state=None,
        output_final_state=True,
        head_first=False,
    ):
        if w.device.type != "cuda":
            from ..native_keras_op import generalized_delta_rule_sane_single_step

            return generalized_delta_rule_sane_single_step(
                r=r,
                w=w,
                k=k,
                v=v,
                a=a,
                b=b,
                tau=tau,
                do_sane=do_sane,
                initial_state=initial_state,
                output_final_state=output_final_state,
                head_first=head_first,
            )

        time_axis = 2 if head_first else 1
        if r.shape[time_axis] != 1:
            raise ValueError(
                f"Single-step kernel requires time dimension = 1, "
                f"but got shape {tuple(r.shape)}."
            )

        if head_first:
            r = r.squeeze(2)
            w = w.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)
            a = a.squeeze(2)
            b = b.squeeze(2)
        else:
            r = r.squeeze(1)
            w = w.squeeze(1)
            k = k.squeeze(1)
            v = v.squeeze(1)
            a = a.squeeze(1)
            b = b.squeeze(1)

        B, H, K = r.shape
        if initial_state is None:
            initial_state = torch.zeros(
                B, H, K, K, dtype=torch.float32, device=r.device
            )

        if isinstance(do_sane, bool):
            do_sane = torch.full((B,), int(do_sane), dtype=torch.int8, device=r.device)
        else:
            do_sane = do_sane.to(torch.int8)

        y, h1 = run_single_step(w, r, k, v, a, b, tau, do_sane, initial_state)
        y = y.unsqueeze(1)

        return (y, h1) if output_final_state else y

    return generalized_delta_rule_sane_single_step

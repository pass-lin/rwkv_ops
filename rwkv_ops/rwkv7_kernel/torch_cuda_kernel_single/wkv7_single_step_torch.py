"""PyTorch 版 RWKV-7 单步 CUDA kernel 封装。"""

import os
import torch
from torch.utils.cpp_extension import load


def get_torch_generalized_delta_rule_single_step(HEAD_SIZE=64, chunk_size: int = 16):
    # chunk_size 仅用于签名一致，单步 kernel 内部固定 CHUNK_LEN_=1，忽略该值。
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
        name="wind_backstepping_single_step",
        sources=[
            os.path.join(current_dir, "wkv7_single_step_cuda.cu"),
            os.path.join(current_dir, "wkv7_single_step_op.cpp"),
        ],
        is_python_module=False,
        verbose=False,
        extra_cuda_cflags=flags,
    )

    class WindBacksteppingSingleStep(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, h0):
            DTYPE = q.dtype
            w = w.contiguous().bfloat16()
            q = q.contiguous().bfloat16()
            k = k.contiguous().bfloat16()
            v = v.contiguous().bfloat16()
            a = a.contiguous().bfloat16()
            b = b.contiguous().bfloat16()
            h0 = h0.contiguous().float()
            y = torch.empty_like(v)
            h1 = torch.empty_like(h0)
            torch.ops.wind_backstepping_single_step.forward_single_step(
                w, q, k, v, a, b, h0, y, h1
            )
            return y.to(DTYPE), h1

        @staticmethod
        def backward(ctx, *grads):
            raise NotImplementedError("single-step kernel does not support backward")

    def run_single_step(w, q, k, v, a, b, h0):
        return WindBacksteppingSingleStep.apply(w, q, k, v, a, b, h0)

    def generalized_delta_rule(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        initial_state: torch.Tensor = None,
        output_final_state: bool = True,
        head_first: bool = False,
        chunk_size: int = 16,
    ):
        """RWKV-7 单步广义 delta 规则（仅前向）。

        仅支持 CUDA；非 CUDA 设备自动回退到 native 实现。

        Args:
            r, w, k, v, a, b: [B, 1, H, K] 或 [B, H, 1, K]，bfloat16。
            initial_state: [B, H, K, K], float32, 可选。None 则零初始化。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, 1, K])。

        Returns:
            out: [B, 1, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 时不返回。

        Raises:
            NotImplementedError: 单步 kernel 不支持反向。
        """
        if w.device.type != "cuda":
            from ..native_keras_op import generalized_delta_rule

            return generalized_delta_rule(
                r=r,
                k=k,
                v=v,
                a=a,
                b=b,
                w=w,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
            )
        # 统一转成 (B, H, K) 以调用单步 kernel
        if head_first:  # (B, H, 1, K) -> (B, H, K)
            r = r.squeeze(2)
            w = w.squeeze(2)
            k = k.squeeze(2)
            v = v.squeeze(2)
            a = a.squeeze(2)
            b = b.squeeze(2)
        else:  # (B, 1, H, K) -> (B, H, K)
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

        # 调用单步 kernel 并恢复 T 维度
        y, h1 = run_single_step(w, r, k, v, a, b, initial_state)  # y:(B,H,K)
        y = y.unsqueeze(1)  # (B, 1, H, K)

        return (y, h1) if output_final_state else y

    return generalized_delta_rule

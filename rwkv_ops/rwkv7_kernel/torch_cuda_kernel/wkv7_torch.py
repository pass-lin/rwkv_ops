"""PyTorch 版 RWKV-7 chunkwise CUDA kernel 封装。"""

import os
import torch
from torch.utils.cpp_extension import load
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.numpy import transpose


def transpose_head(x, head_first):
    if head_first:
        return transpose(x, (0, 2, 1, 3))
    else:
        return x


def get_torch_generalized_delta_rule(HEAD_SIZE=64, chunk_size: int = 16):
    flags = [
        "-res-usage",
        f"-D_C_={HEAD_SIZE}",
        f"-D_CHUNK_LEN_={chunk_size}",
        f"-DTORCH_LIBRARY_NAME=wind_backstepping_{HEAD_SIZE}_{chunk_size}",
        "--use_fast_math",
        "-O3",
        "-Xptxas -O3",
        "--extra-device-vectorization",
    ]

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    lib_name = f"wind_backstepping_{HEAD_SIZE}_{chunk_size}"

    load(
        name=lib_name,
        sources=[
            os.path.join(current_dir_path, "wkv7_cuda.cu"),
            os.path.join(current_dir_path, "wkv7_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cflags=[f"-DTORCH_LIBRARY_NAME={lib_name}"],
        extra_cuda_cflags=flags,
    )

    ops = getattr(torch.ops, lib_name)

    # 原版无 Mask Autograd Function（内部使用）
    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]

            if T % chunk_size != 0:
                raise ValueError(
                    f"RWKV inputs sequence length must be divisible by {chunk_size}"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // chunk_size, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            # 注意原版接口：第5个参数是z(对应a)，第6个是a(对应b)
            ops.forward(w, q, k, v, a, b, y, s, sa, h0)

            ctx.save_for_backward(w, q, k, v, a, b, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))
            return cast(y, DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            DTYPE = dy.dtype
            dy = cast(dy, torch.bfloat16).contiguous()
            dht = cast(dht, "float32").contiguous()
            w, q, k, v, a, b, s, sa = ctx.saved_tensors

            dh0 = torch.empty(dht.shape, dtype=dht.dtype, device=dht.device)
            dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]

            # 原版接口：第5个输出是dz(对应da)，第6个是da(对应db)
            ops.backward(w, q, k, v, a, b, dy, s, sa, dht, dh0, dw, dq, dk, dv, da, db)
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(da, DTYPE),
                cast(db, DTYPE),
                dh0,
            )

    # 带 Mask Autograd Function（内部使用）
    class WindBacksteppingWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype

            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            mask = cast(mask, "bfloat16").contiguous()

            if T % chunk_size != 0:
                raise ValueError(
                    f"RWKV inputs sequence length must be divisible by {chunk_size}"
                )

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // chunk_size, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            # Mask版本接口：参数直接对应 w,q,k,v,a,b,mask
            ops.forward_with_mask(w, q, k, v, a, b, mask, y, s, sa, h0)

            ctx.save_for_backward(w, q, k, v, a, b, mask, s, sa)

            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))
            return cast(y, DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            DTYPE = dy.dtype
            dy = cast(dy, torch.bfloat16).contiguous()
            dht = cast(dht, "float32").contiguous()

            w, q, k, v, a, b, mask, s, sa = ctx.saved_tensors

            dh0 = torch.empty(dht.shape, dtype=dht.dtype, device=dht.device)
            dw, dq, dk, dv, da, db = [torch.empty_like(x) for x in [w, q, k, v, a, b]]

            ops.backward_with_mask(
                w, q, k, v, a, b, mask, dy, s, sa, dht, dh0, dw, dq, dk, dv, da, db
            )
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(da, DTYPE),
                cast(db, DTYPE),
                None,
                dh0,  # mask的梯度为None
            )

    _compiled_chunk_size = chunk_size

    # 统一对外接口：Training（根据mask自动选择）
    def generalized_delta_rule(
        r,
        w,
        k,
        v,
        a,
        b,
        initial_state=None,
        output_final_state: bool = True,
        head_first: bool = False,
        chunk_size: int = _compiled_chunk_size,
        mask=None,
    ):
        """RWKV-7 chunkwise 广义 delta 规则（训练版）。

        非 CUDA 设备自动回退到 native 实现。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。T 必须被 chunk_size 整除。
            initial_state: [B, H, K, K], float32, 可选。None 则零初始化。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。
            chunk_size: int, chunk 长度，必须与编译时的 chunk_size 一致。
            mask: [B, T], float32 或 None。>0 更新状态、0 冻结状态。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 时不返回。

        Raises:
            ValueError: T 不被 chunk_size 整除，或 chunk_size 与编译值不一致。
        """
        if chunk_size != _compiled_chunk_size:
            raise ValueError(
                f"CUDA kernel was compiled for chunk_size={_compiled_chunk_size}, "
                f"got {chunk_size}"
            )

        # CPU回退
        if w.device.type != "cuda":
            from ..native_keras_op import generalized_delta_rule

            return generalized_delta_rule(
                r=r,
                w=w,
                k=k,
                v=v,
                a=a,
                b=b,
                mask=mask,
                initial_state=initial_state,
                output_final_state=output_final_state,
                chunk_size=chunk_size,
            )

        # 维度转置
        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)

        B, T, H, N = w.shape
        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        # 统一处理：将Python参数名映射到Kernel参数
        # r->q, w->w, k->k, v->v, a->a/b(视版本而定), b->b/a(视版本而定)
        if mask is None:
            # 无mask版本：参数映射为 w,q,k,v,a(z),b
            out, state = WindBackstepping.apply(w, r, k, v, a, b, initial_state)
        else:
            # 带mask版本：需要确保mask是float32且在cuda上
            if not mask.is_cuda:
                mask = mask.to(torch.bfloat16)
            out, state = WindBacksteppingWithMask.apply(
                w, r, k, v, a, b, mask, initial_state
            )

        return (out, state) if output_final_state else out

    # 原版推理（内部）
    class Wkv7Inference(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            y = torch.empty_like(v)
            s = torch.empty(B, H, N, N, dtype=torch.float32, device=w.device)
            ops.forward_inference(w, q, k, v, a, b, y, s, h0)
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError

    # 带Mask推理（内部）

    class Wkv7InferenceWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            mask = cast(mask, "bfloat16").contiguous()
            y = torch.empty_like(v)
            s = torch.empty(B, H, N, N, dtype=torch.float32, device=w.device)
            ops.forward_inference_with_mask(w, q, k, v, a, b, mask, y, s, h0)
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError

    def generalized_delta_rule_inference(
        r,
        w,
        k,
        v,
        a,
        b,
        initial_state=None,
        head_first: bool = False,
        output_final_state: bool = True,
        chunk_size: int = _compiled_chunk_size,
        mask=None,
    ):
        """RWKV-7 chunkwise 广义 delta 规则推理入口（无梯度）。

        仅支持 CUDA；不保存反向 checkpoint，因此显存低于训练版。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。
            initial_state: [B, H, K, K], float32, 可选。None 则零初始化。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。
            output_final_state: bool, 是否返回最终 state。
            chunk_size: int, chunk 长度，必须与编译时的 chunk_size 一致。
            mask: [B, T], float32 或 None。>0 更新状态、0 冻结状态。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 时不返回。

        Raises:
            NotImplementedError: 非 CUDA 设备。
            ValueError: chunk_size 与编译值不一致。
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
        if initial_state is None:
            initial_state = torch.zeros(
                B, H, N, N, dtype=torch.float32, device=r.device
            )
        else:
            initial_state = cast(initial_state, "float32")

        if mask is None:
            out, final_state = Wkv7Inference.apply(w, r, k, v, a, b, initial_state)
        else:
            if not mask.is_cuda:
                mask = mask.to(torch.bfloat16)
            out, final_state = Wkv7InferenceWithMask.apply(
                w, r, k, v, a, b, mask, initial_state
            )

        return (out, final_state) if output_final_state else out

    return [generalized_delta_rule, generalized_delta_rule_inference]

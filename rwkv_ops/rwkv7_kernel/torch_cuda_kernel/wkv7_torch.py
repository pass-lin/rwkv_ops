import os
import torch
from torch.utils.cpp_extension import load
from keras.src.backend.torch.core import cast
from keras.src.backend.torch.numpy import transpose, zeros


def transpose_head(x, head_first):
    if head_first:
        return transpose(x, (0, 2, 1, 3))
    else:
        return x


def get_torch_generalized_delta_rule(HEAD_SIZE=64):
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

    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    load(
        name="wind_backstepping",
        sources=[
            os.path.join(current_dir_path, "wkv7_cuda.cu"),
            os.path.join(current_dir_path, "wkv7_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=flags,
    )

    # ============================================================
    # 原版无 Mask Autograd Function（内部使用）
    # ============================================================
    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]

            if T % CHUNK_LEN != 0:
                raise ValueError("RWKV inputs sequence length must be divisible by 16")

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            # 注意原版接口：第5个参数是z(对应a)，第6个是a(对应b)
            torch.ops.wind_backstepping.forward(w, q, k, v, a, b, y, s, sa, h0)

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
            torch.ops.wind_backstepping.backward(
                w, q, k, v, a, b, dy, s, sa, dht, dh0, dw, dq, dk, dv, da, db
            )
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(da, DTYPE),
                cast(db, DTYPE),
                dh0,
            )

    # ============================================================
    # 带 Mask Autograd Function（内部使用）
    # ============================================================
    class WindBacksteppingWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype

            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            mask = cast(mask, "float32").contiguous()  # mask转fp32

            if T % CHUNK_LEN != 0:
                raise ValueError("RWKV inputs sequence length must be divisible by 16")

            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)

            # Mask版本接口：参数直接对应 w,q,k,v,a,b,mask
            torch.ops.wind_backstepping.forward_with_mask(
                w, q, k, v, a, b, mask, y, s, sa, h0
            )

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

            torch.ops.wind_backstepping.backward_with_mask(
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

    # ============================================================
    # 统一对外接口：Training（根据mask自动选择）
    # ============================================================
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
        mask=None,
    ):
        """
        统一的RWKV7 Delta Rule前向/反向接口
        Args:
            mask: None时使用无mask版本，否则使用mask版本（自动转换为fp32）
        """
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
            initial_state = zeros((B, H, N, N), "float32", device=w.device)
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
                mask = mask.to(torch.float32)
            out, state = WindBacksteppingWithMask.apply(
                w, r, k, v, a, b, mask, initial_state
            )

        return (out, state) if output_final_state else out

    # ============================================================
    # 原版推理（内部）
    # ============================================================
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
            torch.ops.wind_backstepping.forward_inference(w, q, k, v, a, b, y, s, h0)
            return cast(y, DTYPE), s

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError

    # ============================================================
    # 带Mask推理（内部）
    # ============================================================
    class Wkv7InferenceWithMask(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, a, b, mask, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q, k, v, a, b, w = [
                cast(x, "bfloat16").contiguous() for x in [q, k, v, a, b, w]
            ]
            mask = cast(mask, "float32").contiguous()
            y = torch.empty_like(v)
            s = torch.empty(B, H, N, N, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward_inference_with_mask(
                w, q, k, v, a, b, mask, y, s, h0
            )
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
        mask=None,
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
        if initial_state is None:
            initial_state = zeros((B, H, N, N), "float32", device=w.device)
        else:
            initial_state = cast(initial_state, "float32")

        if mask is None:
            out, final_state = Wkv7Inference.apply(w, r, k, v, a, b, initial_state)
        else:
            if not mask.is_cuda:
                mask = mask.to(torch.float32)
            out, final_state = Wkv7InferenceWithMask.apply(
                w, r, k, v, a, b, mask, initial_state
            )

        return (out, final_state) if output_final_state else out

    return [generalized_delta_rule, generalized_delta_rule_inference]

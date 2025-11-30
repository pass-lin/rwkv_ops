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
    # 获取当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取当前文件的目录路径
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

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b, h0):
            B, T, H, N = w.shape
            DTYPE = q.dtype
            q = cast(q, "bfloat16")
            k = cast(k, "bfloat16")
            v = cast(v, "bfloat16")
            z = cast(z, "bfloat16")
            b = cast(b, "bfloat16")
            w = cast(w, "bfloat16")
            if T % CHUNK_LEN != 0:
                raise ValueError(
                    "RWKV输入的序列长度必须可以被16整除"
                    "Please make sure the sequence length is divisible by 16"
                )
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, N, N, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, N, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa, h0)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            last_state = torch.empty_like(h0)
            last_state.copy_(transpose(s[:, :, -1], [0, 1, 3, 2]))

            return cast(y, DTYPE), last_state

        @staticmethod
        def backward(ctx, dy, dht):
            DTYPE = dy.dtype
            dy = cast(dy, torch.bfloat16)
            dy = dy.contiguous()

            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dht = cast(dht, "float32")
            dht = dht.contiguous()
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy, dht])
            dh0 = torch.empty(dht.shape, dtype=dht.dtype, device=dht.device)
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [w, q, k, v, z, b]]

            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dht, dh0, dw, dq, dk, dv, dz, db
            )
            return (
                cast(dw, DTYPE),
                cast(dq, DTYPE),
                cast(dk, DTYPE),
                cast(dv, DTYPE),
                cast(dz, DTYPE),
                cast(db, DTYPE),
                dh0,
            )

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b, h0):
        B, T, H, C = q.shape
        q = q.contiguous()
        w = w.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        a = a.contiguous()
        b = b.contiguous()
        out, state = WindBackstepping.apply(w, q, k, v, a, b, h0)
        return out, state

    def generalized_delta_rule(
        r: torch.Tensor,
        w: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        initial_state: torch.Tensor = None,
        output_final_state: bool = True,
        head_first: bool = False,
        use_chunk: bool = True,
    ):
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
            )
        r = transpose_head(r, head_first)
        k = transpose_head(k, head_first)
        v = transpose_head(v, head_first)
        a = transpose_head(a, head_first)
        b = transpose_head(b, head_first)
        w = transpose_head(w, head_first)
        B, T, H, N = w.shape
        if initial_state is None:
            initial_state = zeros((B, H, N, N), "float32")
        else:
            initial_state = cast(initial_state, "float32")
        out, state = RUN_CUDA_RWKV7g(r, w, k, v, a, b, initial_state)
        if output_final_state:
            return out, state
        return out

    return generalized_delta_rule

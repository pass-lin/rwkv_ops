import keras
from distutils.util import strtobool
import os
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def get_generalized_delta_rule(HEAD_SIZE=64, KERNEL_TYPE="native"):
    USE_KERNEL = False
    if keras.config.backend() == "torch":
        import torch

        if KERNEL_TYPE.lower() == "triton":
            from .torch_op import generalized_delta_rule

            USE_KERNEL = True

        elif KERNEL_TYPE.lower() == "cuda":
            CHUNK_LEN = 16
            USE_KERNEL = True
            from torch.utils.cpp_extension import load

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

            # 获取上一级目录的路径
            parent_dir_path = os.path.abspath(
                os.path.join(current_dir_path, os.path.pardir)
            )
            load(
                name="wind_backstepping",
                sources=[
                    os.path.join(parent_dir_path, "cuda_kernel/wkv7_cuda.cu"),
                    os.path.join(parent_dir_path, "cuda_kernel/wkv7_op.cpp"),
                ],
                is_python_module=False,
                verbose=True,
                extra_cuda_cflags=flags,
            )

            class WindBackstepping(torch.autograd.Function):
                @staticmethod
                def forward(ctx, w, q, k, v, z, b):
                    B, T, H, C = w.shape
                    DTYPE = q.dtype
                    q = ops.cast(q, "bfloat16")
                    k = ops.cast(k, "bfloat16")
                    v = ops.cast(v, "bfloat16")
                    z = ops.cast(z, "bfloat16")
                    b = ops.cast(b, "bfloat16")
                    w = ops.cast(w, "bfloat16")
                    assert T % CHUNK_LEN == 0
                    assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
                    y = torch.empty_like(v)
                    s = torch.empty(
                        B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
                    )
                    sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
                    torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
                    ctx.save_for_backward(w, q, k, v, z, b, s, sa)
                    return ops.cast(y, DTYPE)

                @staticmethod
                def backward(ctx, dy):
                    DTYPE = dy.dtype
                    dy = ops.cast(dy, torch.bfloat16)
                    dy = dy.contiguous()
                    assert all(i.dtype == torch.bfloat16 for i in [dy])
                    assert all(i.is_contiguous() for i in [dy])
                    w, q, k, v, z, b, s, sa = ctx.saved_tensors
                    dw, dq, dk, dv, dz, db = [
                        torch.empty_like(x) for x in [w, q, k, v, z, b]
                    ]
                    torch.ops.wind_backstepping.backward(
                        w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
                    )
                    return (
                        ops.cast(dw, DTYPE),
                        ops.cast(dq, DTYPE),
                        ops.cast(dk, DTYPE),
                        ops.cast(dv, DTYPE),
                        ops.cast(dz, DTYPE),
                        ops.cast(db, DTYPE),
                    )

            def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
                B, T, H, C = q.shape
                q = q.contiguous()
                w = w.contiguous()
                k = k.contiguous()
                v = v.contiguous()
                a = a.contiguous()
                b = b.contiguous()
                return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, H * C)

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
                r = transpose_head(r, head_first)
                k = transpose_head(k, head_first)
                v = transpose_head(v, head_first)
                a = transpose_head(a, head_first)
                b = transpose_head(b, head_first)
                w = transpose_head(w, head_first)
                return RUN_CUDA_RWKV7g(r, w, k, v, a, b), None
        else:
            from .native_keras_op import generalized_delta_rule

            USE_KERNEL = False
    elif keras.config.backend() == "jax":
        from jax.lib import xla_bridge
        import jax
        import os
        import logging

        logging.basicConfig(level=logging.ERROR)
        os.environ["TRITON_LOG_LEVEL"] = "ERROR"  # 只显示错误级别的日志
        os.environ["TRITON_DISABLE_AUTOTUNE"] = "1"  # 禁用自动调优日志
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 禁用自动调优日志
        if (
            xla_bridge.get_backend().platform == "gpu"
            and KERNEL_TYPE.lower() == "triton"
        ):
            os.environ["JAX_LOG_COMPUTATION"] = "0"
            from .jax_op import generalized_delta_rule

            USE_KERNEL = True
        else:
            from .native_keras_op import generalized_delta_rule

    else:
        from .native_keras_op import generalized_delta_rule
    return generalized_delta_rule, USE_KERNEL

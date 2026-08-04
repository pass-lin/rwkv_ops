import os
import warnings
import torch
from torch.utils.cpp_extension import load


def get_torch_rwkv6(head_size: int = 64, max_sequence_length: int = 4096):
    """
    构建并返回 RWKV-6 Torch CUDA 函数式算子。
    """
    current_file_path = os.path.abspath(__file__)
    current_dir_path = os.path.dirname(current_file_path)

    extra_cuda_cflags = [
        "--use_fast_math",
        f"-D_N_={head_size}",
        f"-D_T_={max_sequence_length}",
    ]

    load(
        name="wkv6",
        sources=[
            os.path.join(current_dir_path, "wkv6_cuda.cu"),
            os.path.join(current_dir_path, "wkv6_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cuda_cflags=extra_cuda_cflags,
    )

    # ============================================================
    # 训练用 Autograd Function（无状态）
    # ============================================================
    class RWKV6Function(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            assert r.dtype == k.dtype == v.dtype == w.dtype == u.dtype
            assert r.dtype in [torch.float32, torch.bfloat16, torch.float16]
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()

            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            ctx.dtype = r.dtype
            ctx.save_for_backward(r, k, v, w, u)

            y_dtype = r.dtype if r.dtype != torch.float16 else torch.float32
            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )

            if r.dtype == torch.float32:
                torch.ops.wkv6.forward_fp32(B, T, C, H, r, k, v, w, u, y)
            elif r.dtype == torch.bfloat16:
                torch.ops.wkv6.forward_bf16(B, T, C, H, r, k, v, w, u, y)
            else:
                torch.ops.wkv6.forward_fp16(B, T, C, H, r, k, v, w, u, y)

            return y

        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            r, k, v, w, u = ctx.saved_tensors

            y_dtype = ctx.dtype if ctx.dtype != torch.float16 else torch.float32
            gr = torch.empty(
                (B, T, C),
                device=gy.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )
            gk = torch.empty(
                (B, T, C),
                device=gy.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )
            gv = torch.empty(
                (B, T, C),
                device=gy.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )
            gw = torch.empty(
                (B, T, C),
                device=gy.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )
            gu = torch.empty(
                (B, C),
                device=gy.device,
                dtype=y_dtype,
                memory_format=torch.contiguous_format,
            )

            if ctx.dtype == torch.float32:
                torch.ops.wkv6.backward_fp32(
                    B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu
                )
            elif ctx.dtype == torch.bfloat16:
                torch.ops.wkv6.backward_bf16(
                    B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu
                )
            else:
                torch.ops.wkv6.backward_fp16(
                    B, T, C, H, r, k, v, w, u, gy, gr, gk, gv, gw, gu
                )

            # gu 原始 shape 为 (B, C)，需要按 batch 求和后 reshape 为与输入 u 一致
            gu = torch.sum(gu, dim=0).view(u.shape)

            return (None, None, None, None, gr, gk, gv, gw, gu)

    # ============================================================
    # 带初始状态/最终状态的前向（仅前向，无梯度）
    # ============================================================
    class RWKV6ForwardWithState(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, is_custom_state, state_map, r, k, v, w, u, s):
            assert r.dtype == k.dtype == v.dtype == w.dtype == u.dtype
            assert r.dtype in [torch.float16, torch.float32, torch.bfloat16]

            if r.dtype in [torch.float32, torch.bfloat16]:
                o_dtype = r.dtype
            else:
                o_dtype = torch.float32

            y = torch.empty(
                (B, T, C),
                device=r.device,
                dtype=o_dtype,
                memory_format=torch.contiguous_format,
            )
            ys = torch.empty(
                (B, H, head_size, head_size),
                device=r.device,
                dtype=o_dtype,
                memory_format=torch.contiguous_format,
            )

            if r.dtype == torch.bfloat16:
                torch.ops.wkv6.forward_with_state_bf16(
                    B, T, C, H, is_custom_state, state_map, r, k, v, w, u, s, y, ys
                )
            elif r.dtype == torch.float32:
                torch.ops.wkv6.forward_with_state_fp32(
                    B, T, C, H, is_custom_state, state_map, r, k, v, w, u, s, y, ys
                )
            else:
                torch.ops.wkv6.forward_with_state_fp16(
                    B, T, C, H, is_custom_state, state_map, r, k, v, w, u, s, y, ys
                )

            return y, ys

        @staticmethod
        def backward(ctx, *args):
            raise NotImplementedError(
                "RWKV6 forward_with_state does not support backward"
            )

    # ============================================================
    # 对外函数式接口
    # ============================================================
    def rwkv6(
        r,
        k,
        v,
        w,
        u,
        initial_state=None,
        output_final_state: bool = False,
        state_map=None,
    ):
        if r.device.type != "cuda":
            from ..native_keras_op import rwkv6 as native_rwkv6

            return native_rwkv6(
                r=r,
                k=k,
                v=v,
                w=w,
                u=u,
                initial_state=initial_state,
                output_final_state=output_final_state,
                state_map=state_map,
                head_size=head_size,
            )

        B, T, C = r.shape
        if T > max_sequence_length:
            raise ValueError(
                f"序列长度 T={T} 超过编译期最大序列长度 max_sequence_length={max_sequence_length}"
            )
        assert C % head_size == 0
        H = C // head_size

        if not isinstance(u, torch.Tensor):
            u = u.value

        assert r.is_cuda and k.is_cuda and v.is_cuda and w.is_cuda and u.is_cuda
        assert r.device == k.device == v.device == w.device == u.device
        assert r.dtype == k.dtype == v.dtype == w.dtype == u.dtype
        assert r.is_contiguous()
        assert k.is_contiguous()
        assert v.is_contiguous()
        assert w.is_contiguous()
        assert u.is_contiguous()

        if r.dtype != torch.bfloat16:
            warnings.warn(
                f"RWKV-6 Torch CUDA kernel expects bfloat16 inputs, got {r.dtype}. "
                "Casting to bfloat16. This may introduce precision differences.",
                stacklevel=2,
            )
            r = r.bfloat16()
            k = k.bfloat16()
            v = v.bfloat16()
            w = w.bfloat16()
            u = u.bfloat16()

        s_dtype = torch.bfloat16

        if output_final_state or initial_state is not None:
            is_custom_state = initial_state is not None
            if initial_state is None:
                s = torch.zeros((0,), device=r.device, dtype=s_dtype)
                state_map_t = torch.zeros((0,), device=r.device, dtype=torch.int64)
            else:
                assert len(initial_state.shape) in [3, 4]
                if len(initial_state.shape) == 3:
                    initial_state = initial_state[None, :]
                assert initial_state.shape[1:] == (H, head_size, head_size)
                initial_state = initial_state.to(s_dtype)
                assert initial_state.device == r.device

                n_state = initial_state.shape[0]
                if state_map is None:
                    assert n_state == 1 or n_state == B
                    if n_state == 1:
                        state_map_t = torch.zeros(
                            (B,), dtype=torch.int64, device=r.device
                        )
                    else:
                        state_map_t = torch.tensor(
                            [i for i in range(B)], dtype=torch.int64, device=r.device
                        )
                else:
                    if isinstance(state_map, list):
                        state_map_t = torch.tensor(
                            state_map, dtype=torch.int64, device=r.device
                        )
                    elif isinstance(state_map, torch.Tensor):
                        assert state_map.dtype in [torch.int32, torch.int64]
                        state_map_t = state_map.to(torch.int64).to(r.device)
                    else:
                        raise ValueError("state_map must be list or torch.Tensor")
                    assert state_map_t.shape == (B,)
                    assert (state_map_t >= 0).all() and (state_map_t < n_state).all()

                s = initial_state

            y, ys = RWKV6ForwardWithState.apply(
                B, T, C, H, is_custom_state, state_map_t, r, k, v, w, u, s
            )
            return (y, ys) if output_final_state else y
        else:
            y = RWKV6Function.apply(B, T, C, H, r, k, v, w, u)
            return y

    return rwkv6

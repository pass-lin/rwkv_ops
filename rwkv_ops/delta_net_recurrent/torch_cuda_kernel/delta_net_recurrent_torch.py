"""PyTorch 版 DeltaNet recurrent CUDA kernel 封装。"""

import os

import torch
from torch.utils.cpp_extension import load

# 按 (K, V, chunk_size) 缓存已编译的 CUDA 库，首次调用时懒编译。
_LOADED_OPS = {}

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_ops(K, V, chunk_size):
    """按 (K, V, chunk_size) 懒编译并返回 CUDA 算子命名空间。"""
    key = (K, V, chunk_size)
    if key in _LOADED_OPS:
        return _LOADED_OPS[key]

    lib_name = f"delta_net_recurrent_cuda_{K}_{V}_{chunk_size}"
    flags = [
        "-O3",
        "-Xptxas",
        "-O3",
        "-res-usage",
        "--extra-device-vectorization",
        f"-D_K_={K}",
        f"-D_V_={V}",
        f"-D_CHUNK_LEN_={chunk_size}",
        f"-DTORCH_LIBRARY_NAME={lib_name}",
    ]
    load(
        name=lib_name,
        sources=[
            os.path.join(_CURRENT_DIR, "delta_net_recurrent_cuda.cu"),
            os.path.join(_CURRENT_DIR, "delta_net_recurrent_op.cpp"),
        ],
        is_python_module=False,
        verbose=True,
        extra_cflags=[f"-DTORCH_LIBRARY_NAME={lib_name}"],
        extra_cuda_cflags=flags,
    )
    ops = getattr(torch.ops, lib_name)
    _LOADED_OPS[key] = ops
    return ops


def _normalize_inputs(q, k, v, beta, head_first):
    """把输入统一转成 [B, H, T, *] 并保证连续，k/v 对齐 q 的 dtype。"""
    if not head_first:
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        beta = beta.transpose(1, 2)
    if q.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError(
            f"DeltaNet recurrent CUDA kernel only supports bfloat16 "
            f"or float32 inputs, got {q.dtype}"
        )
    q, k, v = [x.to(q.dtype).contiguous() for x in [q, k, v]]
    beta = beta.to(torch.float32).contiguous()
    return q, k, v, beta


def _prepare_initial_state(initial_state, B, H, K, V, device):
    """准备 float32 连续初始 state，支持 [1, H, K, V] 广播。"""
    if initial_state is None:
        return torch.zeros(B, H, K, V, dtype=torch.float32, device=device)
    h0 = initial_state.to(torch.float32).contiguous()
    if h0.shape[0] == 1 and B > 1:
        h0 = h0.expand(B, *h0.shape[1:]).contiguous()
    return h0


class _DeltaNetRecurrentCudaFunction(torch.autograd.Function):
    """DeltaNet recurrent 训练 CUDA 封装。"""

    @staticmethod
    def forward(
        ctx, q, k, v, beta, initial_state, output_final_state, head_first, chunk_size
    ):
        input_dtype = q.dtype
        q, k, v, beta = _normalize_inputs(q, k, v, beta, head_first)

        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        if T % chunk_size != 0:
            raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")

        ops = _load_ops(K, V, chunk_size)

        o = torch.empty_like(v)
        kv_mem = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
        state_chkp = torch.empty(
            B, H, T // chunk_size, K, V, dtype=torch.float32, device=q.device
        )
        inv_norm_q = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        inv_norm_k = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
        h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

        ops.forward(
            q,
            k,
            v,
            beta,
            h0,
            scale,
            o,
            kv_mem,
            state_chkp,
            inv_norm_q,
            inv_norm_k,
            final_state,
        )

        ctx.save_for_backward(
            q, k, v, beta, h0, kv_mem, state_chkp, inv_norm_q, inv_norm_k
        )
        ctx.input_dtype = input_dtype
        ctx.head_first = head_first
        ctx.output_final_state = output_final_state
        ctx.chunk_size = chunk_size

        o = o.to(input_dtype)
        if not head_first:
            o = o.transpose(1, 2)

        if output_final_state:
            return o, final_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        q, k, v, beta, h0, kv_mem, state_chkp, inv_norm_q, inv_norm_k = (
            ctx.saved_tensors
        )

        if not ctx.head_first:
            do = do.transpose(1, 2)
        do = do.to(q.dtype).contiguous()

        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        ops = _load_ops(K, V, ctx.chunk_size)

        dq = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        dk = torch.empty(B, H, T, K, dtype=torch.float32, device=q.device)
        dv = torch.empty(B, H, T, V, dtype=torch.float32, device=q.device)
        dbeta = torch.empty(B, H, T, dtype=torch.float32, device=q.device)
        dh0 = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)

        if dht is not None:
            dht = dht.to(torch.float32).contiguous()
        else:
            dht = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)

        ops.backward(
            q,
            k,
            v,
            beta,
            do,
            dht,
            kv_mem,
            inv_norm_q,
            inv_norm_k,
            h0,
            state_chkp,
            scale,
            dq,
            dk,
            dv,
            dbeta,
            dh0,
        )

        input_dtype = ctx.input_dtype
        if not ctx.head_first:
            dq = dq.transpose(1, 2)
            dk = dk.transpose(1, 2)
            dv = dv.transpose(1, 2)
            dbeta = dbeta.transpose(1, 2)

        dq = dq.to(input_dtype)
        dk = dk.to(input_dtype)
        dv = dv.to(input_dtype)
        dbeta = dbeta.to(input_dtype)

        return dq, dk, dv, dbeta, dh0, None, None, None


class _DeltaNetRecurrentInferenceCudaFunction(torch.autograd.Function):
    """DeltaNet recurrent 推理 CUDA 封装（无反向）。"""

    @staticmethod
    def forward(
        ctx, q, k, v, beta, initial_state, output_final_state, head_first, chunk_size
    ):
        input_dtype = q.dtype
        q, k, v, beta = _normalize_inputs(q, k, v, beta, head_first)

        B, H, T, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        ops = _load_ops(K, V, chunk_size)

        o = torch.empty_like(v)
        final_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
        h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

        ops.forward_inference(q, k, v, beta, h0, scale, o, final_state)

        o = o.to(input_dtype)
        if not head_first:
            o = o.transpose(1, 2)

        if output_final_state:
            return o, final_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "DeltaNet recurrent inference does not support backward."
        )


class _DeltaNetRecurrentSingleStepCudaFunction(torch.autograd.Function):
    """DeltaNet recurrent 单步 RNN CUDA 封装（无反向）。"""

    @staticmethod
    def forward(
        ctx, q, k, v, beta, initial_state, output_final_state, head_first, chunk_size
    ):
        if not head_first:
            raise NotImplementedError(
                "delta_net_recurrent_single_step currently only "
                "supports head_first=True."
            )

        input_dtype = q.dtype
        if q.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError(
                f"DeltaNet recurrent CUDA kernel only supports bfloat16 "
                f"or float32 inputs, got {q.dtype}"
            )
        q, k, v = [x.to(q.dtype).contiguous() for x in [q, k, v]]
        beta = beta.to(torch.float32).contiguous()

        B, H, K = q.shape
        V = v.shape[-1]
        scale = K**-0.5

        ops = _load_ops(K, V, chunk_size)

        o = torch.empty_like(v)
        next_state = torch.empty(B, H, K, V, dtype=torch.float32, device=q.device)
        h0 = _prepare_initial_state(initial_state, B, H, K, V, q.device)

        ops.single_step(q, k, v, beta, h0, scale, o, next_state)

        o = o.to(input_dtype)
        if output_final_state:
            return o, next_state
        return o, None

    @staticmethod
    def backward(ctx, do, dht):
        raise NotImplementedError(
            "DeltaNet recurrent single step does not support backward."
        )


def delta_net_recurrent(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    chunk_size=16,
):
    """DeltaNet recurrent 训练算子（PyTorch CUDA 实现）。

    非 CUDA 设备自动回退到 native 实现。

    Args:
        q, k: [B, T, H, K]，bfloat16 或 float32。
        v: [B, T, H, V]，bfloat16 或 float32。
        beta: [B, T, H]，float32，写入强度，必须已在外部过 sigmoid 并落在
            (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。作为编译期常量按
            (K, V, chunk_size) 懒编译。

    Returns:
        out: [B, T, H, V]，与输入同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True
            时返回。

    Raises:
        TypeError: 输入 dtype 不是 bfloat16 或 float32。
        ValueError: T 不被 chunk_size 整除。
    """
    if q.device.type != "cuda":
        from ..native_keras_op import delta_net_recurrent as native_op

        return native_op(
            q,
            k,
            v,
            beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            chunk_size=chunk_size,
        )

    return _DeltaNetRecurrentCudaFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        head_first,
        chunk_size,
    )


def delta_net_recurrent_inference(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """DeltaNet recurrent 推理算子（PyTorch CUDA 实现，无梯度）。

    不保存反向 checkpoint，显存低于训练版。

    Args:
        q, k: [B, T, H, K]，bfloat16 或 float32。
        v: [B, T, H, V]，bfloat16 或 float32。
        beta: [B, T, H]，float32，写入强度，必须已在外部过 sigmoid 并落在
            (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。作为编译期常量按
            (K, V, chunk_size) 懒编译，推理 kernel 不依赖其数值。

    Returns:
        out: [B, T, H, V]，与输入同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True
            时返回。

    Raises:
        TypeError: 输入 dtype 不是 bfloat16 或 float32。
        NotImplementedError: 输入不在 CUDA 设备上。
    """
    if q.device.type != "cuda":
        raise NotImplementedError(
            "DeltaNet recurrent CUDA inference kernel only supports CUDA devices."
        )

    return _DeltaNetRecurrentInferenceCudaFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        head_first,
        chunk_size,
    )


def delta_net_recurrent_single_step(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=True,
    head_first=True,
    chunk_size=16,
):
    """DeltaNet recurrent 单步 RNN 算子（PyTorch CUDA 实现）。

    Args:
        q, k: [B, H, K]，bfloat16 或 float32。
        v: [B, H, V]，bfloat16 或 float32。
        beta: [B, H]，float32，写入强度，必须已在外部过 sigmoid 并落在
            (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回下一步 state。
        head_first: bool，输入输出是否 head 维优先。单步默认
            True（[B, H, *]）。
        chunk_size: int，chunk 长度，默认 16。仅用于编译缓存键，单步
            kernel 不依赖其数值。

    Returns:
        out: [B, H, V]，与输入同 dtype。
        next_state: [B, H, K, V]，float32；仅当 output_final_state=True
            时返回。

    Raises:
        TypeError: 输入 dtype 不是 bfloat16 或 float32。
        NotImplementedError: 输入不在 CUDA 设备上，或 head_first=False。
    """
    if q.device.type != "cuda":
        raise NotImplementedError(
            "DeltaNet recurrent CUDA single step kernel only supports CUDA devices."
        )

    return _DeltaNetRecurrentSingleStepCudaFunction.apply(
        q,
        k,
        v,
        beta,
        initial_state,
        output_final_state,
        head_first,
        chunk_size,
    )

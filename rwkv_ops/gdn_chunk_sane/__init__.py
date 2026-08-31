"""Gated DeltaNet chunkwise SANE 算子后端分发器。"""

import functools

import keras

from ..utils import _use_triton


def get_gated_delta_net_chunk_sane(KERNEL_TYPE="native", chunk_size=16):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet chunkwise SANE 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "triton"。
            当前 "triton" 在 torch CUDA / JAX GPU 后端提供加速实现，其余静默回退 native。
        chunk_size: int，chunk 长度，默认 16。返回的算子签名中已包含该参数。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_chunk_sane` 一致。
    """
    from .native_keras_op import gated_delta_net_chunk_sane

    backend = keras.config.backend()

    if backend == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import gated_delta_net_chunk_sane as triton_op

            return functools.partial(triton_op, chunk_size=chunk_size)

    if backend == "jax" and KERNEL_TYPE == "triton":
        if _use_triton(KERNEL_TYPE):
            from .jax_triton_kernel import gated_delta_net_chunk_sane as triton_op

            return functools.partial(triton_op, chunk_size=chunk_size)

    return functools.partial(gated_delta_net_chunk_sane, chunk_size=chunk_size)

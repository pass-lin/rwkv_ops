"""DeltaNet recurrent 算子后端分发器。"""

import functools

import keras

from ..utils import _use_triton


def _use_jax_triton(KERNEL_TYPE):
    """jax + GPU 平台且 KERNEL_TYPE=triton 时启用 Triton kernel。"""
    if not _use_triton(KERNEL_TYPE):
        return False
    try:
        import jax
        import jax_triton  # noqa: F401
        import triton  # noqa: F401
    except Exception:
        return False
    return jax.devices()[0].platform == "gpu"


def get_delta_net_recurrent(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            "triton" 在 torch CUDA / jax GPU 后端提供加速实现
            （jax 侧需安装 jax-triton），其余静默回退 native。
        chunk_size: int，chunk 长度，默认 16。纯 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent` 一致。
    """
    from .native_keras_op import delta_net_recurrent

    if keras.config.backend() == "torch":
        if KERNEL_TYPE in ("native", "triton") and _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import delta_net_recurrent as triton_op

            return functools.partial(triton_op, chunk_size=chunk_size)
    elif keras.config.backend() == "jax":
        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import delta_net_recurrent as triton_op

            return functools.partial(triton_op, chunk_size=chunk_size)

    return functools.partial(delta_net_recurrent, chunk_size=chunk_size)


def get_delta_net_recurrent_inference(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 推理算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            "triton" 在 torch CUDA / jax GPU 后端提供加速实现
            （jax 侧需安装 jax-triton），其余静默回退 native。
        chunk_size: int，chunk 长度，默认 16。纯 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent_inference` 一致。
    """
    from .native_keras_op import delta_net_recurrent_inference

    if keras.config.backend() == "torch":
        if KERNEL_TYPE in ("native", "triton") and _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                delta_net_recurrent_inference as triton_op,
            )

            return functools.partial(triton_op, chunk_size=chunk_size)
    elif keras.config.backend() == "jax":
        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import (
                delta_net_recurrent_inference as triton_op,
            )

            return functools.partial(triton_op, chunk_size=chunk_size)

    return functools.partial(delta_net_recurrent_inference, chunk_size=chunk_size)


def get_delta_net_recurrent_single_step(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 单步 RNN 算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            "triton" 在 torch CUDA / jax GPU 后端提供加速实现
            （jax 侧需安装 jax-triton），其余静默回退 native。
        chunk_size: int，chunk 长度，默认 16。单步 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent_single_step` 一致。
    """
    from .native_keras_op import delta_net_recurrent_single_step

    if keras.config.backend() == "torch":
        if KERNEL_TYPE in ("native", "triton") and _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                delta_net_recurrent_single_step as triton_op,
            )

            return functools.partial(triton_op, chunk_size=chunk_size)
    elif keras.config.backend() == "jax":
        if KERNEL_TYPE == "triton" and _use_jax_triton(KERNEL_TYPE):
            from .jax_triton_kernel import (
                delta_net_recurrent_single_step as triton_op,
            )

            return functools.partial(triton_op, chunk_size=chunk_size)

    return functools.partial(delta_net_recurrent_single_step, chunk_size=chunk_size)

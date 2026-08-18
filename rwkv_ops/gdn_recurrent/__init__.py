"""Gated DeltaNet recurrent 算子后端分发器。"""

import os

import keras


def _force_keras_native():
    """RWKV_OPS_KERAS_NATIVE=1 时强制 native 使用纯 Keras ops。"""
    v = os.environ.get("RWKV_OPS_KERAS_NATIVE", "").lower()
    return v not in ("", "0", "false")


def _use_triton(KERNEL_TYPE):
    """torch + 非 CPU 平台且 KERNEL_TYPE=native/triton 时启用 Triton kernel。"""
    if KERNEL_TYPE not in ("native", "triton") or _force_keras_native():
        return False
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    return torch.cuda.is_available()


def get_gated_delta_net_recurrent(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅 "triton" 在 torch CUDA 后端提供加速实现，其余静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import gated_delta_net_recurrent as triton_op

            return triton_op

    return gated_delta_net_recurrent


def get_gated_delta_net_recurrent_inference(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent 推理算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅 "triton" 在 torch CUDA 后端提供加速实现，其余静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent_inference` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent_inference

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                gated_delta_net_recurrent_inference as triton_op,
            )

            return triton_op

    return gated_delta_net_recurrent_inference


def get_gated_delta_net_recurrent_single_step(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet recurrent 单步 RNN 算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅 "triton" 在 torch CUDA 后端提供加速实现，其余静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_recurrent_single_step` 一致。
    """
    from .native_keras_op import gated_delta_net_recurrent_single_step

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import (
                gated_delta_net_recurrent_single_step as triton_op,
            )

            return triton_op

    return gated_delta_net_recurrent_single_step

"""RWKV-7 算子后端分发器。"""

import os

import keras
from keras import ops


def transpose_head(x, head_first):
    """在 [B, T, H, K] 与 [B, H, T, K] 两种 layout 间切换。

    Args:
        x: [B, T, H, K] 或 [B, H, T, K]，任意常见 dtype。
        head_first: bool，为 True 时执行 transpose(0, 2, 1, 3)，否则原样返回。

    Returns:
        head_first=True 时返回 [B, H, T, K]，否则返回原张量。
    """
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def _force_keras_native():
    """RWKV_OPS_KERAS_NATIVE=1 时强制 jax/torch native 使用纯 Keras ops。"""
    v = os.environ.get("RWKV_OPS_KERAS_NATIVE", "").lower()
    return v not in ("", "0", "false")


def _use_pallas(KERNEL_TYPE):
    """jax + GPU/TPU 且 KERNEL_TYPE=native 时启用 Pallas kernel。"""
    if KERNEL_TYPE != "native" or _force_keras_native():
        return False
    import jax

    return jax.devices()[0].platform in ("gpu", "tpu")


def _use_triton(KERNEL_TYPE):
    """torch + 非 CPU 平台且 KERNEL_TYPE=native 时启用 Triton kernel。"""
    if KERNEL_TYPE != "native" or _force_keras_native():
        return False
    try:
        import torch
        import triton  # noqa: F401
    except Exception:
        return False
    if torch.cuda.is_available():
        return True
    xpu = getattr(torch, "xpu", None)
    if xpu is not None:
        try:
            return bool(xpu.is_available())
        except Exception:
            return False
    return False


def get_generalized_delta_rule(HEAD_SIZE=64, KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 RWKV-7 chunkwise 训练算子对。

    Args:
        HEAD_SIZE: int，head 维度大小，必须被 4 整除。
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。

    Returns:
        (training_op, inference_op): 均为 Callable。
        当后端/硬件不支持所选 KERNEL_TYPE 时静默回退 native_keras_op。
    """
    assert HEAD_SIZE % 4 == 0
    from .native_keras_op import generalized_delta_rule

    if keras.config.backend() == "jax":
        import jax

        platform = jax.devices()[0].platform
        if platform == "gpu":
            if KERNEL_TYPE == "cuda":
                from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule

                return get_jax_generalized_delta_rule(HEAD_SIZE)
            elif KERNEL_TYPE == "triton":
                from .jax_triton_kernel import generalized_delta_rule as jax_kernel

                return jax_kernel, generalized_delta_rule
        if _use_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import get_jax_generalized_delta_rule

            return get_jax_generalized_delta_rule(HEAD_SIZE)
    elif keras.config.backend() == "torch":
        import torch

        if torch.cuda.is_available():
            if KERNEL_TYPE == "cuda":
                from .torch_cuda_kernel.wkv7_torch import (
                    get_torch_generalized_delta_rule,
                )

                return get_torch_generalized_delta_rule(HEAD_SIZE)
            elif KERNEL_TYPE == "triton":
                from .torch_triton_kernel import generalized_delta_rule as triton_kernel

                return triton_kernel, generalized_delta_rule
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import generalized_delta_rule as triton_kernel

            return triton_kernel, generalized_delta_rule

    return generalized_delta_rule, generalized_delta_rule


def get_rnn_generalized_delta_rule(HEAD_SIZE=64, KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 RWKV-7 单步（T=1）算子。

    Args:
        HEAD_SIZE: int，head 维度大小，必须被 4 整除。
        KERNEL_TYPE: str，目前仅 "cuda" 提供加速实现，其余回退 native。

    Returns:
        single_step_op: Callable，输入 T 必须为 1。
    """
    assert HEAD_SIZE % 4 == 0
    from .native_keras_op import generalized_delta_rule

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_cuda_kernel_single.wkv7_single_step_jax import (
                    get_jax_generalized_delta_rule_single_step,
                )

                return get_jax_generalized_delta_rule_single_step(HEAD_SIZE)
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_cuda_kernel_single.wkv7_single_step_torch import (
                    get_torch_generalized_delta_rule_single_step,
                )

                return get_torch_generalized_delta_rule_single_step(HEAD_SIZE)
    return generalized_delta_rule

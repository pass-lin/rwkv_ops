"""RWKV-7 算子后端分发器。"""

import os

import keras
from keras import ops
from ..utils import _use_triton


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


def get_generalized_delta_rule(
    HEAD_SIZE=64, KERNEL_TYPE="native", chunk_size: int = 16
):
    """按后端与 KERNEL_TYPE 返回 RWKV-7 chunkwise 训练算子对。

    Args:
        HEAD_SIZE: int，head 维度大小，必须被 4 整除。
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
        chunk_size: int，chunk 长度，必须整除序列长度。

    Returns:
        (training_op, inference_op): 均为 Callable。
        当后端/硬件不支持所选 KERNEL_TYPE 时静默回退 native_keras_op。
    """
    assert HEAD_SIZE % 4 == 0
    assert chunk_size > 0
    from .native_keras_op import generalized_delta_rule

    if keras.config.backend() == "jax":
        import jax
        from ..pallas_utils import _use_jax_pallas

        platform = jax.devices()[0].platform
        if platform == "gpu":
            if KERNEL_TYPE == "cuda":
                from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule

                return get_jax_generalized_delta_rule(HEAD_SIZE, chunk_size=chunk_size)
            elif KERNEL_TYPE == "triton":
                from .jax_triton_kernel import get_jax_generalized_delta_rule

                return get_jax_generalized_delta_rule(HEAD_SIZE, chunk_size=chunk_size)

        if _use_jax_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import get_jax_generalized_delta_rule

            return get_jax_generalized_delta_rule(HEAD_SIZE, chunk_size=chunk_size)
    elif keras.config.backend() == "torch":
        import torch

        if torch.cuda.is_available():
            if KERNEL_TYPE == "cuda":
                from .torch_cuda_kernel.wkv7_torch import (
                    get_torch_generalized_delta_rule,
                )

                return get_torch_generalized_delta_rule(
                    HEAD_SIZE, chunk_size=chunk_size
                )
            elif KERNEL_TYPE == "triton":
                from .torch_triton_kernel import get_torch_generalized_delta_rule

                return get_torch_generalized_delta_rule(
                    HEAD_SIZE, chunk_size=chunk_size
                )
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import get_torch_generalized_delta_rule

            return get_torch_generalized_delta_rule(HEAD_SIZE, chunk_size=chunk_size)

    return generalized_delta_rule, generalized_delta_rule


def get_rnn_generalized_delta_rule(
    HEAD_SIZE=64, KERNEL_TYPE="native", chunk_size: int = 16
):
    """按后端与 KERNEL_TYPE 返回 RWKV-7 单步（T=1）算子。

    Args:
        HEAD_SIZE: int，head 维度大小，必须被 4 整除。
        KERNEL_TYPE: str，目前仅 "cuda" 提供加速实现，其余回退 native。
        chunk_size: int，chunk 长度，单步 kernel 忽略该值。

    Returns:
        single_step_op: Callable，输入 T 必须为 1。
    """
    assert HEAD_SIZE % 4 == 0
    assert chunk_size > 0
    from .native_keras_op import generalized_delta_rule

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_cuda_kernel_single.wkv7_single_step_jax import (
                    get_jax_generalized_delta_rule_single_step,
                )

                return get_jax_generalized_delta_rule_single_step(
                    HEAD_SIZE, chunk_size=chunk_size
                )
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_cuda_kernel_single.wkv7_single_step_torch import (
                    get_torch_generalized_delta_rule_single_step,
                )

                return get_torch_generalized_delta_rule_single_step(
                    HEAD_SIZE, chunk_size=chunk_size
                )
    return generalized_delta_rule

"""RWKV-7-SN 算子工厂（按后端与 KERNEL_TYPE 分发）。"""

import keras
from keras import ops

from ..rwkv7_kernel import _use_pallas, _use_triton


def transpose_head(x, head_first):
    """统一输入布局为 [B, H, T, K]。"""
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    return x


def get_generalized_delta_rule_sn(HEAD_SIZE=64, KERNEL_TYPE="cuda"):
    """返回 RWKV-7-SN chunkwise 训练算子与推理算子。

    Args:
        HEAD_SIZE: int，head 维度大小，必须为 4 的倍数。
        KERNEL_TYPE: str，可选 "cuda" / "triton" / "native" / "pallas"（JAX）。

    Returns:
        (train_op, inference_op)：均为函数。
    """
    assert HEAD_SIZE % 4 == 0
    from .native_keras_op import generalized_delta_rule_sn

    if keras.config.backend() == "jax":
        import jax

        platform = jax.devices()[0].platform
        if platform == "gpu":
            if KERNEL_TYPE == "cuda":
                from .jax_cuda_kernel.wkv7_sn_jax import (
                    get_jax_generalized_delta_rule_sn,
                )

                return get_jax_generalized_delta_rule_sn(HEAD_SIZE)
            elif KERNEL_TYPE == "triton":
                from .jax_triton_kernel import get_jax_generalized_delta_rule_sn

                return get_jax_generalized_delta_rule_sn(HEAD_SIZE)
        if _use_pallas(KERNEL_TYPE):
            from .jax_pallas_kernel import get_jax_generalized_delta_rule_sn

            return get_jax_generalized_delta_rule_sn(HEAD_SIZE)
    elif keras.config.backend() == "torch":
        import torch

        if torch.cuda.is_available():
            if KERNEL_TYPE == "cuda":
                from .torch_cuda_kernel.wkv7_sn_torch import (
                    get_torch_generalized_delta_rule_sn,
                )

                return get_torch_generalized_delta_rule_sn(HEAD_SIZE)
            elif KERNEL_TYPE == "triton":
                from .torch_triton_kernel import get_torch_generalized_delta_rule_sn

                return get_torch_generalized_delta_rule_sn(HEAD_SIZE)
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import get_torch_generalized_delta_rule_sn

            return get_torch_generalized_delta_rule_sn(HEAD_SIZE)

    return generalized_delta_rule_sn, generalized_delta_rule_sn


def get_rnn_generalized_delta_rule_sn(HEAD_SIZE=64, KERNEL_TYPE="cuda"):
    """返回 RWKV-7-SN 单步（T=1）算子。

    Args:
        HEAD_SIZE: int，head 维度大小，必须为 4 的倍数。
        KERNEL_TYPE: str，仅 "cuda" 启用 FFI/C++ 扩展，其余回退 native。

    Returns:
        single_step_op：函数。
    """
    assert HEAD_SIZE % 4 == 0
    from .native_keras_op import generalized_delta_rule_sn_single_step

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_cuda_kernel_single.wkv7_sn_single_step_jax import (
                    get_jax_generalized_delta_rule_sn_single_step,
                )

                return get_jax_generalized_delta_rule_sn_single_step(HEAD_SIZE)
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_cuda_kernel_single.wkv7_sn_single_step_torch import (
                    get_torch_generalized_delta_rule_sn_single_step,
                )

                return get_torch_generalized_delta_rule_sn_single_step(HEAD_SIZE)
    return generalized_delta_rule_sn_single_step

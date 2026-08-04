import os

import keras
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def _force_keras_native():
    """RWKV_OPS_KERAS_NATIVE=1 时，jax/torch 的 native 都强制为纯 keras ops。

    用于无 kernel 的调试/数值对照。未设置或设为 0/false 时不强制。
    """
    v = os.environ.get("RWKV_OPS_KERAS_NATIVE", "").lower()
    return v not in ("", "0", "false")


def _use_pallas(KERNEL_TYPE):
    """jax + 非 CPU 平台时，native 默认使用 pallas kernel。

    可用 RWKV_OPS_KERAS_NATIVE=1 强制回退纯 keras ops（调试用）。
    """
    if KERNEL_TYPE != "native" or _force_keras_native():
        return False
    import jax

    return jax.devices()[0].platform in ("gpu", "tpu")


def _use_triton(KERNEL_TYPE):
    """torch + 非 CPU 平台（CUDA/ROCm/XPU）时，native 默认使用 triton kernel。

    新版 torch 与 triton 强耦合（pip 版自带 triton），triton 之于 torch
    相当于 pallas 之于 jax。XPU/ROCm 未实测，规则上按"非 CPU 且 triton
    可导入"开放。可用 RWKV_OPS_KERAS_NATIVE=1 强制回退纯 keras ops。
    """
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

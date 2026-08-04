import os

import keras
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def _use_pallas(KERNEL_TYPE):
    """jax + 非 CPU 平台时，native 默认使用 pallas kernel。

    可用 RWKV_OPS_JAX_NATIVE=xla 强制回退纯 keras ops（调试用）。
    """
    if KERNEL_TYPE != "native":
        return False
    if os.environ.get("RWKV_OPS_JAX_NATIVE", "").lower() == "xla":
        return False
    import jax

    return jax.devices()[0].platform in ("gpu", "tpu")


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

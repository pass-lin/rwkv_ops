import keras
from distutils.util import strtobool
import os
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def get_generalized_delta_rule(HEAD_SIZE=64, KERNEL_TYPE="native"):
    assert HEAD_SIZE % 4 == 0
    from .native_keras_op import generalized_delta_rule

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule

                return get_jax_generalized_delta_rule(HEAD_SIZE)
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_cuda_kernel.wkv7_torch import (
                    get_torch_generalized_delta_rule,
                )

                return get_torch_generalized_delta_rule(HEAD_SIZE)
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

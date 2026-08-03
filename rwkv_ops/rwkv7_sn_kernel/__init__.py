import keras
from keras import ops


def transpose_head(x, head_first):
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    return x


def get_generalized_delta_rule_sn(HEAD_SIZE=64, KERNEL_TYPE="cuda"):
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
        if platform in ("gpu", "tpu") and KERNEL_TYPE in ("native", "pallas"):
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

    return generalized_delta_rule_sn, generalized_delta_rule_sn


def get_rnn_generalized_delta_rule_sn(HEAD_SIZE=64, KERNEL_TYPE="cuda"):
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

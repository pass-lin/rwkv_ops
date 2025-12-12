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
    from .native_keras_op import generalized_delta_rule as native_op

    if keras.config.backend() == "torch":
        import torch

        if torch.cuda.is_available():
            if KERNEL_TYPE.lower() == "triton":
                from .torch_op import generalized_delta_rule

                return generalized_delta_rule, generalized_delta_rule, True
            elif KERNEL_TYPE.lower() == "cuda":
                from .torch_cuda_kernel.wkv7_torch import (
                    get_torch_generalized_delta_rule,
                )

                return get_torch_generalized_delta_rule(HEAD_SIZE) + [False]
    elif keras.config.backend() == "jax":
        import jax
        import os

        if jax.devices()[0].platform == "gpu":
            if KERNEL_TYPE.lower() == "triton":
                os.environ["JAX_LOG_COMPUTATION"] = "0"
                from .jax_op import generalized_delta_rule

                return generalized_delta_rule, native_op, False
            elif KERNEL_TYPE.lower() == "cuda":
                from .jax_cuda_kernel.wkv7_jax import get_jax_generalized_delta_rule

                return get_jax_generalized_delta_rule(HEAD_SIZE) + [False]
    elif keras.config.backend() == "tensorflow":
        import tensorflow as tf

        if len(tf.config.list_physical_devices("GPU")) > 0:
            if KERNEL_TYPE.lower() == "cuda" and HEAD_SIZE:
                try:
                    from jax.lib import xla_bridge

                    assert xla_bridge.get_backend().platform == "gpu"
                except:
                    raise (
                        "The operation of the TensorFlow kernel depends on the JAX kernel."
                        "Therefore, it is necessary to ensure that it can be used in JAX, so that TensorFlow can be used."
                    )
                print("🎉" * 10)
                print("Tensorflow CUDA kernel onlt support Forward,not get graident")
                print("🎉" * 10)
                from .tf_eager_kernel import get_tf_generalized_delta_rule

                generalized_delta_rule_inference = get_tf_generalized_delta_rule(
                    HEAD_SIZE
                )
                return native_op, generalized_delta_rule_inference, False
    elif keras.config.backend() == "mlx" and KERNEL_TYPE.lower() == "cuda":
        from .mlx_op import generalized_delta_rule

        return native_op, generalized_delta_rule, False
    return native_op, native_op, False


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
        elif keras.config.backend() == "tensorflow":
            import tensorflow as tf

            if len(tf.config.list_physical_devices("GPU")) > 0:
                try:
                    import jax
                except ImportError:
                    return generalized_delta_rule
                if jax.devices()[0].platform == "gpu":
                    from .tf_eager_kernel import (
                        get_tf_generalized_delta_rule_single_step,
                    )

                    return get_tf_generalized_delta_rule_single_step(HEAD_SIZE)
    return generalized_delta_rule

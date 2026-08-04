"""RWKV-6 算子工厂，按后端与 KERNEL_TYPE 分发实现。"""

import keras


def get_rwkv6_kernel(HEAD_SIZE=64, KERNEL_TYPE="native", MAX_SEQUENCE_LENGTH=4096):
    """返回当前后端适用的 RWKV-6 函数式算子。

    根据当前 Keras backend、KERNEL_TYPE 与可用硬件，返回 CUDA（JAX/Torch）
    或原生 Keras-ops 实现的 rwkv6_op。RWKV-6 的 KERNEL_TYPE="triton" 会静默
    回退到 native。

    Args:
        HEAD_SIZE: int，默认 64。head 维度，需被 4 整除。
        KERNEL_TYPE: "native" | "cuda" | "triton"。
        MAX_SEQUENCE_LENGTH: int，默认 4096。CUDA kernel 编译期最大序列长度。

    Returns:
        rwkv6_op: 函数式算子，签名见 native_keras_op.rwkv6 的 docstring。

    Raises:
        AssertionError: HEAD_SIZE 不能被 4 整除，或 KERNEL_TYPE 不支持。
    """
    assert HEAD_SIZE % 4 == 0, f"HEAD_SIZE={HEAD_SIZE} 必须被 4 整除"
    assert KERNEL_TYPE in ["native", "cuda", "triton"], (
        f"不支持的 KERNEL_TYPE={KERNEL_TYPE}"
    )

    from .native_keras_op import rwkv6 as native_rwkv6

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_cuda_kernel.wkv6_jax import get_jax_rwkv6

                return get_jax_rwkv6(
                    head_size=HEAD_SIZE,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                )
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_cuda_kernel.wkv6_torch import get_torch_rwkv6

                return get_torch_rwkv6(
                    head_size=HEAD_SIZE,
                    max_sequence_length=MAX_SEQUENCE_LENGTH,
                )

    from functools import partial

    # native / 未检测到 GPU / 其它 backend 均回退到原生实现。
    return partial(
        native_rwkv6,
        head_size=HEAD_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

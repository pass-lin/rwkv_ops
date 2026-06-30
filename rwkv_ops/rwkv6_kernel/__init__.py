import keras


def get_rwkv6_kernel(HEAD_SIZE=64, KERNEL_TYPE="native", MAX_SEQUENCE_LENGTH=4096):
    """
    获取 RWKV-6 函数式算子。

    根据当前 Keras backend、环境变量 KERNEL_TYPE 以及可用硬件，
    返回 CUDA（JAX/Torch）或原生 Keras-ops 实现的 rwkv6_op。

    Args:
        HEAD_SIZE: head 维度，默认 64，需被 4 整除。
        KERNEL_TYPE: "native" | "cuda" | "triton"（RWKV-6 当前仅支持 native/cuda）。
        MAX_SEQUENCE_LENGTH: CUDA  kernel 编译期最大序列长度，默认 4096。

    Returns:
        rwkv6_op(r, k, v, w, u, initial_state=None, output_final_state=False,
                 state_map=None, head_first=False)
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

    # native / 未检测到 GPU / 其它 backend 均回退到原生实现
    from functools import partial

    return partial(
        native_rwkv6,
        head_size=HEAD_SIZE,
        max_sequence_length=MAX_SEQUENCE_LENGTH,
    )

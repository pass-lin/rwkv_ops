import keras


def get_mhc_kernel(KERNEL_TYPE="native"):
    from .native_op import mhc_pre_op, mhc_post_op

    if KERNEL_TYPE == "triton":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_triton_op.mhc_post_op import mhc_post_op
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_triton_op.mhc_post_op import mhc_post_op

    return mhc_pre_op, mhc_post_op

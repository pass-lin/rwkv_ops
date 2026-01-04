import keras


def get_mhu_kernel(KERNEL_TYPE):
    from .native_keras_op import (
        sinkhorn_knopp,
        rmsnorm,
        stream_aggregate,
        stream_distribute,
        stream_mix,
    )

    if KERNEL_TYPE == "cuda":
        if keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_kernel.mhc_torch import (
                    sinkhorn_knopp,
                    rmsnorm,
                    stream_aggregate,
                    stream_distribute,
                    stream_mix,
                )

    return sinkhorn_knopp, rmsnorm, stream_aggregate, stream_distribute, stream_mix

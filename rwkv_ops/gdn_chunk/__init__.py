"""Gated DeltaNet chunkwise 算子后端分发器。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import keras

from ..utils import _use_triton


def get_gated_delta_net_chunk(KERNEL_TYPE="native"):
    """按后端与 KERNEL_TYPE 返回 Gated DeltaNet chunkwise 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "triton"。
            当前 "triton" 在 torch CUDA 后端提供加速实现，其余静默回退 native。

    Returns:
        Callable，签名与 `native_keras_op.gated_delta_net_chunk` 一致。
    """
    from .native_keras_op import gated_delta_net_chunk

    if keras.config.backend() == "torch" and KERNEL_TYPE in ("native", "triton"):
        if _use_triton(KERNEL_TYPE):
            from .torch_triton_kernel import gated_delta_net_chunk as triton_op

            return triton_op

    return gated_delta_net_chunk

"""DeltaNet chunkwise SANE 算子后端分发器。"""

import functools


def get_delta_net_chunk_sane(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet chunkwise SANE 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "triton"。当前仅 native 实现，
            其余取值静默回退 native。
        chunk_size: int，chunk 长度，默认 16。返回的算子签名中已包含该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_chunk_sane` 一致。
    """
    from .native_keras_op import delta_net_chunk_sane

    return functools.partial(delta_net_chunk_sane, chunk_size=chunk_size)

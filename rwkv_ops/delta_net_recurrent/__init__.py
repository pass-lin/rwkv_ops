"""DeltaNet recurrent 算子后端分发器。"""

import functools


def get_delta_net_recurrent(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 训练算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅提供 native 实现，其余取值静默回退 native。
        chunk_size: int，chunk 长度，默认 16。纯 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent` 一致。
    """
    from .native_keras_op import delta_net_recurrent

    return functools.partial(delta_net_recurrent, chunk_size=chunk_size)


def get_delta_net_recurrent_inference(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 推理算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅提供 native 实现，其余取值静默回退 native。
        chunk_size: int，chunk 长度，默认 16。纯 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent_inference` 一致。
    """
    from .native_keras_op import delta_net_recurrent_inference

    return functools.partial(delta_net_recurrent_inference, chunk_size=chunk_size)


def get_delta_net_recurrent_single_step(KERNEL_TYPE="native", chunk_size: int = 16):
    """按后端与 KERNEL_TYPE 返回 DeltaNet recurrent 单步 RNN 算子。

    Args:
        KERNEL_TYPE: str，"native" / "cuda" / "triton"。
            当前仅提供 native 实现，其余取值静默回退 native。
        chunk_size: int，chunk 长度，默认 16。单步 recurrent 实现忽略该参数。

    Returns:
        Callable，签名与 `native_keras_op.delta_net_recurrent_single_step` 一致。
    """
    from .native_keras_op import delta_net_recurrent_single_step

    return functools.partial(delta_net_recurrent_single_step, chunk_size=chunk_size)

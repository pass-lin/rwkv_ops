"""Gated DeltaNet chunkwise Triton 公共工具。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import triton.language as tl


try:
    import torch

    IS_TF32_SUPPORTED = (
        torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    )
except Exception:
    IS_TF32_SUPPORTED = False

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


def chunk_local_cumsum_native(g, chunk_size=64, reverse=False):
    """在 chunk 内部做 cumsum 或反向 cumsum（纯 Keras 参考实现）。

    Args:
        g: [B, H, T]，log-space decay 或其梯度。
        chunk_size: int，chunk 长度。
        reverse: bool，是否做反向 cumsum。

    Returns:
        [B, H, T]，每个 chunk 内独立 cumsum 后的结果。
    """
    import keras

    B, H, T = g.shape
    assert T % chunk_size == 0, f"T={T} 必须被 chunk_size={chunk_size} 整除"
    g = g.reshape(B, H, T // chunk_size, chunk_size)
    if reverse:
        g = keras.ops.flip(g, axis=-1)
        g = keras.ops.cumsum(g, axis=-1)
        g = keras.ops.flip(g, axis=-1)
    else:
        g = keras.ops.cumsum(g, axis=-1)
    return g.reshape(B, H, T)

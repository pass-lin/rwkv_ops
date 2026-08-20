"""Gated DeltaNet chunkwise Triton 公共工具。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""


def chunk_local_cumsum(g, chunk_size=64, reverse=False):
    """在 chunk 内部做 cumsum 或反向 cumsum。

    Args:
        g: [B, H, T]，log-space decay 或其梯度。
        chunk_size: int，chunk 长度。
        reverse: bool，是否做反向 cumsum。

    Returns:
        [B, H, T]，每个 chunk 内独立 cumsum 后的结果。
    """
    B, H, T = g.shape
    assert T % chunk_size == 0, f"T={T} 必须被 chunk_size={chunk_size} 整除"
    g = g.reshape(B, H, T // chunk_size, chunk_size)
    if reverse:
        g = g.flip(-1).cumsum(dim=-1).flip(-1)
    else:
        g = g.cumsum(dim=-1)
    return g.reshape(B, H, T)

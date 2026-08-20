"""Gated DeltaNet chunkwise Triton 公共工具。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""


def chunk_local_cumsum(g, chunk_size=64):
    """在 chunk 内部做 cumsum。

    Args:
        g: [B, H, T]，log-space decay。
        chunk_size: int，chunk 长度。

    Returns:
        [B, H, T]，每个 chunk 内独立 cumsum 后的 decay。
    """
    B, H, T = g.shape
    assert T % chunk_size == 0, f"T={T} 必须被 chunk_size={chunk_size} 整除"
    g = g.reshape(B, H, T // chunk_size, chunk_size)
    g = g.cumsum(dim=-1)
    return g.reshape(B, H, T)

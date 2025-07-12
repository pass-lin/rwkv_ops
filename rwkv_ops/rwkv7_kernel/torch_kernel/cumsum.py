from ..triton_kernel.cumsum import *
import torch


def chunk_rwkv6_fwd_cumsum(
    g: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    B, T, H, S = g.shape
    BT = chunk_size
    NT = triton.cdiv(T, BT)

    gi, ge = (
        torch.empty_like(g, dtype=torch.float),
        torch.empty_like(g, dtype=torch.float),
    )

    def grid(meta):
        return (triton.cdiv(meta["S"], meta["BS"]), NT, B * H)

    # keep cummulative normalizer in fp32
    chunk_rwkv6_fwd_cumsum_kernel[grid](
        g,
        T,
        gi,
        ge,
        H=H,
        S=S,
        BT=BT,
    )
    return gi, ge

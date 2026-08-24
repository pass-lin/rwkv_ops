"""chunkwise 局部 cumsum Triton kernel。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
    ],
    key=["B", "H", "C"],
)
@triton.jit
def _chunk_local_cumsum_kernel(
    s_ptr,
    o_ptr,
    B,
    H,
    T,
    C: tl.constexpr,
    REVERSE: tl.constexpr,
):
    """chunk 内局部 cumsum。

    输入输出 layout 均为 [B, H, T]，每个 program 处理一个 (batch, head, chunk)。

    Args:
        s_ptr: [B, H, T]，输入标量序列。
        o_ptr: [B, H, T]，输出 cumsum 结果。
        B, H, T: 维度。
        C: chunk 长度，编译期常量。
        REVERSE: 是否反向 cumsum。
    """
    i_bh = tl.program_id(0).to(tl.int64)
    i_c = tl.program_id(1).to(tl.int64)

    b = i_bh // H
    h = i_bh % H

    o_c = tl.arange(0, C)
    o_t = i_c * C + o_c
    m_t = o_t < T

    base = (b * H + h) * T
    p_s = s_ptr + base + o_t
    p_o = o_ptr + base + o_t

    b_s = tl.load(p_s, mask=m_t, other=0.0).to(tl.float32)
    if REVERSE:
        b_o = tl.cumsum(b_s, axis=0, reverse=True)
    else:
        b_o = tl.cumsum(b_s, axis=0)
    tl.store(p_o, b_o.to(o_ptr.dtype.element_ty), mask=m_t)


def chunk_local_cumsum_torch(s, chunk_size=64, reverse=False):
    """PyTorch 封装：在 [B, H, T] 上计算 chunk 内局部 cumsum。

    Args:
        s: [B, H, T]，输入。
        chunk_size: int，chunk 长度，必须是 2 的幂。
        reverse: bool，是否反向 cumsum。

    Returns:
        [B, H, T]，与输入同 layout 的 cumsum 结果，dtype 为 float32。
    """
    B, H, T = s.shape
    if T % chunk_size != 0:
        raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")

    out = torch.empty_like(s, dtype=torch.float32)
    grid = (B * H, T // chunk_size)
    _chunk_local_cumsum_kernel[grid](
        s,
        out,
        B,
        H,
        T,
        C=chunk_size,
        REVERSE=reverse,
    )
    return out

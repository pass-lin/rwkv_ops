"""Gated DeltaNet chunkwise L2 归一化 Triton kernel。

代码参考自 https://github.com/fla-org/flash-linear-attention
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [32, 64, 128]
        for num_warps in [2, 4]
    ],
    key=["K"],
)
@triton.jit
def _gdn_chunk_l2norm_fwd_kernel(
    x_ptr,
    out_ptr,
    inv_norm_ptr,
    T,
    K,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """L2 归一化前向 kernel。

    把前导维度 flatten 成 T，每个 program 处理 BT 个 (b,h,t) 位置，沿 K 维归一化。

    Args:
        x_ptr: [T, K]，输入指针。
        out_ptr: [T, K]，输出指针。
        inv_norm_ptr: [T]，保存的逆范数 `1 / sqrt(sum(x^2) + eps)`。
        T: 前导维度乘积 B*H*T_seq。
        K: 特征维度。
        BK: K 维 block 大小。
        BT: 时间/block 大小。
    """
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_k[None, :] < K)

    p_x = x_ptr + o_t[:, None] * K + o_k[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)

    inv_norm = tl.rsqrt(tl.sum(b_x * b_x, 1) + 1e-6)
    b_out = b_x * inv_norm[:, None]

    tl.store(
        out_ptr + o_t[:, None] * K + o_k[None, :],
        b_out.to(out_ptr.dtype.element_ty),
        mask=m_x,
    )
    tl.store(inv_norm_ptr + o_t, inv_norm.to(inv_norm_ptr.dtype.element_ty), mask=m_t)


def gdn_chunk_l2norm_fwd(x):
    """L2 归一化前向封装。

    Args:
        x: [B, H, T, K]，torch.Tensor。

    Returns:
        out: [B, H, T, K]，归一化后的张量。
        inv_norm: [B, H, T]，逆范数。
    """
    B, H, T, K = x.shape
    x_2d = x.reshape(B * H * T, K)
    out_2d = torch.empty_like(x_2d)
    inv_norm = torch.empty(B * H * T, dtype=torch.float32, device=x.device)

    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(B * H * T, meta["BT"]),)

    _gdn_chunk_l2norm_fwd_kernel[grid](
        x_2d,
        out_2d,
        inv_norm,
        B * H * T,
        K,
        BK=BK,
    )
    out = out_2d.view(B, H, T, K)
    inv_norm = inv_norm.view(B, H, T)
    return out, inv_norm


@triton.autotune(
    configs=[
        triton.Config({"BT": BT}, num_warps=num_warps)
        for BT in [32, 64, 128]
        for num_warps in [2, 4]
    ],
    key=["K"],
)
@triton.jit
def _gdn_chunk_l2norm_bwd_kernel(
    x_ptr,
    inv_norm_ptr,
    dout_ptr,
    dx_ptr,
    T,
    K,
    BK: tl.constexpr,
    BT: tl.constexpr,
):
    """L2 归一化反向 kernel。

    dx = (dout - x * sum(dout * x) * inv_norm^2) * inv_norm

    Args:
        x_ptr: [T, K]，前向原始输入。
        inv_norm_ptr: [T]，前向保存的逆范数。
        dout_ptr: [T, K]，输出梯度。
        dx_ptr: [T, K]，输入梯度输出。
        T, K: 维度。
        BK, BT: block 大小。
    """
    i_t = tl.program_id(0).to(tl.int64)
    o_t = i_t * BT + tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    m_t = o_t < T
    m_x = m_t[:, None] & (o_k[None, :] < K)

    p_x = x_ptr + o_t[:, None] * K + o_k[None, :]
    p_dout = dout_ptr + o_t[:, None] * K + o_k[None, :]
    b_x = tl.load(p_x, mask=m_x, other=0.0).to(tl.float32)
    b_dout = tl.load(p_dout, mask=m_x, other=0.0).to(tl.float32)

    inv_norm = tl.load(inv_norm_ptr + o_t, mask=m_t, other=0.0).to(tl.float32)
    inv_norm2 = inv_norm * inv_norm

    sum_dout_x = tl.sum(b_dout * b_x, 1)
    b_dx = (b_dout - b_x * sum_dout_x[:, None] * inv_norm2[:, None]) * inv_norm[:, None]

    tl.store(
        dx_ptr + o_t[:, None] * K + o_k[None, :],
        b_dx.to(dx_ptr.dtype.element_ty),
        mask=m_x,
    )


def gdn_chunk_l2norm_bwd(x, inv_norm, dout):
    """L2 归一化反向封装。

    Args:
        x: [B, H, T, K]，前向原始输入。
        inv_norm: [B, H, T]，前向保存的逆范数。
        dout: [B, H, T, K]，输出梯度。

    Returns:
        dx: [B, H, T, K]，输入梯度。
    """
    B, H, T, K = x.shape
    x_2d = x.reshape(B * H * T, K)
    dout_2d = dout.reshape(B * H * T, K)
    dx_2d = torch.empty_like(x_2d)
    inv_norm_1d = inv_norm.reshape(B * H * T)

    BK = triton.next_power_of_2(K)

    def grid(meta):
        return (triton.cdiv(B * H * T, meta["BT"]),)

    _gdn_chunk_l2norm_bwd_kernel[grid](
        x_2d,
        inv_norm_1d,
        dout_2d,
        dx_2d,
        B * H * T,
        K,
        BK=BK,
    )
    return dx_2d.view(B, H, T, K)

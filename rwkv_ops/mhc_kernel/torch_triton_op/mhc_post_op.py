"""mHC Post-Op PyTorch -> Triton 桥接。"""

import torch
import triton
from ..triton_kernel.mhc_post_op import (
    mhc_fused_backward_kernel,
    mhc_fused_forward_kernel,
)


def mhc_post_op_forward(
    layer_out: torch.tensor,
    x_expanded: torch.tensor,
    h_post_raw: torch.tensor,
    H_res: torch.tensor,
) -> torch.tensor:
    """PyTorch 前向 Triton launcher（私有）。"""
    batch, time, NSIZE, channel = x_expanded.shape
    total_batch_time = batch * time

    x_v = x_expanded.reshape(-1, NSIZE, channel).contiguous()
    h_v = h_post_raw.reshape(-1, NSIZE).contiguous()
    H_v = H_res.reshape(-1, NSIZE, NSIZE).contiguous()
    l_v = layer_out.reshape(-1, channel).contiguous()

    output_tensor = torch.empty_like(x_v)

    def grid(META):
        return (total_batch_time, triton.cdiv(channel, META["BLOCK_CHANNEL"]))

    mhc_fused_forward_kernel[grid](
        x_v,
        h_v,
        H_v,
        l_v,
        output_tensor,
        stride_output_batch_time=output_tensor.stride(0),
        stride_output_n_size=output_tensor.stride(1),
        stride_output_channel=output_tensor.stride(2),
        stride_x_batch_time=x_v.stride(0),
        stride_x_n_size=x_v.stride(1),
        stride_x_channel=x_v.stride(2),
        stride_h_batch_time=h_v.stride(0),
        stride_h_n_size=h_v.stride(1),
        stride_H_batch_time=H_v.stride(0),
        stride_H_n_size_1=H_v.stride(1),
        stride_H_n_size_2=H_v.stride(2),
        stride_layer_out_batch_time=l_v.stride(0),
        stride_layer_out_channel=l_v.stride(1),
        # Constants
        CHANNEL_SIZE=channel,
        NSIZE=NSIZE,
    )

    return output_tensor.view(batch, time, NSIZE, channel)


def mhc_post_op_backward(
    grad_output: torch.Tensor,
    layer_out: torch.Tensor,
    x_expanded: torch.Tensor,
    h_post_raw: torch.Tensor,
    H_res: torch.Tensor,
):
    """PyTorch 反向 Triton launcher（私有）。"""
    B, T, n, C = x_expanded.shape
    total_bt = B * T

    x_v = x_expanded.reshape(-1, n, C).contiguous()
    h_v = h_post_raw.reshape(-1, n).contiguous()
    H_v = H_res.reshape(-1, n, n).contiguous()
    l_v = layer_out.reshape(-1, C).contiguous()
    g_out_v = grad_output.reshape(-1, n, C).contiguous()

    grad_x = torch.empty_like(x_v)
    grad_l = torch.empty_like(l_v)
    # 规约结果用 FP32 保证精度，再转回目标 dtype。
    grad_h = torch.empty_like(h_v, dtype=torch.float32)
    grad_H = torch.empty_like(H_v, dtype=torch.float32)

    # Grid Y = 1 使单个 program 处理整行 C，从而消除原子加。
    grid = (total_bt, 1)

    mhc_fused_backward_kernel[grid](
        x_v,
        h_v,
        H_v,
        l_v,
        g_out_v,
        grad_x,
        grad_h,
        grad_H,
        grad_l,
        # Strides
        stride_x_bt=x_v.stride(0),
        stride_x_n=x_v.stride(1),
        stride_x_c=x_v.stride(2),
        stride_h_bt=h_v.stride(0),
        stride_h_n=h_v.stride(1),
        stride_H_bt=H_v.stride(0),
        stride_H_n1=H_v.stride(1),
        stride_H_n2=H_v.stride(2),
        stride_l_bt=l_v.stride(0),
        stride_l_c=l_v.stride(1),
        stride_g_bt=g_out_v.stride(0),
        stride_g_n=g_out_v.stride(1),
        stride_g_c=g_out_v.stride(2),
        stride_gx_bt=grad_x.stride(0),
        stride_gx_n=grad_x.stride(1),
        stride_gx_c=grad_x.stride(2),
        stride_gl_bt=grad_l.stride(0),
        stride_gl_c=grad_l.stride(1),
        stride_gh_bt=grad_h.stride(0),
        stride_gh_n=grad_h.stride(1),
        stride_gH_bt=grad_H.stride(0),
        stride_gH_n1=grad_H.stride(1),
        stride_gH_n2=grad_H.stride(2),
        # Constants
        CHANNEL_SIZE=C,
        NSIZE=n,
    )

    return (
        grad_l.view(B, T, C),
        grad_x.view(B, T, n, C),
        grad_h.view(B, T, n).to(h_post_raw.dtype),
        grad_H.view(B, T, n, n).to(H_res.dtype),
    )


class MHCPostOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_out, x_expanded, h_post_raw, H_res):
        # 保存张量以供反向传播使用
        ctx.save_for_backward(layer_out, x_expanded, h_post_raw, H_res)
        return mhc_post_op_forward(layer_out, x_expanded, h_post_raw, H_res)

    @staticmethod
    def backward(ctx, grad_output):
        # 获取 forward 中保存的张量
        layer_out, x_expanded, h_post_raw, H_res = ctx.saved_tensors
        # 计算梯度
        grads = mhc_post_op_backward(
            grad_output, layer_out, x_expanded, h_post_raw, H_res
        )
        return grads


def mhc_post_op(
    layer_out: torch.Tensor,
    x_expanded: torch.Tensor,
    h_post_raw: torch.Tensor,
    H_res: torch.Tensor,
) -> torch.Tensor:
    """Multi-Head Control (mHC) Post-Operation（Triton 加速）。

    该算子将核心层单流输出通过 h_post 门控分发回多流，并与 H_res 混合后的
    多流残差相加，得到更新后的多流表示。

    Args:
        layer_out: [B, T, C], bfloat16。核心层（Attention/FFN）输出。
        x_expanded: [B, T, n, C], bfloat16。原始多流残差。
        h_post_raw: [B, T, n], float32/bfloat16。未激活分发权重。
        H_res: [B, T, n, n], float32/bfloat16。双随机流混合矩阵。

    Returns:
        [B, T, n, C], bfloat16。更新后的多流残差。

    Raises:
        ValueError: C 不能被 128 整除。

    Examples:
        >>> x_next = mhc_post_op(layer_out, x_expanded, h_post_raw, H_res)
    """
    C = layer_out.shape[-1]
    if C % 128 != 0:
        raise ValueError(f"mhc_post_op (triton) requires C % 128 == 0, got C={C}")
    return MHCPostOpFunction.apply(
        layer_out.to(torch.bfloat16),
        x_expanded.to(torch.bfloat16),
        h_post_raw.to(torch.float32),
        H_res.to(torch.float32),
    )

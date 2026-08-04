"""mHC Pre-Op PyTorch -> Triton 桥接。"""

import torch
import triton

from ..triton_kernel.mhc_pre_op import (
    sinkhorn_aggregate_bwd_kernel,
    sinkhorn_aggregate_fused_kernel,
)


def mhc_pre_op_fwd_kernel_call(
    x: torch.Tensor,
    h_res_in: torch.Tensor,
    h_pre_in: torch.Tensor,
    n: int,
    num_iters: int = 20,
    eps: float = 1e-8,
):
    """PyTorch 前向 Triton launcher（私有）。"""
    B, T, _, C = x.shape
    Total_BT = B * T

    x = x.contiguous()
    h_res_in = h_res_in.contiguous()
    h_pre_in = h_pre_in.contiguous()

    out = torch.empty((B, T, C), device=x.device, dtype=x.dtype)
    H_res_out = torch.empty((B, T, n, n), device=x.device, dtype=torch.float32)

    def grid(META):
        return (
            triton.cdiv(Total_BT, META["BLOCK_BT"]),
            triton.cdiv(C, META["BLOCK_C"]),
        )

    sinkhorn_aggregate_fused_kernel[grid](
        x,
        h_res_in,
        h_pre_in,
        out,
        H_res_out,
        Total_BT_CONST=Total_BT,
        NSIZE=n,
        CSIZE=C,
        NUM_ITERS=num_iters,
        EPS=eps,
        # x 原始为 [B, T, n, C]；contiguous 后 view 为 [Total_BT, n, C]，
        # 但 stride 仍沿用原始张量的最后三维。
        stride_x_bt=x.view(Total_BT, n, C).stride(0),
        stride_x_n=x.stride(2),
        stride_x_c=x.stride(3),
        stride_h_res_in_bt=h_res_in.view(Total_BT, n, n).stride(0),
        stride_h_res_in_n1=h_res_in.stride(2),
        stride_h_res_in_n2=h_res_in.stride(3),
        stride_h_pre_in_bt=h_pre_in.view(Total_BT, n).stride(0),
        stride_h_pre_in_n=h_pre_in.stride(2),
        stride_out_bt=out.view(Total_BT, C).stride(0),
        stride_out_c=out.stride(2),
        stride_Hr_out_bt=H_res_out.view(Total_BT, n, n).stride(0),
        stride_Hr_out_n1=H_res_out.stride(2),
        stride_Hr_out_n2=H_res_out.stride(3),
    )

    return out, H_res_out


def mhc_pre_op_bwd_kernel_call(grad_out, grad_H_res, x, h_res, h_pre, n, iters, eps):
    """PyTorch 反向 Triton launcher（私有）。"""
    B, T, _, C = x.shape
    BT = B * T
    gx = torch.empty_like(x)
    gh_res = torch.empty_like(h_res)
    gh_pre = torch.empty_like(h_pre)
    # Grid Y = 1 使单个 program 处理整行 C，从而消除原子加。
    grid = (BT, 1)

    sinkhorn_aggregate_bwd_kernel[grid](
        grad_out,
        grad_H_res,
        x,
        h_res,
        h_pre,
        gx,
        gh_res,
        gh_pre,
        TOTAL_BT_CONST=BT,
        NSIZE=n,
        CHANNEL_SIZE=C,
        NUM_ITERS=iters,
        EPS=eps,
        stride_gout_bt=grad_out.view(BT, C).stride(0),
        stride_gout_c=grad_out.stride(2),
        stride_gH_bt=grad_H_res.view(BT, n, n).stride(0),
        stride_gH_n1=grad_H_res.stride(2),
        stride_gH_n2=grad_H_res.stride(3),
        stride_x_bt=x.view(BT, n, C).stride(0),
        stride_x_n=x.stride(2),
        stride_x_c=x.stride(3),
        stride_h_res_bt=h_res.view(BT, n, n).stride(0),
        stride_h_res_n1=h_res.stride(2),
        stride_h_res_n2=h_res.stride(3),
        stride_h_pre_bt=h_pre.view(BT, n).stride(0),
        stride_h_pre_n=h_pre.stride(2),
        stride_gx_bt=gx.view(BT, n, C).stride(0),
        stride_gx_n=gx.stride(2),
        stride_gx_c=gx.stride(3),
        stride_gh_res_bt=gh_res.view(BT, n, n).stride(0),
        stride_gh_res_n1=gh_res.stride(2),
        stride_gh_res_n2=gh_res.stride(3),
        stride_gh_pre_bt=gh_pre.view(BT, n).stride(0),
        stride_gh_pre_n=gh_pre.stride(2),
    )
    return gx, gh_res, gh_pre


class MHCFusedPreOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, h_res_in, h_pre_in, n, num_iters, eps):
        out, H_res_out = mhc_pre_op_fwd_kernel_call(
            x, h_res_in, h_pre_in, n, num_iters, eps
        )
        ctx.save_for_backward(x, h_res_in, h_pre_in)
        ctx.params = (n, num_iters, eps)
        return out, H_res_out

    @staticmethod
    def backward(ctx, grad_out, grad_H_res):
        x, h_res, h_pre = ctx.saved_tensors
        n, iters, eps = ctx.params
        if grad_out is None:
            grad_out = torch.zeros_like(x[:, :, 0])
        if grad_H_res is None:
            grad_H_res = torch.zeros_like(h_res)
        gx, gh_res, gh_pre = mhc_pre_op_bwd_kernel_call(
            grad_out, grad_H_res, x, h_res, h_pre, n, iters, eps
        )
        return gx, gh_res, gh_pre, None, None, None


def mhc_pre_op_fused(
    x: torch.Tensor,
    h_res_in: torch.Tensor,
    h_pre_in: torch.Tensor,
    num_iters: int = 20,
    eps: float = 1e-8,
):
    """mHC Pre-Op PyTorch 公开入口（Triton 加速）。

    将多流输入通过 Sinkhorn-Knopp 生成双随机残差矩阵，并聚合为单流层输入。

    Args:
        x: [B, T, n, C], bfloat16。多流输入。
        h_res_in: [B, T, n, n], float32。未归一化残差矩阵。
        h_pre_in: [B, T, n], float32。未激活聚合权重。
        num_iters: int，默认 20。Sinkhorn-Knopp 迭代次数。
        eps: float，默认 1e-8。数值稳定常数。

    Returns:
        out: [B, T, C], bfloat16。聚合后的层输入。
        H_res_out: [B, T, n, n], float32。双随机残差矩阵。

    Raises:
        ValueError: C 不能被 128 整除。
    """
    C = x.shape[-1]
    if C % 128 != 0:
        raise ValueError(f"mhc_pre_op_fused (triton) requires C % 128 == 0, got C={C}")
    n = x.shape[-2]
    return MHCFusedPreOp.apply(x, h_res_in, h_pre_in, n, num_iters, eps)

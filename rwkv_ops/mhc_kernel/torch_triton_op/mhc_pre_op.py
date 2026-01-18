import torch
import triton
import triton.language as tl

# 假设你的 kernel 代码保存在这个路径
from ..triton_kernel.mhc_pre_op import *


def mhc_pre_op_fwd_kernel_call(
    x: torch.Tensor,
    h_res_in: torch.Tensor,
    h_pre_in: torch.Tensor,
    n: int,
    num_iters: int = 20,
    eps: float = 1e-8,
):
    """
    Triton Kernel Launcher for mHC Pre-Op Forward
    """
    # 1. 形状检查与准备
    B, T, _, C = x.shape
    Total_BT = B * T

    # 2. 确保内存连续 (Triton 性能关键)
    x = x.contiguous()
    h_res_in = h_res_in.contiguous()
    h_pre_in = h_pre_in.contiguous()

    # 3. 准备输出张量
    # x_layer_in: [B, T, C] (BF16)
    out = torch.empty((B, T, C), device=x.device, dtype=x.dtype)
    # H_res_out: [B, T, n, n] (FP32)
    H_res_out = torch.empty((B, T, n, n), device=x.device, dtype=torch.float32)

    # 4. 展平视图 (View as [Total_BT, ...])
    # 注意：stride 的获取必须基于传入的张量，而不是 view 后的
    # 但由于我们做了 contiguous，view 后的 stride 也是标准的

    # 5. Grid 计算函数
    # Grid: (Total_BT // BLOCK_BT, C // BLOCK_C)
    grid = lambda META: (
        triton.cdiv(Total_BT, META["BLOCK_BT"]),
        triton.cdiv(C, META["BLOCK_C"]),
    )

    # 6. 启动 Kernel
    sinkhorn_aggregate_fused_kernel[grid](
        # --- 指针 ---
        x,
        h_res_in,
        h_pre_in,
        out,
        H_res_out,
        # --- 常量 (constexpr) ---
        Total_BT_CONST=Total_BT,
        NSIZE=n,
        CSIZE=C,
        NUM_ITERS=num_iters,
        EPS=eps,
        # --- 步幅 (View as [Total_BT, ...]) ---
        # x: [BT, n, C] -> stride(0) 是 B维度stride, stride(1) 是 T维度stride
        # 对于 flatten 后的 [BT, n, C]，stride_bt 就是 n*C (若连续)
        # 最稳妥的方式是用 reshape 后的 stride
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
    B, T, _, C = x.shape
    BT = B * T
    gx = torch.empty_like(x)
    gh_res = torch.empty_like(h_res)
    gh_pre = torch.empty_like(h_pre)
    grid = (BT, 1)  # 消除原子操作的关键：Grid Y = 1

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
        # 步幅映射 (与 Kernel 内部签名一一对应)
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
            grad_out = torch.zeros_like(x[:, :, 0])  # dummy
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
    """
    Triton 加速版 mHC Pre-Op (Fused Sinkhorn + Stream Aggregate)

    参数:
        x: [B, T, n, C] (BF16)
        h_res_in: [B, T, n, n] (FP32) - 未归一化的残差矩阵
        h_pre_in: [B, T, n] (FP32) - 未激活的聚合权重

    返回:
        out: [B, T, C] (BF16) - 聚合后的层输入
        H_res_out: [B, T, n, n] (FP32) - 双随机残差矩阵
    """
    n = x.shape[-2]
    return MHCFusedPreOp.apply(x, h_res_in, h_pre_in, n, num_iters, eps)

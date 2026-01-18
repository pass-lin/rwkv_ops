import torch
from ..triton_kernel.mhc_post_op import *


def mhc_post_op_forward(
    layer_out: torch.tensor,
    x_expanded: torch.tensor,
    h_post_raw: torch.tensor,
    H_res: torch.tensor,
) -> torch.tensor:
    batch, time, NSIZE, channel = x_expanded.shape
    total_batch_time = batch * time

    # 内存连续化处理
    x_v = x_expanded.reshape(-1, NSIZE, channel).contiguous()
    h_v = h_post_raw.reshape(-1, NSIZE).contiguous()
    H_v = H_res.reshape(-1, NSIZE, NSIZE).contiguous()
    l_v = layer_out.reshape(-1, channel).contiguous()

    output_tensor = torch.empty_like(x_v)

    grid = lambda META: (total_batch_time, triton.cdiv(channel, META["BLOCK_CHANNEL"]))

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
    B, T, n, C = x_expanded.shape
    total_bt = B * T
    device = x_expanded.device

    # 1. 连续化
    x_v = x_expanded.reshape(-1, n, C).contiguous()
    h_v = h_post_raw.reshape(-1, n).contiguous()
    H_v = H_res.reshape(-1, n, n).contiguous()
    l_v = layer_out.reshape(-1, C).contiguous()
    g_out_v = grad_output.reshape(-1, n, C).contiguous()

    # 2. 准备输出
    grad_x = torch.empty_like(x_v)
    grad_l = torch.empty_like(l_v)

    # 3. Workspace 策略 (按最小 Block 128 分配)
    MIN_BLOCK = 128
    GRID_Y_MAX = (C + MIN_BLOCK - 1) // MIN_BLOCK
    
    # 必须用 zeros，防止大 block 时部分 slot 未触及
    grad_h_ws = torch.zeros((total_bt, GRID_Y_MAX, n), device=device, dtype=torch.float32)
    grad_H_ws = torch.zeros((total_bt, GRID_Y_MAX, n, n), device=device, dtype=torch.float32)

    # 4. Grid
    grid = lambda META: (total_bt, triton.cdiv(C, META["BLOCK_CHANNEL"]))

    mhc_fused_backward_kernel_workspace[grid](
        x_v, h_v, H_v, l_v, g_out_v,
        grad_x, grad_h_ws, grad_H_ws, grad_l,
        # Strides
        x_v.stride(0), x_v.stride(1), x_v.stride(2),
        h_v.stride(0), h_v.stride(1),
        H_v.stride(0), H_v.stride(1), H_v.stride(2),
        l_v.stride(0), l_v.stride(1),
        g_out_v.stride(0), g_out_v.stride(1), g_out_v.stride(2),
        grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
        grad_l.stride(0), grad_l.stride(1),
        grad_h_ws.stride(0), grad_h_ws.stride(1), grad_h_ws.stride(2),
        grad_H_ws.stride(0), grad_H_ws.stride(1), grad_H_ws.stride(2), grad_H_ws.stride(3),
        # Constants
        CHANNEL_SIZE=C,
        NSIZE=n,
    )

    # 5. 最后归约
    grad_h = grad_h_ws.sum(dim=1).to(h_post_raw.dtype)
    grad_H = grad_H_ws.sum(dim=1).to(H_res.dtype)

    return (
        grad_l.view(B, T, C),
        grad_x.view(B, T, n, C),
        grad_h.view(B, T, n),
        grad_H.view(B, T, n, n),
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
    """
    Multi-Head Control (MHC) Post-Operation.

    该算子实现了 RWKV 模型中的 MHC 后处理融合逻辑，包括：
    1. 门控计算: w = 2 * sigmoid(h_post_raw)
    2. 通道分配: x_delta = w * layer_out
    3. 矩阵混合: x_mixed = H_res @ x_expanded
    4. 融合输出: Out = x_mixed + x_delta

    使用 Triton 实现全融合 Kernel，相比原生 PyTorch 具有更低的显存带宽占用和更高的执行效率。

    参数:
        layer_out (torch.Tensor): 形状为 [B, T, C]，bfloat16 类型。
        x_expanded (torch.Tensor): 形状为 [B, T, n, C]，bfloat16 类型。
        h_post_raw (torch.Tensor): 形状为 [B, T, n]，float32 或 bfloat16 类型。
        H_res (torch.Tensor): 形状为 [B, T, n, n]，float32 或 bfloat16 类型。

    返回:
        torch.Tensor: 形状为 [B, T, n, C]，bfloat16 类型。

    形状说明:
        B: Batch Size
        T: Sequence Length
        C: Channel Size (Hidden Dimension)
        n: Head Size (State Dimension, 通常为 4 或 8)
    """
    return MHCPostOpFunction.apply(
        layer_out.to(torch.bfloat16),
        x_expanded.to(torch.bfloat16),
        h_post_raw.to(torch.float32),
        H_res.to(torch.float32),
    )

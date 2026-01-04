import os
import torch
from torch.utils.cpp_extension import load

# 路径配置
current_dir = os.path.dirname(os.path.abspath(__file__))
common_inc = os.path.abspath(os.path.join(current_dir, "../common_kernel/include"))
common_ker = os.path.abspath(os.path.join(current_dir, "../common_kernel/kernels"))

mhc_lib = load(
    name="mhc_cuda_kernel",
    sources=[
        os.path.join(current_dir, "mhc_op.cpp"),
        os.path.join(current_dir, "mhc_cuda.cu"),
    ],
    extra_include_paths=[common_inc, common_ker],
    extra_cuda_cflags=[
        "-O3",
        "--use_fast_math",
        "-std=c++17",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    verbose=True,
)


class SinkhornKnoppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, num_iters=20, eps=1e-8):
        x = inp.float().contiguous()
        x_max = torch.amax(x, dim=(-1, -2), keepdim=True)
        out = mhc_lib.sinkhorn_fwd(x - x_max, num_iters, eps)
        ctx.save_for_backward(out, x - x_max)
        ctx.num_iters, ctx.eps = num_iters, eps
        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out, x_stabilized = ctx.saved_tensors
        d_inp = mhc_lib.sinkhorn_bwd(
            grad_output.float().contiguous(), out, x_stabilized, ctx.num_iters, ctx.eps
        )
        return d_inp.to(grad_output.dtype), None, None


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, eps=1e-5):
        inp = inp.to(torch.bfloat16).contiguous()
        out = mhc_lib.rmsnorm_fwd(inp, eps)
        ctx.save_for_backward(inp)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (inp,) = ctx.saved_tensors
        dx = mhc_lib.rmsnorm_bwd(
            grad_output.to(torch.bfloat16).contiguous(), inp, ctx.eps
        )
        return dx, None


class StreamMixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, M):
        inp = inp.to(torch.bfloat16).contiguous()
        M = M.float().contiguous()
        out = mhc_lib.stream_mix_fwd(inp, M)
        ctx.save_for_backward(inp, M)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, M = ctx.saved_tensors
        grad_output_fp32 = grad_output.float().contiguous()
        d_inp, d_M = mhc_lib.stream_mix_backward(grad_output_fp32, inp, M)
        return d_inp, d_M


# --- 新增：Stream Aggregate 功能 ---
class StreamAggregateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, H_pre):
        inp = inp.to(torch.bfloat16).contiguous()
        H_pre = H_pre.float().contiguous()

        # 判断权重模式
        per_token = H_pre.dim() == 3

        out = mhc_lib.stream_aggregate_fwd(inp, H_pre, per_token)
        ctx.save_for_backward(inp, H_pre)
        ctx.per_token = per_token
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, H_pre = ctx.saved_tensors
        # 精度核心：强制将梯度转为 float32 传入内核进行规约
        grad_output_fp32 = grad_output.float().contiguous()
        d_inp, d_H_pre = mhc_lib.stream_aggregate_bwd(
            grad_output_fp32, inp, H_pre, ctx.per_token
        )
        return d_inp, d_H_pre


def stream_aggregate(inp, H_pre):
    return StreamAggregateFunction.apply(inp, H_pre)


class StreamDistributeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, H_post):
        """
        inp: [B, T, C] (通常为 bf16)
        H_post: [B, T, n] (通常为 fp32)
        返回: [B, T, n, C] (bf16)
        """
        # 1. 强制连续性以适配 CUDA 内核
        ctx.inp_dtype = inp.dtype
        ctx.H_post_dtype = H_post.dtype
        inp = inp.bfloat16().contiguous()
        H_post = H_post.float().contiguous()

        B, T, C = inp.shape
        n = H_post.shape[-1]

        out = mhc_lib.stream_distribute_fwd(inp, H_post)

        ctx.save_for_backward(inp, H_post)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output: [B, T, n, C] (反向传回的梯度)
        返回: d_inp, d_H_post
        """
        inp, H_post = ctx.saved_tensors
        grad_output = grad_output.contiguous()

        # 调用 C++ 绑定的反向内核
        # 内核内部会计算:
        # d_inp = sum_i(grad_output[..., i, :] * H_post[..., i])
        # d_H_post = sum_c(grad_output[..., :, c] * inp[..., c])
        d_inp, d_H_post = mhc_lib.stream_distribute_bwd(grad_output, inp, H_post)

        # 对应 forward 的参数顺序：inp, H_post
        return d_inp.to(ctx.inp_dtype), d_H_post.to(ctx.H_post_dtype)


def stream_distribute(inp, H_post):
    """
    mHC 分发算子 (1 -> n): 将单流信号按照权重分发到 n 个并行流中。
    """
    return StreamDistributeFunction.apply(inp, H_post)


# 辅助接口
def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    return SinkhornKnoppFunction.apply(inp, num_iters, eps)


def rmsnorm(inp, eps=1e-5):
    return RMSNormFunction.apply(inp, eps)


def stream_mix(inp, M):
    return StreamMixFunction.apply(inp, M)

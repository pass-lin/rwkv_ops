import os
import torch
from torch.utils.cpp_extension import load

# 路径配置：自动定位当前目录下的源文件
current_dir = os.path.dirname(os.path.abspath(__file__))
common_inc = os.path.abspath(os.path.join(current_dir, "../common_kernel/include"))
common_ker = os.path.abspath(os.path.join(current_dir, "../common_kernel/kernels"))

# 编译并加载 CUDA 扩展
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


# -------------------- Sinkhorn Knopp 封装 --------------------
class SinkhornKnoppFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, num_iters=20, eps=1e-8):
        # 确保 FP32 运算以保证迭代稳定性
        x = inp.float().contiguous()
        # 减去最大值防止 exp 溢出
        x_max = torch.amax(x, dim=(-1, -2), keepdim=True)
        x_stabilized = x - x_max

        out = mhc_lib.sinkhorn_fwd(x_stabilized, num_iters, eps)

        ctx.save_for_backward(out, x_stabilized)
        ctx.num_iters = num_iters
        ctx.eps = eps
        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        out, x_stabilized = ctx.saved_tensors
        grad_output = grad_output.float().contiguous()

        d_inp = mhc_lib.sinkhorn_bwd(
            grad_output, out, x_stabilized, ctx.num_iters, ctx.eps
        )
        return d_inp.to(out.dtype), None, None


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    return SinkhornKnoppFunction.apply(inp, num_iters, eps)


class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, eps=1e-5):
        # 仅接收输入张量
        x = inp.to(torch.bfloat16).contiguous()
        out = mhc_lib.rmsnorm_fwd(x, eps)
        ctx.save_for_backward(x)
        ctx.eps = eps
        return out.to(inp.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_output = grad_output.to(torch.bfloat16).contiguous()
        # 调用不带 weight 的反向内核
        dx = mhc_lib.rmsnorm_bwd(grad_output, x, ctx.eps)
        # 只返回 dx 的梯度，eps 不需要梯度
        return dx.to(x.dtype), None


def rmsnorm(inp, eps=1e-5):
    """
    对齐 mHC 论文实现的不带 Weight 的 RMSNorm
    """
    return RMSNormFunction.apply(inp, eps)


class StreamMixFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, M):
        inp = inp.to(torch.bfloat16).contiguous()
        M = M.float().contiguous()
        out = mhc_lib.stream_mix_fwd(inp, M)
        ctx.save_for_backward(inp, M)
        ctx.inp_dtype = inp.dtype
        ctx.M_dtype = M.dtype
        return out

    @staticmethod
    def backward(ctx, grad_output):
        inp, M = ctx.saved_tensors
        # --- 核心改进：将梯度转为 float32 以对齐 CUDA 并行规约精度 ---
        grad_output_fp32 = grad_output.float().contiguous()
        d_inp, d_M = mhc_lib.stream_mix_backward(grad_output_fp32, inp, M)
        return d_inp.to(ctx.inp_dtype), d_M.to(ctx.M_dtype)


def stream_mix(inp, M):
    """
    Mix (n -> n): 残差流之间的线性交互。
    inp: [B, T, n, C]
    M: [B, T, n, n] (通常来自 sinkhorn_knopp)
    """
    return StreamMixFunction.apply(inp, M)

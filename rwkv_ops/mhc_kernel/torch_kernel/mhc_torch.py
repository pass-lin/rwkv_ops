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


class MHCPostOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, layer_out, x_expanded, H_post, H_res):
        # 强制连续性
        layer_out = layer_out.contiguous()
        x_expanded = x_expanded.contiguous()
        H_post = H_post.contiguous()
        H_res = H_res.contiguous()

        # 保存用于反向传播的张量
        ctx.save_for_backward(layer_out, x_expanded, H_post, H_res)

        # 调用融合前向内核
        x_next = mhc_lib.mhc_post_op_fwd(layer_out, x_expanded, H_post, H_res)
        return x_next

    @staticmethod
    def backward(ctx, grad_next):
        # 获取保存的张量
        layer_out, x_expanded, H_post, H_res = ctx.saved_tensors
        grad_next = grad_next.contiguous()

        # 调用全量融合反向内核
        # 返回列表: [d_layer_out, d_x_expanded, d_H_post, d_H_res]
        grads = mhc_lib.mhc_post_op_bwd(grad_next, layer_out, x_expanded, H_post, H_res)

        # 返回 4 个梯度，对应 forward 的 4 个输入
        return grads[0], grads[1], grads[2], grads[3]


def mhc_post_op(layer_out, x_expanded, H_post, H_res):
    """
    mHC 融合后处理算子
    layer_out: [B, T, C]
    x_expanded: [B, T, n, C]
    H_post: [B, T, n]
    H_res: [B, T, n, n]
    """
    return MHCPostOpFunction.apply(layer_out, x_expanded, H_post, H_res)


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


class MHCPreOpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, x_expanded, h_pre_raw, h_post_raw, h_res_raw, num_iters=20, eps=1e-8
    ):
        # 1. 保存原始类型
        ctx.x_dtype = x_expanded.dtype
        ctx.h_dtype = h_pre_raw.dtype  # 通常是 fp32，但需要记录

        # 2. 强制类型检查与转换 (为了对齐 C++ 接口)
        # x_expanded 必须是 bfloat16 (对应 nv_bfloat16*)
        x_expanded = x_expanded.to(dtype=torch.bfloat16).contiguous()
        # 参数类 tensor 必须是 float32 (对应 float*)
        h_pre_raw = h_pre_raw.to(dtype=torch.float32).contiguous()
        h_post_raw = h_post_raw.to(dtype=torch.float32).contiguous()
        h_res_raw = h_res_raw.to(dtype=torch.float32).contiguous()

        # 3. 调用 CUDA 接口 (返回: x_layer_in [bf16], H_pre [f32], H_post [f32], H_res [f32])
        x_layer_in, H_pre, H_post, H_res = mhc_lib.mhc_pre_op_fwd(
            x_expanded, h_pre_raw, h_post_raw, h_res_raw, num_iters, eps
        )

        # 4. 保存反向传播需要的中间变量
        ctx.save_for_backward(x_expanded, H_pre, H_post, H_res, h_res_raw)
        ctx.num_iters = num_iters
        ctx.eps = eps

        # 5. 将主干输出转回原始类型 (通常是 bf16)
        return x_layer_in.to(dtype=ctx.x_dtype), H_post, H_res

    @staticmethod
    def backward(ctx, grad_layer_in, grad_H_post, grad_H_res):
        x_expanded, H_pre, H_post, H_res, h_res_raw = ctx.saved_tensors

        # 1. 强制梯度类型对齐 C++ 反向接口
        grad_layer_in = grad_layer_in.to(dtype=torch.bfloat16).contiguous()
        grad_H_post = grad_H_post.to(dtype=torch.float32).contiguous()
        grad_H_res = grad_H_res.to(dtype=torch.float32).contiguous()

        # 2. 调用 CUDA 反向内核
        # 返回 grads: [d_x_expanded, d_h_pre_raw, d_h_post_raw, d_h_res_raw]
        grads = mhc_lib.mhc_pre_op_bwd(
            grad_layer_in,
            grad_H_post,
            grad_H_res,
            x_expanded,
            H_pre,
            H_post,
            H_res,
            h_res_raw,
            ctx.num_iters,
            ctx.eps,
        )

        # 3. 类型还原：将计算出的梯度转回输入时的原始数据类型
        # 防止下游优化器（如 Adam）因为梯度类型不匹配而报错或增加额外的 cast 开销
        dx = grads[0].to(dtype=ctx.x_dtype)
        d_h_pre = grads[1].to(dtype=ctx.h_dtype)
        d_h_post = grads[2].to(dtype=ctx.h_dtype)
        d_h_res = grads[3].reshape(h_res_raw.shape).to(dtype=ctx.h_dtype)

        # 返回 4 个输入对应的梯度，最后两个参数 num_iters/eps 对应 None
        return dx, d_h_pre, d_h_post, d_h_res, None, None


def mhc_pre_op(x_expanded, h_pre_raw, h_post_raw, h_res_raw, num_iters=20, eps=1e-8):
    """
    mHC 前处理融合算子接口
    """
    # 预处理：h_res_raw 可能是 [B, T, n, n] 或 [B, T, n*n]
    if h_res_raw.dim() == 4:
        h_res_raw_flat = h_res_raw.reshape(h_res_raw.shape[0], h_res_raw.shape[1], -1)
    else:
        h_res_raw_flat = h_res_raw

    return MHCPreOpFunction.apply(
        x_expanded, h_pre_raw, h_post_raw, h_res_raw_flat, num_iters, eps
    )

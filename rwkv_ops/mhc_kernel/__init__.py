import keras
from keras import ops


def get_mhc_kernel(KERNEL_TYPE="native"):
    from .native_op import (
        linear_and_reshape,
        mhc_post_op,
        mhc_pre_op_fused,
        mhc_rmsnorm,
    )

    if KERNEL_TYPE == "triton":
        if keras.config.backend() == "jax":
            import jax

            if jax.devices()[0].platform == "gpu":
                from .jax_triton_op.mhc_pre_op import mhc_pre_op_fused
                from .jax_triton_op.mhc_post_op import mhc_post_op
        elif keras.config.backend() == "torch":
            import torch

            if torch.cuda.is_available():
                from .torch_triton_op.mhc_pre_op import mhc_pre_op_fused
                from .torch_triton_op.mhc_post_op import mhc_post_op

    def mhc_pre_op(
        x,
        alpha_pre,
        alpha_post,
        alpha_res,
        phi,
        bias_pre,
        bias_post,
        bias_res,
        n,
        num_iters=20,
        eps=1e-8,
    ):
        """
        mHC 预处理算子（高层封装）

        功能：生成 mHC 核心层所需的完整输入，包括聚合后的单流特征、分发权重和双随机残差矩阵。
        实现：linear_and_reshape → Sinkhorn-Knopp → stream_aggregate

        --- 参数 ---
        x: [batch_size, seq_len, n , hidden_size], BF16/FP32
            多流输入特征

        alpha_pre, alpha_post, alpha_res: (1,), float32
            分支缩放系数

        phi: [n * hidden_size, M], BF16
            投影矩阵（M = n*(n+2)）

        bias: [M], float32
            线性偏置

        n: int
            流数量

        num_iters: int, default=20
            Sinkhorn-Knopp 迭代次数

        eps: float, default=1e-8
            Sinkhorn-Knopp 数值稳定常数

        --- 返回 ---
        x_layer_in: [batch_size, seq_len, hidden_size], 与 x.dtype 相同
            聚合后的单流特征，计算公式：sum(x * sigmoid(h_pre_raw), axis=-2)
            作为 Attention/FFN 等核心层的输入

        H_post: [batch_size, seq_len, n], 与 x.dtype 相同
            分发权重（已激活），用于将核心层输出广播回多流

        H_res: [batch_size, seq_len, n, n], 与 x.dtype 相同
            双随机残差矩阵，实现 n 个流之间的可逆混合
        """

        original_dtype = x.dtype
        B, T = ops.shape(x)[:2]
        x_flat = ops.reshape(x, [B, T, -1])
        x_norm = mhc_rmsnorm(x_flat, eps)
        h_pre_raw, h_post_raw, h_res_reshaped = linear_and_reshape(
            x_norm,
            alpha_pre,
            alpha_post,
            alpha_res,
            phi,
            bias_pre,
            bias_post,
            bias_res,
            n,
            eps=eps,
        )

        x_layer_in, H_res = mhc_pre_op_fused(
            x,
            h_res_reshaped,
            h_pre_raw,
            num_iters,
            eps,
        )

        return (
            ops.cast(x_layer_in, original_dtype),
            ops.cast(h_post_raw, "float32"),
            ops.cast(H_res, "float32"),
        )

    return mhc_pre_op, mhc_post_op

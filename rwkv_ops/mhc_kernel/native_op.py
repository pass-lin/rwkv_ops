"""mHC 原生 Keras-ops 参考实现。"""

from keras import ops


def fp32_sigmoid(x):
    """在 float32 下计算 sigmoid，再 cast 回输入 dtype。"""
    dtype = x.dtype
    return ops.cast(ops.nn.sigmoid(ops.cast(x, "float32")), dtype)


def stream_distribute(inp, h_post_raw):
    """将单流输出分发回多流（Distribute 1 -> n）。

    强制在 float32 下做广播乘法，与 Triton 内核的 to_float 精度对齐。

    Args:
        inp: [B, T, C]，BF16/FP32。单流输入。
        h_post_raw: [B, T, n]，float32。未激活的原始分发权重。

    Returns:
        [B, T, n, C]，与 inp.dtype 相同。
    """
    # 记录原始类型
    original_dtype = inp.dtype
    # 原论文要求的fp32精度
    x_fp32 = ops.cast(ops.expand_dims(inp, -2), "float32")
    w_fp32 = ops.expand_dims(2.0 * fp32_sigmoid(ops.cast(h_post_raw, "float32")), -1)
    res_fp32 = x_fp32 * w_fp32
    return ops.cast(res_fp32, original_dtype)


def stream_mix(inp, M):
    """多流残差之间的线性交互（Mix n -> n）。

    Args:
        inp: [B, T, n, C]，BF16/FP32。
        M: [n, n] 或 [B, T, n, n]，float32。由 sinkhorn_knopp 生成的双随机矩阵。

    Returns:
        [B, T, n, C]，与 inp.dtype 相同。
    """
    dtype = inp.dtype
    inp = ops.cast(inp, M.dtype)
    if len(ops.shape(M)) == 2:
        out = ops.einsum("ij,btjk->btik", M, inp)
    else:
        out = ops.einsum("btij,btjk->btik", M, inp)
    return ops.cast(out, dtype)


def mhc_post_op(layer_out, x_expanded, h_post_raw, H_res):
    """mHC 后处理融合算子。

    Args:
        layer_out: [B, T, C]，BF16/FP32。核心层（Attention/FFN）处理后的输出。
        x_expanded: [B, T, n, C]，BF16/FP32。原始多流残差。
        h_post_raw: [B, T, n]，float32。未激活的分发权重。
        H_res: [B, T, n, n]，float32。双随机流混合矩阵。

    Returns:
        x_next: [B, T, n, C]，与 x_expanded.dtype 相同。更新后的多流残差。

    Examples:
        >>> x_next = mhc_post_op(layer_out, x_expanded, h_post_raw, H_res)
    """
    dtype = x_expanded.dtype
    layer_out = ops.cast(layer_out, "float32")
    x_expanded = ops.cast(x_expanded, "float32")
    h_post_raw = ops.cast(h_post_raw, "float32")
    H_res = ops.cast(H_res, "float32")
    # mhc要求fp32精度
    x_mixed_f32 = stream_mix(x_expanded, H_res)
    x_delta_f32 = stream_distribute(layer_out, h_post_raw)
    x_next_f32 = x_mixed_f32 + x_delta_f32
    return ops.cast(x_next_f32, dtype)


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    """将输入矩阵投影为双随机矩阵（Doubly Stochastic）。

    输入通常位于 log 域（H_res_raw）。先减最大值防指数溢出，再行/列
    交替归一化。

    Args:
        inp: [..., n, n]，任意实 dtype。
        num_iters: int，默认 20。迭代次数。
        eps: float，默认 1e-8。防除零常数。

    Returns:
        [..., n, n]，float32。双随机矩阵。
    """
    x = ops.cast(inp, "float32")
    # 防溢出技巧：减去最大值
    x = x - ops.max(x, axis=(-1, -2), keepdims=True)
    P = ops.exp(x)
    # 行列迭代归一化
    for _ in range(num_iters):
        P = P / (ops.sum(P, axis=-1, keepdims=True) + eps)
        P = P / (ops.sum(P, axis=-2, keepdims=True) + eps)

    return P


def mhc_rmsnorm(inp, eps=1e-5):
    """标准 RMSNorm。

    Args:
        inp: [..., C]，任意实 dtype。
        eps: float，默认 1e-5。数值稳定常数。

    Returns:
        [..., C]，与 inp.dtype 相同。
    """
    dtype = inp.dtype
    x = ops.cast(inp, "float32")
    rms = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + eps)
    x_normed = x / rms
    return ops.cast(x_normed, dtype)


def stream_aggregate(inp, H_pre):
    """将多流输入聚合为单流输出（Aggregate n -> 1）。

    在 float32 下完成 sigmoid 激活与加权求和，再 cast 回原始 dtype。

    Args:
        inp: [B, T, n, C]，任意实 dtype。
        H_pre: [B, T, n]，任意实 dtype。未激活的聚合权重。

    Returns:
        [B, T, C]，与 inp.dtype 相同。
    """
    inp_f32 = ops.cast(inp, "float32")
    H_f32 = ops.cast(H_pre, "float32")
    H_f32 = fp32_sigmoid(H_f32)
    out_f32 = ops.sum(inp_f32 * ops.expand_dims(H_f32, -1), axis=-2)
    return ops.cast(out_f32, inp.dtype)


def linear_and_reshape(
    x_norm,
    alpha_pre,
    alpha_post,
    alpha_res,
    phi,
    bias_pre,
    bias_post,
    bias_res,
    n,
    eps=1e-5,
):
    """mHC 动态投影与分支生成算子（底层实现）。

    将展平后的多流特征通过线性投影生成 pre/post/res 三组混合系数。

    Args:
        x_norm: [batch_size, seq_len, n * hidden_size]，BF16/FP32。
            展平后的多流特征。
        alpha_pre, alpha_post, alpha_res: (1,), float32。分支缩放系数。
        phi: [n * hidden_size, M]，BF16。投影矩阵，M = n * (n + 2)。
        bias_pre, bias_post, bias_res: [M / n], float32。各分支偏置。
            实际切片长度为 n，分别对应三个分支。
        n: int。扩展率（流数量）。
        eps: float，默认 1e-5。RMSNorm 数值稳定常数。

    Returns:
        h_pre_raw: [batch_size, seq_len, n]，float32。原始聚合权重，
            需配合 stream_aggregate 使用。
        h_post_raw: [batch_size, seq_len, n]，float32。原始分发权重，
            需配合 stream_distribute 使用。
        h_res_reshaped: [batch_size, seq_len, n, n]，float32。残差混合矩阵，
            需输入 sinkhorn_knopp 生成双随机矩阵。

    Raises:
        AssertionError: M 不是 32 的倍数。
    """
    M = phi.shape[-1]
    assert M % 32 == 0, "输入的 M 必须是 32 的倍数"

    shape = ops.shape(x_norm)
    B, T = shape[0], shape[1]
    h_native = ops.cast(ops.matmul(x_norm, phi), "float32")

    h_res_raw = alpha_res * h_native[..., : n * n] + bias_res
    h_pre_raw = alpha_pre * h_native[..., n * n : n * (n + 1)] + bias_pre
    h_post_raw = alpha_post * h_native[..., n * (n + 1) : n * (n + 2)] + bias_post

    h_res_reshaped = ops.reshape(h_res_raw, (B, T, n, n))
    return (
        h_pre_raw,
        h_post_raw,
        h_res_reshaped,
    )


def mhc_pre_op_fused(
    x,
    h_res_reshaped,
    h_pre_raw,
    num_iters=20,
    eps=1e-8,
):
    """mHC 预处理融合算子（native 版）。

    Args:
        x: [B, T, n, C]，任意实 dtype。
        h_res_reshaped: [B, T, n, n]，任意实 dtype。未归一化残差矩阵。
        h_pre_raw: [B, T, n]，任意实 dtype。未激活聚合权重。
        num_iters: int，默认 20。Sinkhorn-Knopp 轮数。
        eps: float，默认 1e-8。数值稳定常数。

    Returns:
        x_layer_in: [B, T, C]，与 x.dtype 相同。聚合后的层输入。
        H_res: [B, T, n, n]，float32。双随机残差矩阵。
    """
    H_res = sinkhorn_knopp(h_res_reshaped, num_iters, eps)
    x_layer_in = stream_aggregate(x, h_pre_raw)
    return x_layer_in, H_res

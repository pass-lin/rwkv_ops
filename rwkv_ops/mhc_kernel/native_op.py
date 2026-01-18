from keras import ops


def fp32_sigmoid(x):
    dtype = x.dtype
    return ops.cast(ops.nn.sigmoid(ops.cast(x, "float32")), dtype)


def stream_distribute(inp, h_post_raw):
    """
    Distribute (1 -> n): 将单流输出分发回多流。
    对齐精度版：强制在 FP32 下进行广播乘法。

    inp: [B, T, C] (BF16)
    H_post: [B, T, n] (FP32)
    """
    # 1. 记录原始类型
    original_dtype = inp.dtype

    # 2. 提升到 FP32 进行运算 (对齐 CUDA 内核内部的 to_float 逻辑)
    # [B, T, 1, C]
    x_fp32 = ops.cast(ops.expand_dims(inp, -2), "float32")

    # [B, T, n, 1]
    w_fp32 = ops.expand_dims(2.0 * fp32_sigmoid(ops.cast(h_post_raw, "float32")), -1)

    # 3. 执行广播乘法
    # 结果为 [B, T, n, C]
    res_fp32 = x_fp32 * w_fp32

    # 4. 转回原始类型 (对齐 CUDA 内核末尾的 to_bf 逻辑)
    return ops.cast(res_fp32, original_dtype)


def stream_mix(inp, M):
    """
    Mix (n -> n): 残差流之间的线性交互。
    inp: [B, T, n, C]
    M: [B, T, n, n] 或 [n, n] (由 sinkhorn_knopp 生成的方阵)
    """
    # 使用 einsum 表达矩阵乘法：M @ inp
    # i,j 是流索引，k 是通道索引
    dtype = inp.dtype
    inp = ops.cast(inp, M.dtype)
    if len(ops.shape(M)) == 2:
        out = ops.einsum("ij,btjk->btik", M, inp)
    else:
        out = ops.einsum("btij,btjk->btik", M, inp)
    return ops.cast(out, dtype)


def mhc_post_op(layer_out, x_expanded, h_post_raw, H_res):
    """
    mHC 后处理融合算子
    输入:
        layer_out: [B, T, C] - 核心层 (Attention/FFN) 处理后的输出
        x_expanded: [B, T, n, C] - 之前的扩展残差流 (Pre-Op 之前的状态)
        H_post: [B, T, n] - 分发权重 (来自 Pre-Op)，2*sigmoid后的数值
        H_res: [B, T, n, n] - 流混合矩阵 (来自 Pre-Op)
    返回:
        x_next: [B, T, n, C] - 更新后的扩展残差流
    """
    # 精度转化
    dtype = x_expanded.dtype
    layer_out = ops.cast(layer_out, "float32")
    x_expanded = ops.cast(x_expanded, "float32")
    h_post_raw = ops.cast(h_post_raw, "float32")
    H_res = ops.cast(H_res, "float32")
    # 1. 在 FP32 下计算混合路径
    x_mixed_f32 = stream_mix(x_expanded, H_res)

    # 2. 在 FP32 下计算增量路径
    x_delta_f32 = stream_distribute(layer_out, h_post_raw)

    # 3. 在 FP32 下完成最后的残差加法
    x_next_f32 = x_mixed_f32 + x_delta_f32

    # 4. 只在最后输出时进行一次 BF16 转换
    # 这一步对应 CUDA 内核中最后的 to_bf()
    return ops.cast(x_next_f32, dtype)


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    """
    将输入矩阵投影为双拟随机矩阵 (Doubly Stochastic Matrix)。
    通常 inp 是 log 域的矩阵 (H_res_raw)。
    """

    # 转换到 fp32 并应用 exp (论文 Eq. 9 之前的步骤)
    x = ops.cast(inp, "float32")
    # 防溢出技巧：减去最大值
    x = x - ops.max(x, axis=(-1, -2), keepdims=True)
    P = ops.exp(x)

    for _ in range(num_iters):
        # 行归一化
        P = P / (ops.sum(P, axis=-1, keepdims=True) + eps)
        # 列归一化
        P = P / (ops.sum(P, axis=-2, keepdims=True) + eps)

    return P


def mhc_rmsnorm(inp, eps=1e-5):
    """
    标准 RMSNorm 算子。
    inp: [..., C], weight: [C]
    """
    dtype = inp.dtype
    x = ops.cast(inp, "float32")
    # 计算均方根
    rms = ops.sqrt(ops.mean(ops.square(x), axis=-1, keepdims=True) + eps)
    x_normed = x / rms
    # 应用权重
    return ops.cast(x_normed, dtype)


def stream_aggregate(inp, H_pre):
    # 1. 转换为 float32 进行高精度计算
    inp_f32 = ops.cast(inp, "float32")
    H_f32 = ops.cast(H_pre, "float32")
    H_f32 = fp32_sigmoid(H_f32)
    # 2. 在 float32 空间完成乘法和累加
    out_f32 = ops.sum(inp_f32 * ops.expand_dims(H_f32, -1), axis=-2)

    # 3. 最后转回原先的格式 (如 bf16)
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
    """
    mHC 动态投影与分支生成算子（底层实现）

    功能：将多流特征 x 通过线性投影生成三组混合系数，用于后续的聚合/分发/残差操作。
    实现：RMSNorm → Linear → 动态拆分为三个分支 → 可选激活

    --- 参数 ---
    x: [batch_size, seq_len, n * hidden_size], BF16/BF16
        展平后的多流特征（n 个并行流，每个维度为 hidden_size）

    alpha_pre, alpha_post, alpha_res: (1,), float32
        三个标量系数，分别控制 H_pre、H_post、H_res 分支的缩放强度

    phi: [n * hidden_size, M], BF16
        投影矩阵，M = n * (n + 2)（包含残差/前/后三个分支的拼接维度）
        **注意**：若 M < 32，CUDA 内核会自动填充到 32 的倍数以保证效率

    bias: [M], float32
        线性层偏置项

    n: int
        扩展率（流数量），用于将输出 reshap 为 n×n 矩阵

    eps: float, default=1e-5
        RMSNorm 的数值稳定常数

    --- 返回 ---
    h_pre_raw: [batch_size, seq_len, n], 原输入 dtype
        原始聚合权重（未激活），需配合 stream_aggregate 使用

    H_post: [batch_size, seq_len, n], 原输入 dtype
        分发权重，已应用 2 * sigmoid 激活，取值范围 (0, 2)

    h_res_reshaped: [batch_size, seq_len, n, n], 原输入 dtype
        残差混合矩阵（未归一化），需输入 Sinkhorn-Knopp 生成双随机矩阵
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
    H_res = sinkhorn_knopp(h_res_reshaped, num_iters, eps)
    x_layer_in = stream_aggregate(x, h_pre_raw)
    return x_layer_in, H_res

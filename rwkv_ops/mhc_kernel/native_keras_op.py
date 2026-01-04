import keras
from keras import ops

# --- 辅助函数：确保在 fp32 下计算以保证数值稳定性 ---


@keras.remat
def fp32_sigmoid(x):
    dtype = x.dtype
    return ops.cast(ops.nn.sigmoid(ops.cast(x, "float32")), dtype)


# --- 核心 MHC 算子 ---


def sinkhorn_knopp(inp, num_iters=20, eps=1e-8):
    """
    将输入矩阵投影为双拟随机矩阵 (Doubly Stochastic Matrix)。
    通常 inp 是 log 域的矩阵 (H_res_raw)。
    """
    dtype = inp.dtype
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

    return ops.cast(P, dtype)


def rmsnorm(inp, eps=1e-5):
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
    
    # 2. 在 float32 空间完成乘法和累加
    out_f32 = ops.sum(inp_f32 * ops.expand_dims(H_f32, -1), axis=-2)
    
    # 3. 最后转回原先的格式 (如 bf16)
    return ops.cast(out_f32, inp.dtype)


def stream_distribute(inp, H_post, n=0):
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
    w_fp32 = ops.cast(ops.expand_dims(H_post, -1), "float32")
    
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

"""Gated DeltaNet recurrent Keras 参考实现。"""

import keras
from keras import ops


def _l2norm(x, axis=-1, eps=1e-6):
    """对指定轴做 L2 归一化。

    Args:
        x: 任意形状张量。
        axis: 归一化轴，默认最后一维。
        eps: 防止除零的小常数。

    Returns:
        与 x 同形状，沿 axis 模长为 1 的张量。
    """
    inv_norm = ops.rsqrt(ops.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _gdn_recurrent_step(t, inputs, q, k, v, g, beta, scale, B, H, V, DTYPE):
    """Gated DeltaNet 单步递推（供 fori_loop 使用）。

    Args:
        t: 当前时间步索引。
        inputs: [state, out] 元组，out 为各后端统一的输出占位。
        q, k, v, g, beta: [B, H, T, *] 的 float32 张量。
        scale: query 缩放系数。
        B, H, V: 形状常量。
        DTYPE: 输出 dtype。

    Returns:
        [new_state, new_out] 元组。
    """
    state, out = inputs[0], inputs[1]
    backend = keras.config.backend()

    q_t = q[:, :, t]
    k_t = k[:, :, t]
    v_t = v[:, :, t]
    g_t = g[:, :, t]
    beta_t = beta[:, :, t]

    # state: [B, H, K, V]
    state = state * ops.expand_dims(ops.expand_dims(ops.exp(g_t), -1), -1)

    # kv_mem = state @ k_t = sum_K state * k_t
    kv_mem = ops.sum(state * ops.expand_dims(k_t, -1), axis=-2)

    # delta = beta_t * (v_t - kv_mem)
    delta = (v_t - kv_mem) * ops.expand_dims(beta_t, -1)

    # state += k_t^T delta
    state = state + ops.expand_dims(k_t, -1) * ops.expand_dims(delta, -2)

    # out_t = state @ q_t = sum_K state * q_t
    out_t = ops.sum(state * ops.expand_dims(q_t * scale, -1), axis=-2)

    if backend == "tensorflow":
        out = out.write(t, ops.cast(out_t, DTYPE))
    elif backend == "torch":
        out[:, t : t + 1] = ops.reshape(ops.cast(out_t, DTYPE), (B, 1, H, V))
    else:
        out = ops.slice_update(
            out, [0, t, 0, 0], ops.reshape(ops.cast(out_t, DTYPE), (B, 1, H, V))
        )

    return [state, out]


def gated_delta_net_recurrent(
    q, k, v, g, beta, initial_state=None, output_final_state=False
):
    """Gated DeltaNet 串行 recurrent 原生实现。

    按时间步逐步更新 state，是 chunkwise 算法的数学基准。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，需已落在 (0,1) 内（外部 sigmoid）。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    DTYPE = v.dtype

    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))
    g = ops.transpose(g, (0, 2, 1))
    beta = ops.transpose(beta, (0, 2, 1))

    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    v = ops.cast(v, "float32")
    g = ops.cast(g, "float32")
    beta = ops.cast(beta, "float32")

    q = _l2norm(q, axis=-1)
    k = _l2norm(k, axis=-1)

    B = ops.shape(q)[0]
    H = ops.shape(q)[1]
    T = ops.shape(q)[2]
    K = ops.shape(q)[3]
    V = ops.shape(v)[3]
    scale = 1.0 / (ops.cast(K, "float32") ** 0.5)

    if initial_state is None:
        state = ops.zeros((B, H, K, V), dtype="float32")
    else:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, K, V))

    backend = keras.config.backend()
    if backend == "tensorflow":
        import tensorflow as tf

        out = tf.TensorArray(DTYPE, size=T)
    else:
        out = ops.zeros((B, T, H, V), dtype=DTYPE)

    state, out = ops.fori_loop(
        0,
        T,
        lambda t, inputs: _gdn_recurrent_step(
            t, inputs, q, k, v, g, beta, scale, B, H, V, DTYPE
        ),
        [state, out],
    )

    if backend == "tensorflow":
        out = ops.transpose(out.stack(), (1, 0, 2, 3))

    out = ops.cast(out, DTYPE)

    if output_final_state:
        return out, state
    return out, None


def gated_delta_net_reference(
    q, k, v, g, beta, initial_state=None, output_final_state=False
):
    """Gated DeltaNet 最简 recurrent 黄金参考实现。

    与 `gated_delta_net_recurrent` 数学等价，但用最直接的 Python 列表+
    ops.stack 组织，不依赖 fori_loop，用于验证 recurrent/chunkwise 的正确性。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，需已落在 (0,1) 内。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    DTYPE = v.dtype

    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))
    g = ops.transpose(g, (0, 2, 1))
    beta = ops.transpose(beta, (0, 2, 1))

    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    v = ops.cast(v, "float32")
    g = ops.cast(g, "float32")
    beta = ops.cast(beta, "float32")

    q = _l2norm(q, axis=-1)
    k = _l2norm(k, axis=-1)

    B = ops.shape(q)[0]
    H = ops.shape(q)[1]
    T = ops.shape(q)[2]
    K = ops.shape(q)[3]
    V = ops.shape(v)[3]
    scale = 1.0 / (ops.cast(K, "float32") ** 0.5)

    if initial_state is None:
        state = ops.zeros((B, H, K, V), dtype="float32")
    else:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, K, V))

    outputs = []
    for t in range(T):
        q_t = q[:, :, t]
        k_t = k[:, :, t]
        v_t = v[:, :, t]
        g_t = g[:, :, t]
        beta_t = beta[:, :, t]

        state = state * ops.expand_dims(ops.expand_dims(ops.exp(g_t), -1), -1)
        kv_mem = ops.sum(state * ops.expand_dims(k_t, -1), axis=-2)
        delta = (v_t - kv_mem) * ops.expand_dims(beta_t, -1)
        state = state + ops.expand_dims(k_t, -1) * ops.expand_dims(delta, -2)
        out_t = ops.sum(state * ops.expand_dims(q_t * scale, -1), axis=-2)
        outputs.append(out_t)

    out = ops.stack(outputs, axis=2)
    out = ops.transpose(out, (0, 2, 1, 3))
    out = ops.cast(out, DTYPE)

    if output_final_state:
        return out, state
    return out, None


def gated_delta_net_recurrent_inference(
    q, k, v, g, beta, initial_state=None, output_final_state=True
):
    """Gated DeltaNet recurrent 推理原生封装（无梯度）。

    当前阶段与 `gated_delta_net_recurrent` 数学等价，仅用于保持 API 一致性。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，需已落在 (0,1) 内。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    return gated_delta_net_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=output_final_state,
    )


def gated_delta_net_recurrent_single_step(
    q, k, v, g, beta, initial_state=None, output_final_state=True
):
    """Gated DeltaNet recurrent 单步 RNN 原生实现。

    Args:
        q: [B, H, K]，查询。
        k: [B, H, K]，键。
        v: [B, H, V]，值。
        g: [B, H]，decay gate（对数空间）。
        beta: [B, H]，写入强度门控，需已落在 (0,1) 内。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回 next state。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    q = ops.expand_dims(q, axis=1)
    k = ops.expand_dims(k, axis=1)
    v = ops.expand_dims(v, axis=1)
    g = ops.expand_dims(g, axis=1)
    beta = ops.expand_dims(beta, axis=1)

    out, state = gated_delta_net_recurrent(
        q, k, v, g, beta, initial_state=initial_state, output_final_state=True
    )

    out = ops.squeeze(out, axis=1)
    if output_final_state:
        return out, state
    return out, None

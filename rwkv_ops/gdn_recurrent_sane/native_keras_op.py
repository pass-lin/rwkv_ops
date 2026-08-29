"""Gated DeltaNet recurrent SANE Keras 参考实现。"""

import warnings

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


def _apply_state_norm_cond(state, t, tau, mask, chunk_size):
    """在 chunk 边界按 mask 执行 State Anomaly Neutralization。"""
    is_boundary = ops.equal(ops.mod(t + 1, chunk_size), 0)

    def _true_fn():
        chunk_idx = ops.maximum((t + 1) // chunk_size - 1, 0)
        chunk_idx = ops.reshape(chunk_idx, [1])

        tau_t = ops.take(tau, chunk_idx, axis=2)
        tau_t = ops.squeeze(tau_t, axis=2)

        mask_t = ops.take(mask, chunk_idx, axis=1)
        mask_t = ops.squeeze(mask_t, axis=1)

        m = ops.cast(ops.greater(mask_t, 0), state.dtype)
        m = ops.reshape(m, [ops.shape(m)[0], 1, 1, 1])

        tau_calc = ops.maximum(tau_t, 1e-6)
        tau_calc = ops.reshape(
            tau_calc,
            [ops.shape(tau_calc)[0], ops.shape(tau_calc)[1], 1, 1],
        )
        sane_state = tau_calc * ops.tanh(state / tau_calc)

        return state * (1.0 - m) + sane_state * m

    def _false_fn():
        return state

    return ops.cond(is_boundary, _true_fn, _false_fn)


def _apply_state_norm_uncond(state, t, tau, chunk_size):
    """在 chunk 边界无条件执行 State Anomaly Neutralization。"""
    is_boundary = ops.equal(ops.mod(t + 1, chunk_size), 0)

    def _true_fn():
        chunk_idx = ops.maximum((t + 1) // chunk_size - 1, 0)
        chunk_idx = ops.reshape(chunk_idx, [1])

        tau_t = ops.take(tau, chunk_idx, axis=2)
        tau_t = ops.squeeze(tau_t, axis=2)

        tau_calc = ops.maximum(tau_t, 1e-6)
        tau_calc = ops.reshape(
            tau_calc,
            [ops.shape(tau_calc)[0], ops.shape(tau_calc)[1], 1, 1],
        )
        return tau_calc * ops.tanh(state / tau_calc)

    def _false_fn():
        return state

    return ops.cond(is_boundary, _true_fn, _false_fn)


def _gdn_recurrent_sane_step(
    t,
    inputs,
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask,
    scale,
    B,
    H,
    V,
    DTYPE,
    chunk_size,
    use_mask,
):
    """Gated DeltaNet recurrent SANE 单步递推（供 fori_loop 使用）。

    Args:
        t: 当前时间步索引。
        inputs: [state, out] 元组。
        q, k, v, g, beta: [B, H, T, *] 的 float32 张量。
        tau: [B, H, T//chunk_size]，float32。
        mask: [B, T//chunk_size]，float32 或 None。
        scale: query 缩放系数。
        B, H, V: 形状常量。
        DTYPE: 输出 dtype。
        chunk_size: chunk 长度。
        use_mask: bool，是否使用 mask。

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

    state = state * ops.expand_dims(ops.expand_dims(ops.exp(g_t), -1), -1)

    kv_mem = ops.sum(state * ops.expand_dims(k_t, -1), axis=-2)

    delta = (v_t - kv_mem) * ops.expand_dims(beta_t, -1)

    state = state + ops.expand_dims(k_t, -1) * ops.expand_dims(delta, -2)

    out_t = ops.sum(state * ops.expand_dims(q_t * scale, -1), axis=-2)

    if backend == "tensorflow":
        out = out.write(t, ops.cast(out_t, DTYPE))
    elif backend == "torch":
        out[:, t : t + 1] = ops.reshape(ops.cast(out_t, DTYPE), (B, 1, H, V))
    else:
        out = ops.slice_update(
            out, [0, t, 0, 0], ops.reshape(ops.cast(out_t, DTYPE), (B, 1, H, V))
        )

    if use_mask:
        state = _apply_state_norm_cond(state, t, tau, mask, chunk_size)
    else:
        state = _apply_state_norm_uncond(state, t, tau, chunk_size)

    return [state, out]


def gated_delta_net_recurrent_sane(
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=False,
    head_first=False,
    chunk_size=16,
):
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 原生实现。

    按时间步逐步更新 state，在 chunk 边界（每 chunk_size 个 token）按 mask 对
    state 执行 `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，需已落在 (0,1) 内（外部 sigmoid）。
        tau: [B, T//chunk_size, H]，float32。阈值，必须 > 1。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先 ([B, H, T, *])。
        chunk_size: int，SANE  chunk 长度，默认 16。

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
    tau = ops.transpose(tau, (0, 2, 1))

    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    v = ops.cast(v, "float32")
    g = ops.cast(g, "float32")
    beta = ops.cast(beta, "float32")
    tau = ops.cast(tau, "float32")

    q = _l2norm(q, axis=-1)
    k = _l2norm(k, axis=-1)

    B = ops.shape(q)[0]
    H = ops.shape(q)[1]
    T = ops.shape(q)[2]
    K = ops.shape(q)[3]
    V = ops.shape(v)[3]
    scale = 1.0 / (ops.cast(K, "float32") ** 0.5)

    use_mask = output_final_state and mask is not None
    if use_mask:
        mask = ops.cast(mask, "float32")

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
        lambda t, inputs: _gdn_recurrent_sane_step(
            t,
            inputs,
            q,
            k,
            v,
            g,
            beta,
            tau,
            mask,
            scale,
            B,
            H,
            V,
            DTYPE,
            chunk_size,
            use_mask,
        ),
        [state, out],
    )

    if backend == "tensorflow":
        out = ops.transpose(out.stack(), (1, 0, 2, 3))

    out = ops.cast(out, DTYPE)

    if not output_final_state:
        return out, None

    if mask is None:
        warnings.warn(
            "[gdn_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[gdn_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return out, state


def gated_delta_net_recurrent_sane_inference(
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=True,
    head_first=False,
    chunk_size=16,
):
    """Gated DeltaNet recurrent SANE 推理原生封装（无梯度）。

    当前阶段与 `gated_delta_net_recurrent_sane` 数学等价，仅用于保持 API 一致性。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，需已落在 (0,1) 内。
        tau: [B, T//chunk_size, H]，float32。
        mask: [B, T//chunk_size]，float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先。
        chunk_size: int，SANE chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    return gated_delta_net_recurrent_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=initial_state,
        output_final_state=output_final_state,
        head_first=head_first,
        chunk_size=chunk_size,
    )


def _gdn_recurrent_sane_single_step_core(q, k, v, g, beta, tau, do_sane, state, scale):
    """Gated DeltaNet recurrent SANE 单步核心。

    Args:
        q, k: [B, H, K]，查询与键。
        v: [B, H, V]，值。
        g, beta: [B, H]，gate。
        tau: [B, H]，float32。
        do_sane: [B]，bool。
        state: [B, H, K, V]，float32。
        scale: float。

    Returns:
        out: [B, H, V]。
        next_state: [B, H, K, V]。
    """
    B = ops.shape(state)[0]
    H = ops.shape(state)[1]

    state = state * ops.expand_dims(ops.expand_dims(ops.exp(g), -1), -1)

    kv_mem = ops.sum(state * ops.expand_dims(k, -1), axis=-2)

    delta = (v - kv_mem) * ops.expand_dims(beta, -1)

    state = state + ops.expand_dims(k, -1) * ops.expand_dims(delta, -2)

    out = ops.sum(state * ops.expand_dims(q * scale, -1), axis=-2)

    tau_safe = ops.maximum(tau, 1e-6)
    tau_safe = ops.reshape(tau_safe, [B, H, 1, 1])
    sane_state = tau_safe * ops.tanh(state / tau_safe)

    do_sane_f = ops.cast(do_sane, state.dtype)
    if len(ops.shape(do_sane_f)) == 0:
        do_sane_f = ops.broadcast_to(do_sane_f, (B,))
    do_sane_f = ops.reshape(do_sane_f, [B, 1, 1, 1])

    state = ops.where(do_sane_f > 0.0, sane_state, state)

    return out, state


def gated_delta_net_recurrent_sane_single_step(
    q,
    k,
    v,
    g,
    beta,
    tau,
    do_sane,
    initial_state=None,
    output_final_state=True,
    head_first=False,
):
    """Gated DeltaNet recurrent SANE 单步 RNN 原生实现。

    Args:
        q: [B, H, K]，查询。
        k: [B, H, K]，键。
        v: [B, H, V]，值。
        g: [B, H]，decay gate（对数空间）。
        beta: [B, H]，写入强度门控，需已落在 (0,1) 内。
        tau: [B, H]，float32。
        do_sane: [B]，bool，是否在该步执行 SANE。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回 next state。
        head_first: bool，输入输出是否 head 维优先（单步下无时间维，此参数保留为兼容）。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    DTYPE = v.dtype

    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    v = ops.cast(v, "float32")
    g = ops.cast(g, "float32")
    beta = ops.cast(beta, "float32")
    tau = ops.cast(tau, "float32")

    q = _l2norm(q, axis=-1)
    k = _l2norm(k, axis=-1)

    B = ops.shape(q)[0]
    H = ops.shape(q)[1]
    K = ops.shape(q)[2]
    scale = 1.0 / (ops.cast(K, "float32") ** 0.5)

    V = ops.shape(v)[2]

    if initial_state is None:
        state = ops.zeros((B, H, K, V), dtype="float32")
    else:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, K, V))

    out, state = _gdn_recurrent_sane_single_step_core(
        q, k, v, g, beta, tau, do_sane, state, scale
    )

    out = ops.cast(out, DTYPE)
    if output_final_state:
        return out, state
    return out, None

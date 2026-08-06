"""RWKV-7 State Anomaly Neutralization 原生 Keras 参考实现。"""

import warnings

import keras
from keras import ops


def transpose_head(x, head_first):
    """将输入统一转置为 [B, H, T, N] 并转为 float32。"""
    x = ops.cast(x, "float32")
    if head_first:
        return x
    return ops.transpose(x, (0, 2, 1, 3))


def _apply_state_norm_cond(state, t, tau, mask):
    """训练用：只在 chunk 边界按 mask 执行 State Anomaly Neutralization。"""
    is_boundary = ops.equal(ops.mod(t + 1, 16), 0)

    def _true_fn():
        chunk_idx = ops.maximum((t + 1) // 16 - 1, 0)
        chunk_idx = ops.reshape(chunk_idx, [1])

        tau_t = ops.take(tau, chunk_idx, axis=1)
        tau_t = ops.squeeze(tau_t, axis=1)

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


def _apply_state_norm_uncond(state, t, tau):
    """训练用：在 chunk 边界无条件执行 State Anomaly Neutralization。"""
    is_boundary = ops.equal(ops.mod(t + 1, 16), 0)

    def _true_fn():
        chunk_idx = ops.maximum((t + 1) // 16 - 1, 0)
        chunk_idx = ops.reshape(chunk_idx, [1])

        tau_t = ops.take(tau, chunk_idx, axis=1)
        tau_t = ops.squeeze(tau_t, axis=1)

        tau_calc = ops.maximum(tau_t, 1e-6)
        tau_calc = ops.reshape(
            tau_calc,
            [ops.shape(tau_calc)[0], ops.shape(tau_calc)[1], 1, 1],
        )
        return tau_calc * ops.tanh(state / tau_calc)

    def _false_fn():
        return state

    return ops.cond(is_boundary, _true_fn, _false_fn)


def generalized_delta_rule_sane(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=True,
    head_first=False,
):
    """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（chunkwise 训练版）。

    在 chunk 边界（每 16 个 token）按 mask 对 state 执行
    `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        r, w, k, v, a, b: [B, T, H, K]，bfloat16。T 必须被 16 整除。
        tau: [B, T//16, H]，float32。阈值，必须严格 > 1。
        mask: [B, T//16]，float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先 ([B, H, T, K])。

    Returns:
        out: [B, T, H, K]，与输入同 dtype。
        final_state: [B, H, K, K]，float32。
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 16 整除，或 tau/mask 形状不匹配。

    Examples:
        >>> y, state = generalized_delta_rule_sane(
        ...     r, w, k, v, a, b, tau, mask, initial_state=h0)
    """
    DTYPE = r.dtype

    r = transpose_head(r, head_first)
    k = transpose_head(k, head_first)
    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))

    B, H, T, N = ops.shape(r)

    if ops.mod(T, 16) != 0:
        raise ValueError(
            f"RWKV-SANE training/prefill requires T divisible by 16, but got T={T}."
        )

    tau = ops.cast(tau, "float32")

    # 当且仅当需要 final_state 且用户显式提供了 mask 时才使用带 mask 算子。
    use_mask = output_final_state and mask is not None

    if use_mask:
        mask = ops.cast(mask, "float32")
        if ops.shape(mask) != (B, T // 16):
            raise ValueError(
                f"mask shape {ops.shape(mask)} must match (B, T//16) = ({B}, {T // 16})"
            )
    elif mask is not None and output_final_state:
        # mask 被显式提供但 output_final_state=False：为节省算力将忽略 mask。
        pass

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N))
    state = ops.cast(state, "float32")

    keras_backend = keras.config.backend()

    apply_state_norm = _apply_state_norm_cond if use_mask else _apply_state_norm_uncond

    def step(t, inputs):
        state, out = inputs

        rr = ops.reshape(r[:, :, t, :], (B, H, N, 1))
        kk = ops.reshape(k[:, :, t, :], (B, H, 1, N))
        vv = ops.reshape(v[:, :, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, :, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, :, t, :], (B, H, 1, N))

        state = state * w[:, :, t, None, :] + state @ aa @ bb + vv @ kk

        o = ops.cast(state @ rr, out.dtype)
        o = ops.squeeze(o, axis=-1)

        if keras_backend == "tensorflow":
            out = out.write(t, o)
        elif keras_backend == "torch":
            out[:, :, t : t + 1] = ops.reshape(o, (B, H, 1, N))
        else:
            out = ops.slice_update(out, [0, 0, t, 0], ops.reshape(o, (B, H, 1, N)))

        if use_mask:
            state = apply_state_norm(state, t, tau, mask)
        else:
            state = apply_state_norm(state, t, tau)
        return [state, out]

    if keras_backend == "tensorflow":
        import tensorflow as tf

        out = tf.TensorArray(DTYPE, size=T)
    else:
        out = ops.zeros((B, H, T, N), DTYPE)

    state, out = ops.fori_loop(0, T, step, [state, out])

    # 与 CUDA 后端保持一致：输出统一为 [B, T, H, N]（时间步优先）。
    out = ops.transpose(out, (0, 2, 1, 3))

    out = ops.cast(out, DTYPE)
    if not output_final_state:
        return out

    if mask is None:
        warnings.warn(
            "[rwkv7_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[rwkv7_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return out, state


def rwkv7_step_sane(
    r,
    w,
    k,
    v,
    a,
    b,
    state,
    tau,
    do_sane,
):
    """RWKV-7 单步推理（RNN 模式），带 State Anomaly Neutralization。

    Args:
        r, w, k, v, a, b: [B, H, N]，单步输入，已 head-first。
        state: [B, H, N, N]，float32。
        tau: [B, H] 或 [H]，float32，必须 > 0。
        do_sane: [B]，bool。

    Returns:
        o: [B, H, N]。
        state_out: [B, H, N, N]，float32。
    """
    DTYPE = state.dtype
    B = ops.shape(state)[0]
    H = ops.shape(state)[1]

    rr = ops.expand_dims(r, axis=-1)
    kk = ops.expand_dims(k, axis=-2)
    vv = ops.expand_dims(v, axis=-1)
    aa = ops.expand_dims(a, axis=-1)
    bb = ops.expand_dims(b, axis=-2)

    w_decay = ops.exp(-ops.exp(w))
    w_decay = ops.expand_dims(w_decay, axis=-2)

    new_state = state * w_decay + state @ aa @ bb + vv @ kk

    o = ops.cast(new_state @ rr, DTYPE)
    o = ops.squeeze(o, axis=-1)

    tau_safe = ops.maximum(tau, 1e-6)
    tau_safe = ops.reshape(tau_safe, [B, H, 1, 1])
    sane_state = tau_safe * ops.tanh(new_state / tau_safe)

    do_sane_f = ops.cast(do_sane, state.dtype)
    if len(ops.shape(do_sane_f)) == 0:
        do_sane_f = ops.broadcast_to(do_sane_f, (B,))
    do_sane_f = ops.reshape(do_sane_f, [B, 1, 1, 1])

    state_out = ops.where(do_sane_f > 0.0, sane_state, new_state)

    return o, state_out


def generalized_delta_rule_sane_single_step(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,
    do_sane,
    initial_state=None,
    output_final_state=True,
    head_first=False,
):
    """带 State Anomaly Neutralization 的 RWKV-7 单步推理（native 入口）。

    Args:
        r, w, k, v, a, b: [B, 1, H, N]（head_first=False）或 [B, H, 1, N]（head_first=True）。
        tau: [B, H]，float32，必须 > 0。
        do_sane: [B]，bool。
        initial_state: [B, H, N, N] 或 [1, H, N, N]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先。

    Returns:
        out: [B, 1, H, N]（或 [B, H, 1, N]），与输入同 dtype。
        final_state: [B, H, N, N]，float32。output_final_state=False 时不返回。

    Raises:
        ValueError: 时间维不为 1。
    """
    DTYPE = r.dtype
    time_axis = 2 if head_first else 1
    if r.shape[time_axis] != 1:
        raise ValueError(
            f"Single-step kernel requires time dimension = 1, but got shape {r.shape}."
        )

    if head_first:
        r = ops.squeeze(r, axis=2)
        w = ops.squeeze(w, axis=2)
        k = ops.squeeze(k, axis=2)
        v = ops.squeeze(v, axis=2)
        a = ops.squeeze(a, axis=2)
        b = ops.squeeze(b, axis=2)
    else:
        r = ops.squeeze(r, axis=1)
        w = ops.squeeze(w, axis=1)
        k = ops.squeeze(k, axis=1)
        v = ops.squeeze(v, axis=1)
        a = ops.squeeze(a, axis=1)
        b = ops.squeeze(b, axis=1)

    B, H, N = ops.shape(r)

    if initial_state is None:
        state = ops.zeros((B, H, N, N), "float32")
    else:
        state = ops.cast(initial_state, "float32")
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))

    o, state_out = rwkv7_step_sane(
        r=ops.cast(r, "float32"),
        w=ops.cast(w, "float32"),
        k=ops.cast(k, "float32"),
        v=ops.cast(v, "float32"),
        a=ops.cast(a, "float32"),
        b=ops.cast(b, "float32"),
        state=state,
        tau=ops.cast(tau, "float32"),
        do_sane=ops.cast(do_sane, "bool"),
    )

    out = ops.expand_dims(o, axis=1)
    out = ops.cast(out, DTYPE)
    if output_final_state:
        return out, state_out
    return out

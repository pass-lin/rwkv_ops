"""
RWKV-7 State Norm (Adaptive Tanh Clipping) — 半线性 Chunkwise 训练与单步推理

设计要点：
- 训练时只在 chunk 边界（每 16 token）执行 State Norm，使用 ops.cond 避免非边界
  步的冗余计算与梯度传播。
- 单步推理时每步都计算 SN，再用 ops.where 根据 per-sample mask 选择。
- `tau` 仅作为 State Norm 的阈值（必须 > 0）；是否执行 SN 由同 batch-chunk 维度的
  `mask` 决定，形状为 [B, T//16]。
- 不再内置 token-level mask。调用者需在外部保证 padding 位置 k=0, a=0, w=-inf，
  并把全 padding chunk 的 mask 置 0。
"""

import keras
from keras import ops


def transpose_head(x, head_first):
    """
    将输入张量统一转置为 [B, H, T, N] 并转为 float32。

    参数:
        x:          [B, T, H, N]（head_first=False）或 [B, H, T, N]（head_first=True）
        head_first: True 表示输入已是 [B, H, T, N]，无需转置

    返回:
        [B, H, T, N]，float32
    """
    x = ops.cast(x, "float32")
    if head_first:
        return x
    return ops.transpose(x, (0, 2, 1, 3))


def _apply_state_norm_cond(state, t, tau, mask):
    """
    训练用：只在 chunk 边界执行 State Norm，使用 ops.cond 避免冗余计算。

    参数:
        state: [B, H, N, N]，float32
        t:     scalar Tensor，当前 token 索引
        tau:   [B, T//16, H]，float32，必须 > 0
        mask:  [B, T//16]，float32/bool，>0 表示执行 SN

    返回:
        [B, H, N, N]
    """
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
        sn_state = tau_calc * ops.tanh(state / tau_calc)

        return state * (1.0 - m) + sn_state * m

    def _false_fn():
        return state

    return ops.cond(is_boundary, _true_fn, _false_fn)


def generalized_delta_rule_sn(
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
    """
    带 State Norm 的 RWKV-7 广义 Delta 规则（Chunkwise 训练版本）。

    说明：
    - 训练版本会保存反向传播所需的中间量；纯推理请使用后端对应的 inference
      入口，可减少显存占用。
    - T 必须被 16 整除（tau/mask 是 per-chunk）。
    - `tau` 只表示阈值，必须严格 > 0；是否执行 SN 由 `mask` 决定。

    参数:
        r, w, k, v, a, b:
            [B, T, H, N]（head_first=False）或 [B, H, T, N]（head_first=True）。
            T 必须被 16 整除。
        tau:
            [B, T//16, H]，float32。阈值，必须 > 0。
        mask:
            [B, T//16]，float32/bool。>0 表示该 chunk 边界执行 SN。
            若未提供，默认全 1（所有 chunk 边界都执行 SN）。
        initial_state:
            [B, H, N, N] 或 [1, H, N, N]，可选。
        output_final_state:
            bool，是否返回最终 State。
        head_first:
            bool，输入输出是否 head 维度优先。

    返回:
        out 或 (out, final_state)。
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
            f"RWKV-SN training/prefill requires T divisible by 16, but got T={T}."
        )

    tau = ops.cast(tau, "float32")
    if mask is None:
        mask = ops.ones((B, T // 16), dtype="float32")
    else:
        mask = ops.cast(mask, "float32")
        if ops.shape(mask) != (B, T // 16):
            raise ValueError(
                f"mask shape {ops.shape(mask)} must match (B, T//16) = ({B}, {T // 16})"
            )

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N))
    state = ops.cast(state, "float32")

    keras_backend = keras.config.backend()

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

        state = _apply_state_norm_cond(state, t, tau, mask)
        return [state, out]

    if keras_backend == "tensorflow":
        import tensorflow as tf

        out = tf.TensorArray(DTYPE, size=T)
    else:
        out = ops.zeros((B, H, T, N), DTYPE)

    state, out = ops.fori_loop(0, T, step, [state, out])

    if keras_backend == "tensorflow":
        out = ops.transpose(out.stack(), (1, 0, 2, 3))

    # 与 CUDA 后端保持一致：输出统一为 [B, T, H, N]（时间步优先）。
    out = ops.transpose(out, (0, 2, 1, 3))

    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)


def rwkv7_step_sn(
    r,
    w,
    k,
    v,
    a,
    b,
    state,
    tau,
    do_sn,
):
    """
    RWKV-7 单步推理（RNN 模式），带 State Norm。

    参数:
        r, w, k, v, a, b:
            [B, H, N]，单步输入，已 head-first。
        state:
            [B, H, N, N]，float32。
        tau:
            [B, H] 或 [H]，float32，必须 > 0。
        do_sn:
            [B]，bool。

    返回:
        o:         [B, H, N]
        state_out: [B, H, N, N]
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
    sn_state = tau_safe * ops.tanh(new_state / tau_safe)

    do_sn_f = ops.cast(do_sn, state.dtype)
    if len(ops.shape(do_sn_f)) == 0:
        do_sn_f = ops.broadcast_to(do_sn_f, (B,))
    do_sn_f = ops.reshape(do_sn_f, [B, 1, 1, 1])

    state_out = ops.where(do_sn_f > 0.0, sn_state, new_state)

    return o, state_out


def generalized_delta_rule_sn_single_step(
    r,
    w,
    k,
    v,
    a,
    b,
    tau,
    do_sn,
    initial_state=None,
    output_final_state=True,
    head_first=False,
):
    """
    带 State Norm 的 RWKV-7 单步推理（native 入口）。

    参数:
        r, w, k, v, a, b:
            [B, 1, H, N]（head_first=False）或 [B, H, 1, N]（head_first=True）。
        tau:
            [B, H]，float32，必须 > 0。
        do_sn:
            [B]，bool。
        initial_state, output_final_state, head_first: 同训练版本。

    返回:
        out 或 (out, final_state)。
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

    o, state_out = rwkv7_step_sn(
        r=ops.cast(r, "float32"),
        w=ops.cast(w, "float32"),
        k=ops.cast(k, "float32"),
        v=ops.cast(v, "float32"),
        a=ops.cast(a, "float32"),
        b=ops.cast(b, "float32"),
        state=state,
        tau=ops.cast(tau, "float32"),
        do_sn=ops.cast(do_sn, "bool"),
    )

    out = ops.expand_dims(o, axis=1)
    out = ops.cast(out, DTYPE)
    if output_final_state:
        return out, state_out
    return out

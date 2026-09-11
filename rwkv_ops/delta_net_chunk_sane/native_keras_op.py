"""DeltaNet chunkwise SANE Keras 参考实现。"""

import warnings

from keras import ops


def _l2norm(x, axis=-1, eps=1e-6):
    """对指定轴做 L2 归一化。"""
    inv_norm = ops.rsqrt(ops.sum(x * x, axis=axis, keepdims=True) + eps)
    return x * inv_norm


def _apply_sane(state, tau, mask):
    """对 state 应用 State Anomaly Neutralization 并按 mask blend。

    Args:
        state: [B, H, K, V], float32。
        tau: [B, H], float32。
        mask: [B], float32 或 None。

    Returns:
        [B, H, K, V], float32。
    """
    tau_safe = ops.maximum(tau, 1e-6)
    tau_safe = ops.reshape(
        tau_safe, (ops.shape(tau_safe)[0], ops.shape(tau_safe)[1], 1, 1)
    )
    sane_state = tau_safe * ops.tanh(state / tau_safe)
    if mask is None:
        return sane_state
    m = ops.cast(ops.greater(mask, 0), state.dtype)
    m = ops.reshape(m, (ops.shape(m)[0], 1, 1, 1))
    return state * (1.0 - m) + sane_state * m


def delta_net_chunk_sane(
    q,
    k,
    v,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """带 State Anomaly Neutralization 的 DeltaNet chunkwise 原生实现。

    在每个 chunk 边界对跨 chunk 传递的 state 执行
    `state = tau * tanh(state / tau)`，并按 mask 选择是否执行。
    输出始终基于 SANE 之前的 state（即使用上一 chunk SANE 后的状态作为
    本 chunk 初始 state，但本 chunk 内部不再额外 SANE）。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        beta: [B, T, H]，写入强度门控，必须已在外部过 sigmoid 并落在 (0,1)。
        tau: [B, T//chunk_size, H]，float32。阈值，必须 > 1。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32。output_final_state=False 时不返回；
            mask=None 时为 None。
    """
    input_dtype = q.dtype

    # 统一转 [B, H, T, *] 并在 float32 下计算。
    q = ops.transpose(q, (0, 2, 1, 3))
    k = ops.transpose(k, (0, 2, 1, 3))
    v = ops.transpose(v, (0, 2, 1, 3))
    beta = ops.transpose(beta, (0, 2, 1))
    tau = ops.transpose(tau, (0, 2, 1))

    q = ops.cast(q, "float32")
    k = ops.cast(k, "float32")
    v = ops.cast(v, "float32")
    beta = ops.cast(beta, "float32")
    tau = ops.cast(tau, "float32")

    q = _l2norm(q, axis=-1)
    k = _l2norm(k, axis=-1)

    batch_size = ops.shape(k)[0]
    num_heads = ops.shape(k)[1]
    seq_len = ops.shape(k)[2]
    k_head_dim = ops.shape(k)[3]
    v_head_dim = ops.shape(v)[3]

    scale = 1.0 / (ops.cast(k_head_dim, "float32") ** 0.5)
    q = q * scale

    # Padding 到 chunk_size 整数倍。
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size
    if pad_size is not None:
        q = ops.pad(q, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        k = ops.pad(k, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        v = ops.pad(v, [[0, 0], [0, 0], [0, pad_size], [0, 0]])
        beta = ops.pad(beta, [[0, 0], [0, 0], [0, pad_size]])
    total_len = seq_len + pad_size

    v_beta = v * ops.expand_dims(beta, -1)
    k_beta = k * ops.expand_dims(beta, -1)

    num_chunks = total_len // chunk_size
    q = ops.reshape(q, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim))
    k = ops.reshape(k, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim))
    v = ops.reshape(v, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim))
    k_beta = ops.reshape(
        k_beta, (batch_size, num_heads, num_chunks, chunk_size, k_head_dim)
    )
    v_beta = ops.reshape(
        v_beta, (batch_size, num_heads, num_chunks, chunk_size, v_head_dim)
    )

    triu_mask = ops.triu(ops.ones((chunk_size, chunk_size)), k=0)
    triu_mask_bool = ops.cast(triu_mask, "bool")

    # 无 decay 门，chunk 内衰减矩阵退化为下三角（含对角）全 1 矩阵。
    tril_incl = ops.transpose(ops.triu(ops.ones((chunk_size, chunk_size)), k=0), (1, 0))

    kbk = ops.einsum("bhcid,bhcjd->bhcij", k_beta, k)
    attn = -(kbk * tril_incl)
    attn = ops.where(triu_mask_bool, ops.zeros_like(attn), attn)

    # Neumann 级数迭代修正，把 attn 变成 (I - lower_triangular(k_beta k^T))^-1。
    for i in range(1, chunk_size):
        row = attn[..., i : i + 1, :i]
        sub = attn[..., :i, :i]
        correction = ops.matmul(row, sub)
        correction = ops.squeeze(correction, axis=-2)
        new_row = attn[..., i, :] + ops.pad(
            correction, [[0, 0], [0, 0], [0, 0], [0, chunk_size - i]]
        )
        before = attn[..., :i, :]
        after = attn[..., i + 1 :, :]
        new_row_exp = ops.expand_dims(new_row, axis=-2)
        attn = ops.concatenate([before, new_row_exp, after], axis=-2)

    attn = attn + ops.eye(chunk_size)

    value = ops.einsum("bhcij,bhcjd->bhcid", attn, v_beta)

    k_cumdecay = ops.einsum("bhcij,bhcjd->bhcid", attn, k_beta)

    if initial_state is None:
        last_state = ops.zeros(
            (batch_size, num_heads, k_head_dim, v_head_dim), dtype="float32"
        )
    else:
        last_state = ops.cast(initial_state, "float32")
        if ops.shape(last_state)[0] == 1:
            last_state = ops.broadcast_to(
                last_state, (batch_size, num_heads, k_head_dim, v_head_dim)
            )

    triu1_mask = ops.triu(ops.ones((chunk_size, chunk_size)), k=1)
    triu1_bool = ops.cast(triu1_mask, "bool")

    use_mask = output_final_state and mask is not None
    if use_mask:
        mask = ops.cast(mask, "float32")

    all_chunks_out = []
    for ci in range(num_chunks):
        q_i = q[:, :, ci]
        k_i = k[:, :, ci]
        v_i = value[:, :, ci]

        qk = ops.einsum("bhid,bhjd->bhij", q_i, k_i)
        intra_attn = qk * tril_incl
        intra_attn = ops.where(triu1_bool, ops.zeros_like(intra_attn), intra_attn)

        k_cd_i = k_cumdecay[:, :, ci]
        v_prime = ops.einsum("bhid,bhdv->bhiv", k_cd_i, last_state)
        v_new = v_i - v_prime

        attn_inter = ops.einsum("bhid,bhdv->bhiv", q_i, last_state)

        chunk_out = attn_inter + ops.einsum("bhij,bhjd->bhid", intra_attn, v_new)
        all_chunks_out.append(ops.expand_dims(chunk_out, axis=2))

        state_update = ops.einsum("bhid,bhiv->bhdv", k_i, v_new)
        last_state = last_state + state_update

        # 在 chunk 边界应用 SANE
        tau_i = tau[:, :, ci]
        mask_i = mask[:, ci] if use_mask else None
        last_state = _apply_sane(last_state, tau_i, mask_i)

    output = ops.concatenate(all_chunks_out, axis=2)
    output = ops.reshape(output, (batch_size, num_heads, total_len, v_head_dim))
    output = output[:, :, :seq_len, :]

    final_state = last_state if output_final_state else None

    output = ops.transpose(output, (0, 2, 1, 3))
    output = ops.cast(output, input_dtype)

    if not output_final_state:
        return output, None

    if mask is None:
        warnings.warn(
            "[delta_net_chunk_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[delta_net_chunk_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return output, None

    return output, final_state

"""Gated DeltaNet chunkwise Keras 参考实现。"""

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


def gated_delta_net_chunk(
    q,
    k,
    v,
    g,
    beta,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """Gated DeltaNet chunkwise 原生实现（数值参考）。

    把输入序列按 chunk_size 分块，通过 decay 矩阵的 WY 表示把串行
    recurrent 计算并行化。chunk 之间按 recurrent 方式传递 state。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间，越负遗忘越快）。
        beta: [B, T, H]，写入强度门控，必须已在外部过 sigmoid 并落在 (0,1)。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。
    """
    input_dtype = q.dtype

    # 统一转 [B, H, T, *] 并在 float32 下计算。
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

    # beta 需已在外部过 sigmoid，落在 (0,1) 内；算子内部不再重复做。

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
        g = ops.pad(g, [[0, 0], [0, 0], [0, pad_size]])
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
    g = ops.reshape(g, (batch_size, num_heads, num_chunks, chunk_size))

    triu_mask = ops.triu(ops.ones((chunk_size, chunk_size)), k=0)
    triu_mask_bool = ops.cast(triu_mask, "bool")

    g = ops.cumsum(g, axis=-1)

    g_row = ops.expand_dims(g, -1)
    g_col = ops.expand_dims(g, -2)
    decay_diff = g_row - g_col
    tril_incl = ops.transpose(ops.triu(ops.ones((chunk_size, chunk_size)), k=0), (1, 0))
    tril_incl_bool = ops.cast(tril_incl, "bool")
    # 先对 decay_diff 做 mask，把上三角置 0 后再 exp，避免上三角大正数 exp 溢出。
    # 若先 exp 再 where，ExpBackward0 仍会在上三角计算 inf*0=NaN。
    decay_diff_masked = ops.where(
        tril_incl_bool, decay_diff, ops.zeros_like(decay_diff)
    )
    decay_mask = ops.exp(decay_diff_masked)

    kbk = ops.einsum("bhcid,bhcjd->bhcij", k_beta, k)
    attn = -(kbk * decay_mask)
    attn = ops.where(triu_mask_bool, ops.zeros_like(attn), attn)

    # Neumann 级数迭代修正，把 attn 变成 (I - lower_triangular(k_beta k^T * decay))^-1。
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

    g_exp = ops.exp(g)
    k_cumdecay = ops.einsum(
        "bhcij,bhcjd->bhcid", attn, k_beta * ops.expand_dims(g_exp, -1)
    )

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

    all_chunks_out = []
    for ci in range(num_chunks):
        q_i = q[:, :, ci]
        k_i = k[:, :, ci]
        v_i = value[:, :, ci]
        dm_i = decay_mask[:, :, ci]
        g_i = g[:, :, ci]

        qk = ops.einsum("bhid,bhjd->bhij", q_i, k_i)
        intra_attn = qk * dm_i
        intra_attn = ops.where(triu1_bool, ops.zeros_like(intra_attn), intra_attn)

        k_cd_i = k_cumdecay[:, :, ci]
        v_prime = ops.einsum("bhid,bhdv->bhiv", k_cd_i, last_state)
        v_new = v_i - v_prime

        attn_inter = ops.einsum(
            "bhid,bhdv->bhiv",
            q_i * ops.expand_dims(ops.exp(g_i), -1),
            last_state,
        )

        chunk_out = attn_inter + ops.einsum("bhij,bhjd->bhid", intra_attn, v_new)
        all_chunks_out.append(ops.expand_dims(chunk_out, axis=2))

        g_last = g_i[..., -1]
        state_decay = ops.exp(ops.expand_dims(ops.expand_dims(g_last, -1), -1))
        g_diff = ops.expand_dims(g_last, -1) - g_i
        k_weighted = k_i * ops.expand_dims(ops.exp(g_diff), -1)
        state_update = ops.einsum("bhid,bhiv->bhdv", k_weighted, v_new)
        last_state = last_state * state_decay + state_update

    output = ops.concatenate(all_chunks_out, axis=2)
    output = ops.reshape(output, (batch_size, num_heads, total_len, v_head_dim))
    output = output[:, :, :seq_len, :]

    final_state = last_state if output_final_state else None

    output = ops.transpose(output, (0, 2, 1, 3))
    output = ops.cast(output, input_dtype)

    return output, final_state

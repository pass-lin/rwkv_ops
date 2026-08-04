"""RWKV-7 广义 delta 规则的纯 Keras 参考实现。"""

import keras
from keras import ops


def transpose_head(x, head_first):
    """在 [B, T, H, K] 与 [B, H, T, K] 两种 layout 间切换。

    内部先把输入 cast 到 float32，保证递推数值稳定性。

    Args:
        x: [B, T, H, K] 或 [B, H, T, K]，任意常见 dtype。
        head_first: bool，为 True 时转置为 [B, H, T, K]，否则原样返回。

    Returns:
        head_first=True 时返回 [B, H, T, K] float32，否则返回原张量 float32。
    """
    x = ops.cast(x, "float32")
    if head_first:
        return ops.transpose(x, (0, 2, 1, 3))
    else:
        return x


def generalized_delta_rule(
    r,
    w,
    k,
    v,
    a,
    b,
    initial_state=None,
    output_final_state: bool = True,
    head_first: bool = False,
    mask=None,
):
    """RWKV-7 广义 delta 规则的逐 token 原生实现（数值 ground truth）。

    递推公式：
        w_t = exp(-exp(w_t))
        sa_t = state_{t-1} @ a_t
        state_t = state_{t-1} * w_t + sa_t \otimes b_t + v_t \otimes k_t
        y_t = state_t @ r_t

    Args:
        r, w, k, v, a, b: [B, T, H, K]，任意常见 dtype。
        initial_state: [B, H, K, K] 或 [1, H, K, K]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入/输出是否采用 [B, H, T, K] layout。
        mask: [B, T] 或 [B, T, 1, 1]，float32，1 表示更新状态、0 表示冻结状态。

    Returns:
        out: [B, T, H, K]，与 r 同 dtype。
        final_state: [B, H, K, K]，float32；仅当 output_final_state=True 时返回。
    """
    DTYPE = r.dtype

    r = transpose_head(r, head_first)

    k = transpose_head(k, head_first)

    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))
    B, T, H, N = ops.shape(r)
    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N))
    state = ops.cast(state, "float32")

    keras_backend = keras.config.backend()

    def step_with_mask(t, inputs):
        """单时间步递推（带 mask）。"""
        state, out = inputs
        old_state = state
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        o = ops.cast((state @ rr), out.dtype)
        if keras_backend == "tensorflow":
            out = out.write(t, ops.reshape(o, (B, H, N)))
        elif keras_backend == "torch":
            out[:, t : t + 1] = ops.reshape(o, (B, 1, H, N))
        else:
            out = ops.slice_update(out, [0, t, 0, 0], ops.reshape(o, (B, 1, H, N)))
        mask_t = ops.reshape(mask[:, t], [-1, 1, 1, 1])
        state = state * mask_t + old_state * (1 - mask_t)
        return [state, out]

    def step_wo_mask(t, inputs):
        """单时间步递推（无 mask）。"""
        state, out = inputs
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        o = ops.cast((state @ rr), out.dtype)
        if keras_backend == "tensorflow":
            out = out.write(t, ops.reshape(o, (B, H, N)))
        elif keras_backend == "torch":
            out[:, t : t + 1] = ops.reshape(o, (B, 1, H, N))
        else:
            out = ops.slice_update(out, [0, t, 0, 0], ops.reshape(o, (B, 1, H, N)))
        return [state, out]

    if mask is not None:
        mask = ops.cast(mask, "float32")
        step = step_with_mask
    else:
        step = step_wo_mask
    if keras_backend == "tensorflow":
        import tensorflow as tf

        out = tf.TensorArray(DTYPE, size=T)
    else:
        out = ops.zeros((B, T, H, N), DTYPE)
    state, out = ops.fori_loop(0, T, step, [state, out])
    if keras_backend == "tensorflow":
        out = ops.transpose(out.stack(), [1, 0, 2, 3])
    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)

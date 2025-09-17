import keras
from keras import ops


def transpose_head(x, head_first):
    """
    对输入张量进行转置操作。

    参数:
    x: 输入张量。
    head_first: 布尔值，决定是否进行转置。

    返回:
    转置后的张量（如果head_first为True），否则返回原张量。
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
):
    """
    实现广义delta规则的函数。

    参数:
    r: 输入张量。
    w: 权重张量。
    k, v, a, b: 其他输入张量。
    initial_state: 初始状态张量。
    output_final_state: 是否输出最终状态。
    head_first: 是否在计算中将head维度放在第一位。

    返回:
    根据output_final_state参数决定是否返回最终状态。
    """
    DTYPE = r.dtype
    B, T, H, N = ops.shape(r)
    r = transpose_head(r, head_first)

    k = transpose_head(k, head_first)

    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)
    w = transpose_head(w, head_first)
    w = ops.exp(-ops.exp(w))

    if initial_state is not None:
        state = initial_state
        if ops.shape(state)[0] == 1:
            state = ops.broadcast_to(state, (B, H, N, N))
    else:
        state = ops.zeros((B, H, N, N), dtype="float32")
    out = ops.zeros((B, T, H, N), dtype=r.dtype)

    def step(t, inputs):
        """
        执行单个时间步的计算。

        参数:
        t: 当前时间步。
        inputs: 包含当前状态和输出的列表。

        返回:
        更新后的状态和输出。
        """
        state, out = inputs
        kk = ops.reshape(k[:, t, :], (B, H, 1, N))
        rr = ops.reshape(r[:, t, :], (B, H, N, 1))
        vv = ops.reshape(v[:, t, :], (B, H, N, 1))
        aa = ops.reshape(a[:, t, :], (B, H, N, 1))
        bb = ops.reshape(b[:, t, :], (B, H, 1, N))
        state = state * w[:, t, :, None, :] + state @ aa @ bb + vv @ kk
        out = ops.slice_update(
            out, [0, t, 0, 0], ops.reshape((state @ rr), (B, 1, H, N))
        )
        return [state, out]

    state, out = ops.fori_loop(0, T, step, [state, out])

    if output_final_state:
        return ops.cast(out, DTYPE), state
    return ops.cast(out, DTYPE)

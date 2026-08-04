"""RWKV-6 原生 Keras-ops 逐 timestep 参考实现。"""

from keras import ops


class RWKVKernelOperator:
    """RWKV-6 核算子（while_loop 逐步 RNN，State 全程 fp32）。

    Args:
        head_size: int。每个 head 的维度。
        max_sequence_length: int。最大序列长度（原生实现不限制，
            仅用于与 CUDA 版本签名对齐）。
    """

    def __init__(self, head_size, max_sequence_length):
        self.head_size = head_size
        self.max_sequence_length = max_sequence_length

    def __call__(
        self, r, k, v, w, u, with_state=False, init_state=None, state_map=None
    ):
        """执行 RWKV-6 前向。

        Args:
            r, k, v, w: [B, T, C]，任意实 dtype。
            u: [H, N] 或 [C]，与输入同 dtype。
            with_state: bool，是否返回最终状态。
            init_state: [B, H, N, N] 或 [H, N, N]，float32，可选。
            state_map: [B] int64，可选。

        Returns:
            y: [B, T, C]，与 r.dtype 相同。
            final_state: [B, H, N, N]，float32（当 with_state=True）。
        """
        B, T, C = ops.shape(r)
        assert C % self.head_size == 0
        H = C // self.head_size
        w = ops.reshape(w, [B, T, H, self.head_size, 1])
        k = ops.reshape(k, [B, T, H, self.head_size, 1])

        v = ops.reshape(v, [B, T, H, 1, self.head_size])
        r = ops.reshape(r, [B, T, H, 1, self.head_size])
        u = ops.reshape(u, [1, H, self.head_size, 1])

        if init_state is not None:
            assert len(init_state.shape) in [
                3,
                4,
            ], "init_state的形状必须为(state_kinds,num_heads,head_size,head_size)"
            if len(init_state.shape) == 3:
                assert init_state.shape == (
                    H,
                    self.head_size,
                    self.head_size,
                ), "state_kinds的形状必须为(BatchSize,num_heads,head_size,head_size)"
                init_state = init_state[None, :]
            else:
                assert init_state.shape[1:] == (
                    H,
                    self.head_size,
                    self.head_size,
                ), "state_kinds的形状必须为(BatchSize,num_heads,head_size,head_size)"
                state_kinds = init_state.shape[0]
            if state_map is None:
                state_kinds = init_state.shape[0]
                if state_kinds == 1:
                    state_map = ops.zeros(shape=(B,), dtype="int64")
                elif state_kinds == B:
                    state_map = ops.convert_to_tensor(
                        [i for i in range(B)], dtype="int64"
                    )
                else:
                    raise ValueError(
                        "无法为您推断state_map的形状，请您手动指定state_map"
                    )

            else:
                if isinstance(state_map, list):
                    state_map = ops.convert_to_tensor(state_map, dtype="int64")
                state_map = ops.cast(state_map, "int64")
                assert (state_map >= 0).all() and (state_map < state_kinds).all(), (
                    f"请确保state_map的值域为[0, {state_kinds})"
                )
            s = ops.take(init_state, state_map, axis=0)
            # 与 CUDA 对齐，State 全程用 fp32 累加。
            s = ops.cast(s, "float32")

        else:
            assert state_map is None
            s = ops.zeros((B, H, self.head_size, self.head_size), dtype="float32")

        u = ops.cast(u, "float32")
        w = ops.exp(-ops.exp(ops.cast(w, "float32")))

        def cond(i, k, v, w, r, s, y):
            return i < T

        def body(i, k, v, w, r, s, y):
            k_t = ops.cast(ops.take(k, i, 1), "float32")
            v_t = ops.cast(ops.take(v, i, 1), "float32")
            kv_t = k_t @ v_t
            w_t = ops.take(w, i, 1)

            r_t = ops.cast(ops.take(r, i, 1), "float32")
            y_t = r_t @ (u * kv_t + s)
            y_t = ops.reshape(ops.cast(y_t, r.dtype), (B, 1, C))
            s = kv_t + w_t * s

            y = ops.slice_update(y, [0, i, 0], y_t)
            return i + 1, k, v, w, r, s, y

        y = ops.zeros([B, T, C], r.dtype)
        i, k, v, w, r, s, y = ops.while_loop(cond, body, (0, k, v, w, r, s, y), T)
        if with_state:
            return y, s
        return y, None

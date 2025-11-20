from keras import ops


class RWKVKernelOperator:
    def __init__(self, head_size, max_sequence_length):
        self.head_size = head_size
        self.max_sequence_length = max_sequence_length

    def __call__(
        self, r, k, v, w, u, with_state=False, init_state=None, state_map=None
    ):
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
                    state_map = ops.zeros(shape=(B,), dtype="int32")
                elif state_kinds == B:
                    state_map = ops.convert_to_tensor(
                        [i for i in range(B)], dtype="int32"
                    )
                else:
                    raise ValueError(
                        "无法为您推断state_map的形状，请您手动指定state_map"
                    )

            else:
                if isinstance(state_map, list):
                    state_map = ops.convert_to_tensor(state_map, dtype="int32")
                state_map = ops.cast(state_map, "int32")
                assert (state_map >= 0).all() and (state_map < state_kinds).all(), (
                    f"请确保state_map的值域为[0, {state_kinds})"
                )
            s = ops.take(init_state, state_map, axis=0)

        else:
            assert state_map is None
            s = ops.zeros((B, H, self.head_size, self.head_size), dtype=u.dtype)

        w = ops.exp(-ops.exp(w))

        def cond(i, k, v, w, r, s, y):
            return i < T

        def body(i, k, v, w, r, s, y):
            k_t = ops.take(k, i, 1)
            v_t = ops.take(v, i, 1)
            kv_t = k_t @ v_t
            w_t = ops.take(w, i, 1)

            r_t = ops.take(r, i, 1)
            y_t = r_t @ (u * kv_t + s)
            y_t = ops.reshape(y_t, (B, 1, C))
            s = kv_t + w_t * s

            y = ops.slice_update(y, [0, i, 0], y_t)
            return i + 1, k, v, w, r, s, y

        y = ops.zeros([B, T, C], r.dtype)
        i, k, v, w, r, s, y = ops.while_loop(cond, body, (0, k, v, w, r, s, y), T)
        if with_state:
            return y, s
        return y, None

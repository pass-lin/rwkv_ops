# copy from https://github.com/ml-explore/mlx-lm/pull/580

import mlx.core as mx


def _make_wkv7_kernel():
    if not mx.metal.is_available():
        return None
    source = """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / H;
        auto h_idx = n % H;
        constexpr int n_per_t = D / 32;
        // [B, T, H, D]
        auto r_ = r + b_idx * T * H * D + h_idx * D;
        auto w_ = w + b_idx * T * H * D + h_idx * D;
        auto k_ = k + b_idx * T * H * D + h_idx * D;
        auto v_ = v + b_idx * T * H * D + h_idx * D;
        auto a_ = a + b_idx * T * H * D + h_idx * D;
        auto b_ = b + b_idx * T * H * D + h_idx * D;
        y += b_idx * T * H * D + h_idx * D;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        // state_in, state_out: [B, H, D, D]
        auto i_state = state_in  + (n * D + dv_idx) * D;
        auto o_state = state_out + (n * D + dv_idx) * D;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        for (int t = 0; t < T; ++t) {
          float sa = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            sa += state[i] * a_[s_idx];
            state[i] = state[i] * w_[s_idx];
          }
          sa = simd_sum(sa);
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * v_[dv_idx] + sa * b_[s_idx];
            out += state[i] * r_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          // Increment data pointers to next time step
          r_ += H * D;
          w_ += H * D;
          k_ += H * D;
          v_ += H * D;
          a_ += H * D;
          b_ += H * D;
          y  += H * D;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """
    inputs = ["r", "w", "k", "v", "a", "b", "state_in", "T"]
    return mx.fast.metal_kernel(
        name="wkv7_kernel",
        input_names=inputs,
        output_names=["y", "state_out"],
        source=source,
    )


_wkv7_kernel = _make_wkv7_kernel()


def transpose_head(x, head_first: bool = True):
    if head_first:
        return mx.transpose(x, (0, 2, 1, 3))
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
    state = initial_state

    r = transpose_head(r, head_first)
    k = transpose_head(k, head_first)
    v = transpose_head(v, head_first)
    a = transpose_head(a, head_first)
    b = transpose_head(b, head_first)

    B, T, H, D = r.shape
    input_dtype = r.dtype

    y, out_state = _wkv7_kernel(
        inputs=[r, w, k, v, a, b, state, T],
        template=[
            ("InT", input_dtype),
            ("H", H),
            ("D", D),
        ],
        grid=(32, D, B * H),
        threadgroup=(32, 4, 1),
        output_shapes=[(B, T, H, D), state.shape],
        output_dtypes=[input_dtype, input_dtype],
    )
    if output_final_state:
        return y, out_state
    return y

"""
RWKV-7 JAX CUDA 单步 RNN 接口数值测试。
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


@pytest.fixture
def single_step_inputs(rwkv7_inputs, rng):
    B, _, H, K = rwkv7_inputs["r"].shape
    return {
        name: rng.standard_normal((B, 1, H, K), dtype=np.float32)
        for name in ["r", "k", "v", "a", "b", "w"]
    } | {"h0": rng.standard_normal((B, H, K, K), dtype=np.float32)}


@pytest.mark.jax
def test_rwkv7_single_step_forward_state(
    rwkv7_rnn_op, rwkv7_native_op, single_step_inputs
):
    def make(tensors, dtype):
        return (
            _to_jax(tensors["r"], dtype),
            _to_jax(tensors["k"], dtype),
            _to_jax(tensors["v"], dtype),
            _to_jax(tensors["a"], dtype),
            _to_jax(tensors["b"], dtype),
            _to_jax(tensors["w"], dtype),
            _to_jax(tensors["h0"], "float32"),
        )

    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, h0_ref = make(
        single_step_inputs, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, h0_c = make(single_step_inputs, "bfloat16")

    def call(op, r, k, v, a, b, w, h0):
        return op(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=h0,
            output_final_state=True,
        )

    y_ref, s_ref = call(
        rwkv7_native_op, r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, h0_ref
    )
    y_c, s_c = call(rwkv7_rnn_op, r_c, k_c, v_c, a_c, b_c, w_c, h0_c)

    assert_allclose_with_stats(y_ref, y_c, "y", atol=1.0, rtol=1e-1)
    assert_allclose_with_stats(s_ref, s_c, "final_state", atol=1.0, rtol=1e-1)

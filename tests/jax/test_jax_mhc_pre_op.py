"""
mHC Pre-Op JAX-Triton 正确性测试。
"""

import jax
import jax.numpy as jnp
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


@pytest.fixture(scope="module")
def ops():
    from rwkv_ops.mhc_kernel.jax_triton_op.mhc_pre_op import (
        mhc_pre_op_fused as triton_mhc_pre_op,
    )
    from rwkv_ops.mhc_kernel.native_op import mhc_pre_op_fused as native_mhc_pre_op

    return native_mhc_pre_op, triton_mhc_pre_op


@pytest.mark.jax
@pytest.mark.slow
def test_mhc_pre_op_forward(ops, mhc_pre_inputs):
    native_op, triton_op = ops
    x = _to_jax(mhc_pre_inputs["x"], "bfloat16")
    h_res = _to_jax(mhc_pre_inputs["h_res"], "float32")
    h_pre = _to_jax(mhc_pre_inputs["h_pre"], "float32")

    x_in_n, H_res_n = native_op(x, h_res, h_pre, num_iters=20)
    x_in_t, H_res_t = triton_op(x, h_res, h_pre, num_iters=20)

    assert_allclose_with_stats(x_in_n, x_in_t, "x_layer_in", atol=1e-2, rtol=1e-2)
    assert_allclose_with_stats(H_res_n, H_res_t, "H_res", atol=1e-2, rtol=1e-2)


@pytest.mark.jax
@pytest.mark.slow
def test_mhc_pre_op_backward(ops, mhc_pre_inputs):
    native_op, triton_op = ops
    x = _to_jax(mhc_pre_inputs["x"], "bfloat16")
    h_res = _to_jax(mhc_pre_inputs["h_res"], "float32")
    h_pre = _to_jax(mhc_pre_inputs["h_pre"], "float32")

    def loss_fn(op_func, x, hr, hp):
        out_x, out_h = op_func(x, hr, hp, num_iters=20)
        return jnp.mean(out_x.astype(jnp.float32) ** 2) + jnp.mean(out_h**2)

    grad_native = jax.grad(
        lambda x, hr, hp: loss_fn(native_op, x, hr, hp), argnums=(0, 1, 2)
    )(x, h_res, h_pre)
    grad_triton = jax.grad(
        lambda x, hr, hp: loss_fn(triton_op, x, hr, hp), argnums=(0, 1, 2)
    )(x, h_res, h_pre)

    names = ["x", "h_res", "h_pre"]
    for ref, tgt, name in zip(grad_native, grad_triton, names):
        assert_allclose_with_stats(ref, tgt, f"grad_{name}", atol=1e-2, rtol=1e-2)

"""mHC Post-Op JAX-Triton 正确性测试。"""

import jax
import jax.numpy as jnp
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_jax(arr, dtype):
    """把 numpy 数组转成指定 dtype 的 JAX 数组。

    Args:
        arr: np.ndarray，输入数组。
        dtype: str, jnp dtype 名称。

    Returns:
        jax.Array: 指定 dtype 的 JAX 数组。
    """
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


@pytest.fixture(scope="module")
def ops():
    """返回 native 与 Triton mHC post-op 算子对。

    Returns:
        tuple: (native_mhc_post_op, triton_mhc_post_op)。
    """
    from rwkv_ops.mhc_kernel.jax_triton_op.mhc_post_op import (
        mhc_post_op as triton_mhc_op,
    )
    from rwkv_ops.mhc_kernel.native_op import mhc_post_op as native_mhc_op

    return native_mhc_op, triton_mhc_op


@pytest.mark.jax
@pytest.mark.slow
def test_mhc_post_op_forward(ops, mhc_post_inputs):
    """对比 Triton 与 native mHC post-op 前向输出。"""
    native_mhc_op, triton_mhc_op = ops
    layer_out = _to_jax(mhc_post_inputs["layer_out"], "bfloat16")
    x_expanded = _to_jax(mhc_post_inputs["x_expanded"], "bfloat16")
    h_post = _to_jax(mhc_post_inputs["h_post"], "float32")
    H_res = _to_jax(mhc_post_inputs["H_res"], "float32")

    out_n = native_mhc_op(layer_out, x_expanded, h_post, H_res)
    out_t = triton_mhc_op(layer_out, x_expanded, h_post, H_res)

    assert_allclose_with_stats(out_n, out_t, "output", atol=1e-2, rtol=1e-2)


@pytest.mark.jax
@pytest.mark.slow
def test_mhc_post_op_backward(ops, mhc_post_inputs):
    """对比 Triton 与 native mHC post-op 反向梯度。"""
    native_mhc_op, triton_mhc_op = ops
    layer_out = _to_jax(mhc_post_inputs["layer_out"], "bfloat16")
    x_expanded = _to_jax(mhc_post_inputs["x_expanded"], "bfloat16")
    h_post = _to_jax(mhc_post_inputs["h_post"], "float32")
    H_res = _to_jax(mhc_post_inputs["H_res"], "float32")

    def loss_fn(op_func, lo, x, h, H):
        out = op_func(lo, x, h, H)
        return jnp.mean(out.astype(jnp.float32) ** 2)

    grad_native = jax.grad(
        lambda lo, x, h, H: loss_fn(native_mhc_op, lo, x, h, H), argnums=(0, 1, 2, 3)
    )(layer_out, x_expanded, h_post, H_res)
    grad_triton = jax.grad(
        lambda lo, x, h, H: loss_fn(triton_mhc_op, lo, x, h, H), argnums=(0, 1, 2, 3)
    )(layer_out, x_expanded, h_post, H_res)

    names = ["layer_out", "x_expanded", "h_post", "H_res"]
    for ref, tgt, name in zip(grad_native, grad_triton, names):
        assert_allclose_with_stats(ref, tgt, f"grad_{name}", atol=1e-2, rtol=1e-2)

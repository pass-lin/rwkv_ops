"""
RWKV-7 JAX Pallas kernel 数值测试。

运行方式：
    KERAS_BACKEND=jax pytest tests/jax/test_jax_rwkv7_pallas.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax.experimental.pallas")


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _prepare_inputs(rwkv7_inputs, head_first, dtype="bfloat16"):
    if head_first:
        r = _to_jax(np.transpose(rwkv7_inputs["r"], (0, 2, 1, 3)), dtype)
        k = _to_jax(np.transpose(rwkv7_inputs["k"], (0, 2, 1, 3)), dtype)
        v = _to_jax(np.transpose(rwkv7_inputs["v"], (0, 2, 1, 3)), dtype)
        a = _to_jax(np.transpose(rwkv7_inputs["a"], (0, 2, 1, 3)), dtype)
        b = _to_jax(np.transpose(rwkv7_inputs["b"], (0, 2, 1, 3)), dtype)
        w = _to_jax(np.transpose(rwkv7_inputs["w"], (0, 2, 1, 3)), dtype)
    else:
        r = _to_jax(rwkv7_inputs["r"], dtype)
        k = _to_jax(rwkv7_inputs["k"], dtype)
        v = _to_jax(rwkv7_inputs["v"], dtype)
        a = _to_jax(rwkv7_inputs["a"], dtype)
        b = _to_jax(rwkv7_inputs["b"], dtype)
        w = _to_jax(rwkv7_inputs["w"], dtype)
    h0 = _to_jax(rwkv7_inputs["h0"], "float32")
    return r, k, v, a, b, w, h0


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_forward_state(
    rwkv7_jax_pallas_op, rwkv7_native_op, rwkv7_inputs, head_first
):
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, head_first, "bfloat16")

    y_ref, s_ref = rwkv7_native_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=head_first,
    )
    y_c, s_c = rwkv7_jax_pallas_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=head_first,
    )

    assert_allclose_with_stats(
        y_ref, y_c, f"y_head_first={head_first}", atol=1e-5, rtol=1e-2
    )
    assert_allclose_with_stats(
        s_ref, s_c, f"final_state_head_first={head_first}", atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_backward(
    rwkv7_jax_pallas_op, rwkv7_native_op, rwkv7_inputs, head_first
):
    """Pallas custom_vjp 反向梯度与 native Keras 实现逐元素对比。"""
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, head_first, "bfloat16")

    def loss(op, params):
        w, r, k, v, a, b, h0 = params
        y, state = op(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=h0,
            output_final_state=True,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_native_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )
    pallas_grads = jax.grad(lambda *p: loss(rwkv7_jax_pallas_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_mask_forward_backward(
    rwkv7_jax_pallas_op, rwkv7_native_op, rwkv7_inputs, head_first
):
    """带 mask 路径的前向与反向对比（后半序列 padding）。"""
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, head_first, "bfloat16")
    B, T = rwkv7_inputs["r"].shape[0], rwkv7_inputs["r"].shape[1]
    mask = np.ones((B, T), dtype=np.float32)
    mask[:, T // 2 :] = 0.0
    mask = _to_jax(mask, "float32")

    def loss(op, params):
        w, r, k, v, a, b, h0 = params
        y, state = op(
            r=r,
            k=k,
            v=v,
            a=a,
            b=b,
            w=w,
            initial_state=h0,
            output_final_state=True,
            head_first=head_first,
            mask=mask,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    params = (w, r, k, v, a, b, h0)
    y_ref, s_ref = rwkv7_native_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=head_first,
        mask=mask,
    )
    y_c, s_c = rwkv7_jax_pallas_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=head_first,
        mask=mask,
    )
    assert_allclose_with_stats(
        y_ref, y_c, f"y_mask_head_first={head_first}", atol=1e-5, rtol=1e-2
    )
    assert_allclose_with_stats(
        s_ref, s_c, f"final_state_mask_head_first={head_first}", atol=1e-5, rtol=1e-3
    )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_native_op, p), argnums=range(7))(*params)
    pallas_grads = jax.grad(lambda *p: loss(rwkv7_jax_pallas_op, p), argnums=range(7))(
        *params
    )

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_mask_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )

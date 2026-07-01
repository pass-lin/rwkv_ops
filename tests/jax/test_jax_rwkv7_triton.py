"""
RWKV-7 JAX Triton kernel 数值测试。
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


@pytest.fixture(scope="module")
def triton_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op


@pytest.fixture(scope="module")
def native_op():
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


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
def test_rwkv7_triton_forward_state(triton_op, native_op, rwkv7_inputs, head_first):
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, head_first, "bfloat16")

    y_ref, s_ref = native_op(
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
    y_c, s_c = triton_op(
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
def test_rwkv7_triton_backward(triton_op, native_op, rwkv7_inputs, head_first):
    """Triton custom_vjp 反向梯度与 native Keras 实现逐元素对比。"""
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

    ref_grads = jax.grad(lambda *p: loss(native_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )
    triton_grads = jax.grad(lambda *p: loss(triton_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, triton_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )

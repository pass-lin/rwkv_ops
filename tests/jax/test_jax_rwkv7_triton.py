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
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, h0_ref = _prepare_inputs(
        rwkv7_inputs, head_first, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, h0_c = _prepare_inputs(
        rwkv7_inputs, head_first, "bfloat16"
    )

    y_ref, s_ref = native_op(
        r=r_ref,
        k=k_ref,
        v=v_ref,
        a=a_ref,
        b=b_ref,
        w=w_ref,
        initial_state=h0_ref,
        output_final_state=True,
        head_first=head_first,
    )
    y_c, s_c = triton_op(
        r=r_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        w=w_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=head_first,
    )

    assert_allclose_with_stats(
        y_ref, y_c, f"y_head_first={head_first}", atol=1.0, rtol=1e-1
    )
    assert_allclose_with_stats(
        s_ref, s_c, f"final_state_head_first={head_first}", atol=1.0, rtol=1e-1
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_triton_backward_directional(
    triton_op, native_op, rwkv7_inputs, head_first, rng
):
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, h0_ref = _prepare_inputs(
        rwkv7_inputs, head_first, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, h0_c = _prepare_inputs(
        rwkv7_inputs, head_first, "bfloat16"
    )

    key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
    keys = jax.random.split(key, 7)
    dirs = [
        jax.random.normal(k, p.shape, dtype=jnp.float32)
        for k, p in zip(keys, [w_ref, r_ref, k_ref, v_ref, a_ref, b_ref, h0_ref])
    ]

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

    def directional_fd(op, params, dirs, eps=1e-3):
        plus = [p + eps * d for p, d in zip(params, dirs)]
        minus = [p - eps * d for p, d in zip(params, dirs)]
        return (loss(op, plus) - loss(op, minus)) / (2 * eps)

    ref_val = directional_fd(
        native_op, [w_ref, r_ref, k_ref, v_ref, a_ref, b_ref, h0_ref], dirs
    )

    grad_c = jax.grad(
        lambda w, r, k, v, a, b, h0: loss(triton_op, [w, r, k, v, a, b, h0]),
        argnums=range(7),
    )(w_c, r_c, k_c, v_c, a_c, b_c, h0_c)
    cuda_val = sum(
        jnp.sum(jnp.asarray(g, jnp.float32) * d) for g, d in zip(grad_c, dirs)
    )

    rel = float(jnp.abs(ref_val - cuda_val) / (jnp.abs(ref_val) + 1e-8))
    print(
        f"[directional_derivative head_first={head_first}] ref={float(ref_val):.6e}, "
        f"cuda={float(cuda_val):.6e}, rel_diff={rel:.3e}"
    )
    assert rel < 1e-1, f"方向导数相对差异过大: {rel}"

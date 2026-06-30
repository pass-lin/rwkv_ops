"""
RWKV-7 JAX CUDA kernel 数值测试。

运行方式：
    KERAS_BACKEND=jax pytest tests/jax/test_rwkv7.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _normalize(z, axis=-1, eps=1e-12):
    denom = jnp.linalg.norm(z, axis=axis, keepdims=True)
    denom = jnp.maximum(denom, eps)
    return z / denom


def _prepare_inputs(rwkv7_inputs, head_first, dtype="bfloat16"):
    if head_first:
        # native/CUDA 内部会再次转置，测试直接传入 B,H,T,K
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
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_forward_state(rwkv7_jax_op, rwkv7_native_op, rwkv7_inputs, head_first):
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, h0_ref = _prepare_inputs(
        rwkv7_inputs, head_first, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, h0_c = _prepare_inputs(
        rwkv7_inputs, head_first, "bfloat16"
    )

    y_ref, s_ref = rwkv7_native_op(
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
    y_c, s_c = rwkv7_jax_op(
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
def test_rwkv7_backward_directional(
    rwkv7_jax_op, rwkv7_native_op, rwkv7_inputs, head_first, rng
):
    """
    JAX native 的 fori_loop/while_loop 不支持 reverse-mode 自动求导，
    因此用随机方向的有限差分验证 CUDA custom_vjp 的反向梯度。
    """
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
        rwkv7_native_op,
        [w_ref, r_ref, k_ref, v_ref, a_ref, b_ref, h0_ref],
        dirs,
    )

    def cuda_loss_fn(w, r, k, v, a, b, h0):
        return loss(
            rwkv7_jax_op,
            [w, r, k, v, a, b, h0],
        )

    grad_c = jax.grad(cuda_loss_fn, argnums=range(7))(
        w_c, r_c, k_c, v_c, a_c, b_c, h0_c
    )
    cuda_val = sum(
        jnp.sum(jnp.asarray(g, jnp.float32) * d) for g, d in zip(grad_c, dirs)
    )

    rel = float(jnp.abs(ref_val - cuda_val) / (jnp.abs(ref_val) + 1e-8))
    print(
        f"[directional_derivative head_first={head_first}] ref={float(ref_val):.6e}, "
        f"cuda={float(cuda_val):.6e}, rel_diff={rel:.3e}"
    )
    assert rel < 1e-1, f"方向导数相对差异过大: {rel}"

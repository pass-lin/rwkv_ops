"""RWKV-7 JAX Triton kernel 数值测试。"""

import jax
import jax.numpy as jnp
import numpy as np
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
def triton_op(rwkv7_shape):
    """RWKV-7 JAX Triton 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 Triton 训练 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op


@pytest.fixture(scope="module")
def native_op():
    """RWKV-7 native Keras 参考算子。

    Returns:
        Callable: RWKV-7 native_keras_op.generalized_delta_rule。
    """
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


def _prepare_inputs(rwkv7_inputs, head_first, dtype="bfloat16"):
    """把 rwkv7_inputs 转成 JAX 测试张量。

    Args:
        rwkv7_inputs: dict, 来自 fixture 的 numpy 输入。
        dtype: str, r/k/v/a/b/w 的目标 dtype。
        head_first: bool, 是否将 layout 转置为 [B, H, T, K]。

    Returns:
        tuple: (r, k, v, a, b, w, h0)，前六个为 dtype 的 JAX 数组，h0 为 float32。
    """
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
    """对比 Triton 与 native 前向输出和最终 state。"""
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

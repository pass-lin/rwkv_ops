"""RWKV-7 JAX CUDA kernel 数值测试。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    """把 numpy 数组转成指定 dtype 的 JAX 数组。

    Args:
        arr: np.ndarray，输入数组。
        dtype: str, jnp dtype 名称。

    Returns:
        jax.Array: 指定 dtype 的 JAX 数组。
    """
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _normalize(z, axis=-1, eps=1e-12):
    """沿指定轴做 L2 归一化。

    Args:
        z: jax.Array，输入张量。
        axis: int, 归一化轴。
        eps: float, 防止除零的小常数。

    Returns:
        jax.Array: 归一化后的张量。
    """
    denom = jnp.linalg.norm(z, axis=axis, keepdims=True)
    denom = jnp.maximum(denom, eps)
    return z / denom


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
        # native/CUDA 内部也会再次转置，测试直接传入 B,H,T,K。
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
    """对比 CUDA 与 native 前向输出和最终 state。"""
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
    y_c, s_c = rwkv7_jax_op(
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
def test_rwkv7_backward(rwkv7_jax_op, rwkv7_native_op, rwkv7_inputs, head_first):
    """CUDA custom_vjp 反向梯度与 native Keras 实现逐元素对比。"""
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
    cuda_grads = jax.grad(lambda *p: loss(rwkv7_jax_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, cuda_grads):
        # grad_b 的数值敏感性略高，参考 torch 测试使用 1e-2
        atol = 1e-2 if name == "b" else 7e-3
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_{name}_head_first={head_first}",
            atol=atol,
            rtol=1e-3,
        )

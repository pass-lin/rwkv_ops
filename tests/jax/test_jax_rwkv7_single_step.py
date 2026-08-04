"""RWKV-7 JAX CUDA 单步 RNN 接口数值测试。"""

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


@pytest.fixture
def single_step_inputs(rwkv7_inputs, rng):
    """生成 T=1 的单步 RNN 测试输入。

    Args:
        rwkv7_inputs: dict, 来自 fixture 的 numpy 输入（用于取 B/H/K）。
        rng: np.random.Generator，随机数生成器。

    Returns:
        dict: 包含 r/k/v/a/b/w([B, 1, H, K]) 与 h0([B, H, K, K]) 的 float32 数组。
    """
    B, _, H, K = rwkv7_inputs["r"].shape
    return {
        name: rng.standard_normal((B, 1, H, K), dtype=np.float32)
        for name in ["r", "k", "v", "a", "b", "w"]
    } | {"h0": rng.standard_normal((B, H, K, K), dtype=np.float32)}


@pytest.mark.jax
def test_rwkv7_single_step_forward_state(
    rwkv7_rnn_op, rwkv7_native_op, single_step_inputs
):
    """对比 CUDA 单步 RNN 与 native 单步前向输出和最终 state。"""

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

    r, k, v, a, b, w, h0 = make(single_step_inputs, "bfloat16")

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

    y_ref, s_ref = call(rwkv7_native_op, r, k, v, a, b, w, h0)
    y_c, s_c = call(rwkv7_rnn_op, r, k, v, a, b, w, h0)

    assert_allclose_with_stats(y_ref, y_c, "y", atol=1e-5, rtol=1e-2)
    assert_allclose_with_stats(s_ref, s_c, "final_state", atol=1e-5, rtol=1e-3)

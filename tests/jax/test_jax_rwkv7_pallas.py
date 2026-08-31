"""RWKV-7 JAX Pallas kernel 数值测试。"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax.experimental.pallas")


def _to_jax(arr, dtype):
    """把 numpy 数组转成指定 dtype 的 JAX 数组。

    Args:
        arr: np.ndarray，输入数组。
        dtype: str, jnp dtype 名称。

    Returns:
        jax.Array: 指定 dtype 的 JAX 数组。
    """
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


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


def _get_rwkv7_pallas_op_chunk_size_8(rwkv7_shape):
    """构造 chunk_size=8 的 RWKV-7 JAX Pallas 训练算子。

    非 GPU/TPU 环境下会 pytest.skip。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: chunk_size=8 的 RWKV-7 Pallas 训练 kernel。
    """
    pytest.importorskip("jax.experimental.pallas")
    import jax

    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("pallas 后端仅用于 GPU/TPU")
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="pallas", chunk_size=8)
    return op


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_forward_state(
    rwkv7_jax_pallas_op, rwkv7_native_op, rwkv7_inputs, head_first
):
    """对比 Pallas 与 native 前向输出和最终 state。"""
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


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_forward_state_chunk_size_8(
    rwkv7_native_op, rwkv7_inputs, rwkv7_shape, head_first
):
    """对比 chunk_size=8 时 Pallas 与 native 前向输出和最终 state。"""
    pallas_op = _get_rwkv7_pallas_op_chunk_size_8(rwkv7_shape)
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
        chunk_size=8,
    )
    y_c, s_c = pallas_op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=True,
        head_first=head_first,
        chunk_size=8,
    )

    assert_allclose_with_stats(
        y_ref, y_c, f"y_chunk_size=8_head_first={head_first}", atol=1e-5, rtol=1e-2
    )
    assert_allclose_with_stats(
        s_ref,
        s_c,
        f"final_state_chunk_size=8_head_first={head_first}",
        atol=1e-5,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_backward_chunk_size_8(
    rwkv7_native_op, rwkv7_inputs, rwkv7_shape, head_first
):
    """chunk_size=8 时 Pallas custom_vjp 反向梯度与 native Keras 实现对比。"""
    pallas_op = _get_rwkv7_pallas_op_chunk_size_8(rwkv7_shape)
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
            chunk_size=8,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_native_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )
    pallas_grads = jax.grad(lambda *p: loss(pallas_op, p), argnums=range(7))(
        w, r, k, v, a, b, h0
    )

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_chunk_size=8_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_pallas_mask_forward_backward_chunk_size_8(
    rwkv7_native_op, rwkv7_inputs, rwkv7_shape, head_first
):
    """chunk_size=8 时带 mask 路径的前向与反向对比（后半序列 padding）。"""
    pallas_op = _get_rwkv7_pallas_op_chunk_size_8(rwkv7_shape)
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
            chunk_size=8,
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
        chunk_size=8,
    )
    y_c, s_c = pallas_op(
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
        chunk_size=8,
    )
    assert_allclose_with_stats(
        y_ref,
        y_c,
        f"y_mask_chunk_size=8_head_first={head_first}",
        atol=1e-5,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        s_ref,
        s_c,
        f"final_state_mask_chunk_size=8_head_first={head_first}",
        atol=1e-5,
        rtol=1e-3,
    )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_native_op, p), argnums=range(7))(*params)
    pallas_grads = jax.grad(lambda *p: loss(pallas_op, p), argnums=range(7))(*params)

    names = ["w", "r", "k", "v", "a", "b", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_mask_chunk_size=8_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )

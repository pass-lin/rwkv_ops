"""
RWKV-7 JAX CUDA 推理专用接口数值测试。
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _prepare_inputs(rwkv7_inputs, dtype="bfloat16"):
    return (
        _to_jax(rwkv7_inputs["r"], dtype),
        _to_jax(rwkv7_inputs["k"], dtype),
        _to_jax(rwkv7_inputs["v"], dtype),
        _to_jax(rwkv7_inputs["a"], dtype),
        _to_jax(rwkv7_inputs["b"], dtype),
        _to_jax(rwkv7_inputs["w"], dtype),
        _to_jax(rwkv7_inputs["h0"], "float32"),
    )


def _call_op(op, r, k, v, a, b, w, h0, output_final_state=True, mask=None):
    return op(
        r=r,
        k=k,
        v=v,
        a=a,
        b=b,
        w=w,
        initial_state=h0,
        output_final_state=output_final_state,
        mask=mask,
    )


@pytest.mark.jax
def test_rwkv7_inference_forward_state(
    rwkv7_inference_op, rwkv7_native_op, rwkv7_inputs
):
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_native_op, r, k, v, a, b, w, h0)
    y_c, s_c = _call_op(rwkv7_inference_op, r, k, v, a, b, w, h0)

    assert_allclose_with_stats(y_ref, y_c, "y", atol=1e-5, rtol=1e-2)
    assert_allclose_with_stats(s_ref, s_c, "final_state", atol=1e-5, rtol=1e-3)


@pytest.mark.jax
def test_rwkv7_inference_masked(rwkv7_inference_op, rwkv7_native_op, rwkv7_inputs, rng):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.ones((B, T), dtype=np.float32)
    freeze = rng.random((B, T)) < 0.3
    mask_np[freeze] = 0.0
    mask = jnp.asarray(mask_np)

    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_native_op, r, k, v, a, b, w, h0, mask=mask)
    y_c, s_c = _call_op(rwkv7_inference_op, r, k, v, a, b, w, h0, mask=mask)

    assert_allclose_with_stats(y_ref, y_c, "y_mask", atol=1e-5, rtol=1e-2)
    assert_allclose_with_stats(s_ref, s_c, "final_state_mask", atol=1e-5, rtol=1e-3)


@pytest.mark.jax
def test_rwkv7_inference_mask_all_zero_frozen(rwkv7_inference_op, rwkv7_inputs):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = jnp.zeros((B, T), dtype=jnp.float32)
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, "bfloat16")
    h0_frozen = h0.copy()

    _, state_frozen = _call_op(
        rwkv7_inference_op, r, k, v, a, b, w, h0_frozen, mask=mask
    )

    diff = float(jnp.abs(state_frozen - h0_frozen).max())
    assert diff < 1e-5, f"全 0 Mask 状态被改变 (max_diff={diff:.3e})"


@pytest.mark.jax
def test_rwkv7_inference_mask_all_one_equivalent(rwkv7_inference_op, rwkv7_inputs):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = jnp.ones((B, T), dtype=jnp.float32)
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, "bfloat16")

    y_no_mask, s_no_mask = _call_op(rwkv7_inference_op, r, k, v, a, b, w, h0)
    y_all_one, s_all_one = _call_op(rwkv7_inference_op, r, k, v, a, b, w, h0, mask=mask)

    pred_diff = float(jnp.abs(y_all_one - y_no_mask).max())
    state_diff = float(jnp.abs(s_all_one - s_no_mask).max())
    assert pred_diff < 1e-5, f"全 1 Mask 输出不一致 (max_diff={pred_diff:.3e})"
    assert state_diff < 1e-5, f"全 1 Mask 状态不一致 (max_diff={state_diff:.3e})"


@pytest.mark.jax
def test_rwkv7_inference_last_frame_only(rwkv7_inference_op, rwkv7_inputs):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.zeros((B, T), dtype=np.float32)
    mask_np[:, -1] = 1.0
    mask = jnp.asarray(mask_np)
    r, k, v, a, b, w, h0 = _prepare_inputs(rwkv7_inputs, "bfloat16")

    _, state_last = _call_op(rwkv7_inference_op, r, k, v, a, b, w, h0, mask=mask)

    state_change = float(jnp.abs(state_last - h0).mean())
    assert state_change > 0, "最后一帧 Mask 状态未变化"

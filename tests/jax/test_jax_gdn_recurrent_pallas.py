"""Gated DeltaNet recurrent Pallas kernel 的 JAX 后端测试。"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("jax.experimental.pallas")

import jax
import jax.numpy as jnp

from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent as gdn_native_recurrent,
    gated_delta_net_recurrent_inference as gdn_native_inference,
    gated_delta_net_recurrent_single_step as gdn_native_single_step,
)
from rwkv_ops.gdn_recurrent.jax_pallas_kernel import (
    gated_delta_net_recurrent as gdn_pallas_recurrent,
    gated_delta_net_recurrent_inference as gdn_pallas_inference,
    gated_delta_net_recurrent_single_step as gdn_pallas_single_step,
)
from tests.conftest import assert_allclose_with_stats


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def gdn_jax_device():
    """JAX Pallas recurrent kernel 需要 GPU/TPU，否则跳过整个文件。"""
    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("Gated DeltaNet recurrent Pallas kernel requires JAX GPU/TPU.")
    return jax.devices()[0]


def _gdn_value_and_grad(fn, q, k, v, g, beta, h0):
    """计算 GDN recurrent 算子对 q/k/v/g/beta/h0 的数值与梯度。"""

    def loss_fn(q, k, v, g, beta, h0):
        out, state = fn(q, k, v, g, beta, initial_state=h0, output_final_state=True)
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, beta, h0)


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_pallas_recurrent_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Pallas recurrent 训练算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_pallas, state_pallas = gdn_pallas_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_pallas, "pallas recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas recurrent vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_gdn_pallas_inference_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Pallas recurrent 推理算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_pallas, state_pallas = gdn_pallas_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_pallas, "pallas inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_gdn_pallas_single_step_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Pallas recurrent 单步 RNN 算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"][:, 0], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"][:, 0], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"][:, 0], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"][:, 0], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"][:, 0], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_pallas, state_pallas = gdn_pallas_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_pallas, "pallas single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas single step vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_pallas_recurrent_backward(gdn_inputs, gdn_jax_device):
    """JAX Pallas recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    _, grads_ref = _gdn_value_and_grad(gdn_native_recurrent, q, k, v, g, beta, h0)
    _, grads_pallas = _gdn_value_and_grad(gdn_pallas_recurrent, q, k, v, g, beta, h0)

    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, gr, gp in zip(names, grads_ref, grads_pallas):
        assert_allclose_with_stats(
            gr,
            gp,
            f"grad_{name} pallas vs native",
            atol=7e-3,
            rtol=1e-2,
        )

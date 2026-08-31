"""Gated DeltaNet recurrent Triton kernel 的 JAX 后端测试。"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_triton")

import jax
import jax.numpy as jnp

from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent as gdn_native_recurrent,
    gated_delta_net_recurrent_inference as gdn_native_inference,
    gated_delta_net_recurrent_single_step as gdn_native_single_step,
)
from rwkv_ops.gdn_recurrent.jax_triton_kernel import (
    gated_delta_net_recurrent as gdn_triton_recurrent,
    gated_delta_net_recurrent_inference as gdn_triton_inference,
    gated_delta_net_recurrent_single_step as gdn_triton_single_step,
)
from tests.conftest import assert_allclose_with_stats


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def gdn_jax_device():
    """JAX Triton recurrent kernel 需要 GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("Gated DeltaNet recurrent Triton kernel requires JAX GPU.")
    return jax.devices()[0]


def _gdn_value_and_grad(fn, q, k, v, g, beta, h0):
    """计算 GDN recurrent 算子对 q/k/v/g/beta/h0 的数值与梯度。"""

    def loss_fn(q, k, v, g, beta, h0):
        out, state = fn(q, k, v, g, beta, initial_state=h0, output_final_state=True)
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4, 5))(q, k, v, g, beta, h0)


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_triton_recurrent_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 训练算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_gdn_triton_inference_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 推理算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_tri, state_tri = gdn_triton_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_gdn_triton_single_step_matches_native(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 单步 RNN 算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"][:, 0], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"][:, 0], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"][:, 0], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"][:, 0], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"][:, 0], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    out_tri, state_tri = gdn_triton_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_triton_recurrent_backward(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device)

    _, grads_ref = _gdn_value_and_grad(gdn_native_recurrent, q, k, v, g, beta, h0)
    _, grads_tri = _gdn_value_and_grad(gdn_triton_recurrent, q, k, v, g, beta, h0)

    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, gr, gt in zip(names, grads_ref, grads_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"grad_{name} triton vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_triton_recurrent_chunk_size(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 训练算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device, jnp.float32)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device, jnp.float32)

    out_tri, state_tri = gdn_triton_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = gdn_native_recurrent(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton recurrent chunk_size=8 vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton recurrent chunk_size=8 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_triton_recurrent_backward_chunk_size(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 训练算子非默认 chunk_size 反向梯度与 native 对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device, jnp.float32)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device, jnp.float32)

    def loss_fn(fn, q, k, v, g, beta, h0):
        out, state = fn(
            q,
            k,
            v,
            g,
            beta,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    _, grads_ref = jax.value_and_grad(
        lambda q, k, v, g, beta, h0: loss_fn(
            gdn_native_recurrent, q, k, v, g, beta, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, g, beta, h0)
    _, grads_tri = jax.value_and_grad(
        lambda q, k, v, g, beta, h0: loss_fn(
            gdn_triton_recurrent, q, k, v, g, beta, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, g, beta, h0)

    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, gr, gt in zip(names, grads_ref, grads_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"grad_{name} triton chunk_size=8 vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_gdn_triton_inference_chunk_size(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 推理算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(gdn_inputs["q"], gdn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_inputs["k"], gdn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_inputs["v"], gdn_jax_device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_inputs["g"], gdn_jax_device, jnp.float32)
    beta = _to_jax_tensor(gdn_inputs["beta"], gdn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device, jnp.float32)

    out_tri, state_tri = gdn_triton_inference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = gdn_native_inference(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton inference chunk_size=8 vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton inference chunk_size=8 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_gdn_triton_single_step_chunk_size(gdn_inputs, gdn_jax_device):
    """JAX Triton recurrent 单步 RNN 接受非默认 chunk_size（参数忽略，仅签名一致）。"""
    chunk_size = 8
    q = _to_jax_tensor(gdn_inputs["q"][:, 0], gdn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_inputs["k"][:, 0], gdn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_inputs["v"][:, 0], gdn_jax_device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_inputs["g"][:, 0], gdn_jax_device, jnp.float32)
    beta = _to_jax_tensor(gdn_inputs["beta"][:, 0], gdn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(gdn_inputs["h0"], gdn_jax_device, jnp.float32)

    out_tri, state_tri = gdn_triton_single_step(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = gdn_native_single_step(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton single_step chunk_size=8 vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton single_step chunk_size=8 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )

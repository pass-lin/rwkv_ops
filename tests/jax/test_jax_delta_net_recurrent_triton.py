"""DeltaNet recurrent Triton kernel 的 JAX 后端测试。"""

import pytest

pytest.importorskip("jax")
pytest.importorskip("jax_triton")

import jax
import jax.numpy as jnp

from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent,
    delta_net_recurrent_inference as dn_native_inference,
    delta_net_recurrent_single_step as dn_native_single_step,
)
from rwkv_ops.delta_net_recurrent.jax_triton_kernel import (
    delta_net_recurrent as dn_triton_recurrent,
    delta_net_recurrent_inference as dn_triton_inference,
    delta_net_recurrent_single_step as dn_triton_single_step,
)
from tests.conftest import assert_allclose_with_stats


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def dn_jax_device():
    """JAX Triton recurrent kernel 需要 GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("DeltaNet recurrent Triton kernel requires JAX GPU.")
    return jax.devices()[0]


def _dn_value_and_grad(fn, q, k, v, beta, h0):
    """计算 DeltaNet recurrent 算子对 q/k/v/beta/h0 的数值与梯度。"""

    def loss_fn(q, k, v, beta, h0):
        out, state = fn(q, k, v, beta, initial_state=h0, output_final_state=True)
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, h0)


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_triton_recurrent_matches_native(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 训练算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_triton_inference_matches_native(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 推理算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_tri, state_tri = dn_triton_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_triton_single_step_matches_native(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 单步 RNN 算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"][:, 0], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"][:, 0], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"][:, 0], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, 0], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_tri, state_tri = dn_triton_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_triton_recurrent_backward(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    _, grads_ref = _dn_value_and_grad(dn_native_recurrent, q, k, v, beta, h0)
    _, grads_tri = _dn_value_and_grad(dn_triton_recurrent, q, k, v, beta, h0)

    names = ["q", "k", "v", "beta", "h0"]
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
def test_delta_net_triton_recurrent_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 训练算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_tri, state_tri = dn_triton_recurrent(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = dn_native_recurrent(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton recurrent chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton recurrent chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_triton_recurrent_backward_chunk_size(
    delta_net_inputs, dn_jax_device
):
    """JAX Triton recurrent 训练算子非默认 chunk_size 反向梯度与 native 对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    def loss_fn(fn, q, k, v, beta, h0):
        out, state = fn(
            q,
            k,
            v,
            beta,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    _, grads_ref = jax.value_and_grad(
        lambda q, k, v, beta, h0: loss_fn(dn_native_recurrent, q, k, v, beta, h0),
        argnums=(0, 1, 2, 3, 4),
    )(q, k, v, beta, h0)
    _, grads_tri = jax.value_and_grad(
        lambda q, k, v, beta, h0: loss_fn(dn_triton_recurrent, q, k, v, beta, h0),
        argnums=(0, 1, 2, 3, 4),
    )(q, k, v, beta, h0)

    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gt in zip(names, grads_ref, grads_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"grad_{name} triton chunk_size=8 vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_delta_net_triton_inference_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 推理算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_tri, state_tri = dn_triton_inference(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = dn_native_inference(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton inference chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton inference chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_delta_net_triton_single_step_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX Triton recurrent 单步 RNN 接受非默认 chunk_size（参数忽略，仅签名一致）。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"][:, 0], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"][:, 0], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"][:, 0], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, 0], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_tri, state_tri = dn_triton_single_step(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_ref, state_ref = dn_native_single_step(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "triton single_step chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "triton single_step chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )

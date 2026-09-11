"""DeltaNet recurrent CUDA kernel 的 JAX 后端测试。"""

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, PartitionSpec

from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent,
    delta_net_recurrent_inference as dn_native_inference,
    delta_net_recurrent_single_step as dn_native_single_step,
)
from rwkv_ops.delta_net_recurrent.jax_cuda_kernel.delta_net_recurrent_jax import (
    delta_net_recurrent as dn_cuda_recurrent,
    delta_net_recurrent_inference as dn_cuda_inference,
    delta_net_recurrent_single_step as dn_cuda_single_step,
)
from tests.conftest import assert_allclose_with_stats


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def dn_jax_device():
    """JAX CUDA recurrent kernel 需要 GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("DeltaNet recurrent CUDA kernel requires JAX GPU.")
    return jax.devices()[0]


def _dn_value_and_grad(fn, q, k, v, beta, h0):
    """计算 DeltaNet recurrent 算子对 q/k/v/beta/h0 的数值与梯度。"""

    def loss_fn(q, k, v, beta, h0):
        out, state = fn(q, k, v, beta, initial_state=h0, output_final_state=True)
        return jnp.mean(out.astype(jnp.float32) ** 2) + jnp.mean(state**2)

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q, k, v, beta, h0)


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_cuda_recurrent_matches_native(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_cuda_inference_matches_native(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 推理算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_cuda_single_step_matches_native(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 单步 RNN 算子前向结果与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"][:, 0], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"][:, 0], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"][:, 0], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, 0], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_cuda, state_cuda = dn_cuda_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_cuda_bfloat16(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子 bf16 前向与 native 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda recurrent bf16 vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda recurrent bf16 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_cuda_recurrent_backward(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    _, grads_ref = _dn_value_and_grad(dn_native_recurrent, q, k, v, beta, h0)
    _, grads_cuda = _dn_value_and_grad(dn_cuda_recurrent, q, k, v, beta, h0)

    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, grads_ref, grads_cuda):
        assert_allclose_with_stats(
            gr,
            gc,
            f"grad_{name} cuda vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_delta_net_cuda_recurrent_head_first(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子 head_first 布局与默认布局结果一致。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_ref, state_ref = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_hf, state_hf = dn_cuda_recurrent(
        jnp.transpose(q, (0, 2, 1, 3)),
        jnp.transpose(k, (0, 2, 1, 3)),
        jnp.transpose(v, (0, 2, 1, 3)),
        jnp.transpose(beta, (0, 2, 1)),
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )

    assert_allclose_with_stats(
        out_ref, out_hf, "cuda head_first vs default output", atol=1e-5, rtol=1e-4
    )
    assert_allclose_with_stats(
        state_ref, state_hf, "cuda head_first vs default state", atol=1e-5, rtol=1e-4
    )


@pytest.mark.jax
def test_delta_net_cuda_no_final_state(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent output_final_state=False 时只返回输出张量。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)

    out_only = dn_cuda_recurrent(q, k, v, beta, output_final_state=False)
    out_full, _ = dn_cuda_recurrent(q, k, v, beta, output_final_state=True)

    assert_allclose_with_stats(
        out_full, out_only, "cuda no_final_state output", atol=1e-5, rtol=1e-4
    )


@pytest.mark.jax
def test_delta_net_cuda_initial_state_broadcast(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 支持 [1, H, K, V] 初始 state 广播。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"][:1], dn_jax_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda broadcast h0 vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda broadcast h0 vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_cuda_rejects_arbitrary_length(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子拒绝不被 chunk_size 整除的序列长度。"""
    q = _to_jax_tensor(delta_net_inputs["q"][:, :120], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"][:, :120], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"][:, :120], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, :120], dn_jax_device)

    with pytest.raises(ValueError):
        dn_cuda_recurrent(q, k, v, beta, output_final_state=False)


@pytest.mark.jax
def test_delta_net_cuda_inference_arbitrary_length(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 推理算子支持不被 chunk_size 整除的序列长度。"""
    q = _to_jax_tensor(delta_net_inputs["q"][:, :34], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"][:, :34], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"][:, :34], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, :34], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda inference T=34 vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda inference T=34 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_cuda_recurrent_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_cuda, state_cuda = dn_cuda_recurrent(
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
        out_cuda,
        "cuda recurrent chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda recurrent chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_cuda_recurrent_backward_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 训练算子非默认 chunk_size 反向梯度与 native 对齐。"""
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
    _, grads_cuda = jax.value_and_grad(
        lambda q, k, v, beta, h0: loss_fn(dn_cuda_recurrent, q, k, v, beta, h0),
        argnums=(0, 1, 2, 3, 4),
    )(q, k, v, beta, h0)

    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, grads_ref, grads_cuda):
        assert_allclose_with_stats(
            gr,
            gc,
            f"grad_{name} cuda chunk_size=8 vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_delta_net_cuda_inference_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 推理算子非默认 chunk_size 与 native 参考对齐。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_cuda, state_cuda = dn_cuda_inference(
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
        out_cuda,
        "cuda inference chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda inference chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_delta_net_cuda_single_step_chunk_size(delta_net_inputs, dn_jax_device):
    """JAX CUDA recurrent 单步 RNN 接受非默认 chunk_size（参数忽略，仅签名一致）。"""
    chunk_size = 8
    q = _to_jax_tensor(delta_net_inputs["q"][:, 0], dn_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_inputs["k"][:, 0], dn_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_inputs["v"][:, 0], dn_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(delta_net_inputs["beta"][:, 0], dn_jax_device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device, jnp.float32)

    out_cuda, state_cuda = dn_cuda_single_step(
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
        out_cuda,
        "cuda single_step chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda single_step chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_delta_net_cuda_recurrent_sharding_structure(delta_net_inputs, dn_jax_device):
    """1-device mesh 结构验证：jit + NamedSharding 编译通过且数值一致。"""
    q = _to_jax_tensor(delta_net_inputs["q"], dn_jax_device)
    k = _to_jax_tensor(delta_net_inputs["k"], dn_jax_device)
    v = _to_jax_tensor(delta_net_inputs["v"], dn_jax_device)
    beta = _to_jax_tensor(delta_net_inputs["beta"], dn_jax_device)
    h0 = _to_jax_tensor(delta_net_inputs["h0"], dn_jax_device)

    mesh = jax.make_mesh((1,), ("data",))
    sharding_b = NamedSharding(mesh, PartitionSpec("data", None, None, None))
    sharding_beta = NamedSharding(mesh, PartitionSpec("data", None, None))
    sharding_s = NamedSharding(mesh, PartitionSpec("data", None, None, None))

    qs = jax.device_put(q, sharding_b)
    ks = jax.device_put(k, sharding_b)
    vs = jax.device_put(v, sharding_b)
    bs = jax.device_put(beta, sharding_beta)
    hs = jax.device_put(h0, sharding_s)

    out_jit, state_jit = jax.jit(
        lambda q, k, v, beta, h0: dn_cuda_recurrent(
            q, k, v, beta, initial_state=h0, output_final_state=True
        )
    )(qs, ks, vs, bs, hs)
    out_ref, state_ref = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_jit, "cuda sharded vs unsharded output", atol=1e-5, rtol=1e-4
    )
    assert_allclose_with_stats(
        state_ref, state_jit, "cuda sharded vs unsharded state", atol=1e-5, rtol=1e-4
    )

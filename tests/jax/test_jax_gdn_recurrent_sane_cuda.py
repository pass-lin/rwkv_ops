"""Gated DeltaNet recurrent SANE CUDA kernel 的 JAX 后端测试。"""

import warnings

import pytest

pytest.importorskip("jax")

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec

from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent as gdn_native_recurrent,
    gated_delta_net_recurrent_single_step as gdn_native_single_step,
)
from rwkv_ops.gdn_recurrent_sane.jax_cuda_kernel.gdn_recurrent_sane_jax import (
    gated_delta_net_recurrent_sane as gdn_cuda_sane,
    gated_delta_net_recurrent_sane_inference as gdn_cuda_sane_inference,
    gated_delta_net_recurrent_sane_single_step as gdn_cuda_sane_single_step,
)
from rwkv_ops.gdn_recurrent_sane.native_keras_op import (
    gated_delta_net_recurrent_sane as gdn_native_sane,
    gated_delta_net_recurrent_sane_inference as gdn_native_sane_inference,
    gated_delta_net_recurrent_sane_single_step as gdn_native_sane_single_step,
)
from tests.conftest import assert_allclose_with_stats


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def gdn_sane_jax_cuda_device():
    """JAX CUDA recurrent SANE kernel 需要 GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("Gated DeltaNet recurrent SANE CUDA kernel requires JAX GPU.")
    return jax.devices()[0]


def _prepare_sane_inputs(gdn_sane_inputs, device, dtype="float32"):
    """把 gdn_sane_inputs fixture 转成 JAX 测试张量。"""
    q = _to_jax_tensor(gdn_sane_inputs["q"], device, dtype)
    k = _to_jax_tensor(gdn_sane_inputs["k"], device, dtype)
    v = _to_jax_tensor(gdn_sane_inputs["v"], device, dtype)
    g = _to_jax_tensor(gdn_sane_inputs["g"], device, jnp.float32)
    beta = _to_jax_tensor(gdn_sane_inputs["beta"], device, jnp.float32)
    tau = _to_jax_tensor(gdn_sane_inputs["tau"], device, jnp.float32)
    mask = _to_jax_tensor(gdn_sane_inputs["mask"], device, jnp.float32)
    h0 = _to_jax_tensor(gdn_sane_inputs["h0"], device, jnp.float32)
    return q, k, v, g, beta, tau, mask, h0


def _sane_loss_fn(
    op, q, k, v, g, beta, tau, mask, h0, output_final_state=True, chunk_size=16
):
    """统一的 SANE 损失函数，用于反向梯度测试。"""
    out, state = op(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=output_final_state,
        chunk_size=chunk_size,
    )
    loss = jnp.mean(jnp.asarray(out, jnp.float32) ** 2)
    if state is not None:
        loss = loss + jnp.mean(jnp.asarray(state, jnp.float32) ** 2)
    return loss


def _make_tau_mask_for_chunk_size(gdn_sane_inputs, chunk_size, device):
    """为指定 chunk_size 重新生成 tau 与 mask。"""
    B, T, H, _ = gdn_sane_inputs["q"].shape
    C = T // chunk_size
    rng = np.random.default_rng(42 + chunk_size)
    x = rng.standard_normal((B, max(C, 1), H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    mask = rng.integers(0, 2, (B, max(C, 1))).astype(np.float32)
    return (
        _to_jax_tensor(tau.astype(np.float32), device),
        _to_jax_tensor(mask, device),
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_forward_matches_native(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """JAX CUDA SANE 训练算子前向/最终 state 与 native Keras 参考对齐（fp32）。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="float32"
    )

    out_ref, state_ref = gdn_native_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_cuda, state_cuda = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda sane vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda sane vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_backward_matches_native(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """JAX CUDA SANE 训练算子反向梯度（含 dtau）与 native 参考对齐（fp32）。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="float32"
    )

    ref_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    cuda_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_cuda_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, gr, gc in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(
            gr, gc, f"grad_{name} cuda sane vs native", atol=7e-3, rtol=1e-2
        )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_bfloat16_io(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """bfloat16 I/O 下 JAX CUDA SANE 训练算子前向与反向仍与 native 参考对齐。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_native_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_cuda, state_cuda = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert out_cuda.dtype == jnp.bfloat16
    assert_allclose_with_stats(
        out_ref, out_cuda, "bf16 cuda sane vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "bf16 cuda sane vs native state", atol=1e-2, rtol=1e-2
    )

    ref_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    cuda_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_cuda_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, gr, gc in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(
            gr, gc, f"bf16 grad_{name} cuda sane vs native", atol=1e-2, rtol=1e-2
        )


@pytest.mark.jax
def test_gdn_cuda_sane_no_mask_warning_and_none_state(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """mask=None 且 output_final_state=True 时发出 UserWarning 并返回 None state。"""
    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, final_state = gdn_cuda_sane(
            q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert final_state is None
    assert out.shape == q.shape[:-1] + (v.shape[-1],)


@pytest.mark.jax
def test_gdn_cuda_sane_all_one_mask_equals_no_mask(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """全 1 mask 与 mask=None 的输出一致，且后者返回 None state。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    all_one_mask = jnp.ones((B, T // 16), dtype=jnp.float32)

    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = gdn_cuda_sane(
            q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )

    y_all_one, s_all_one = gdn_cuda_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=all_one_mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert s_no_mask is None
    assert s_all_one is not None
    assert_allclose_with_stats(
        y_no_mask, y_all_one, "no_mask vs all_one_mask output", atol=2e-5, rtol=1e-5
    )


@pytest.mark.jax
def test_gdn_cuda_sane_all_zero_mask_matches_non_sane(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """全 0 mask 等价于不使用 SANE 的 GDN recurrent。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    zero_mask = jnp.zeros((B, T // 16), dtype=jnp.float32)

    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_cuda, state_cuda = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=zero_mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "all_zero_mask vs non-sane output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "all_zero_mask vs non-sane state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_gdn_cuda_sane_inference_matches_native(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """JAX CUDA SANE 推理算子前向/最终 state 与 native 参考对齐。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_cuda, state_cuda = gdn_cuda_sane_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_sane_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda sane inference vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane inference vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_inference_arbitrary_length(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """JAX CUDA SANE 推理算子支持 T 不被 16 整除的任意长度。"""
    T = 34
    device = gdn_sane_jax_cuda_device
    q = _to_jax_tensor(gdn_sane_inputs["q"][:, :T], device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_sane_inputs["k"][:, :T], device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_sane_inputs["v"][:, :T], device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_sane_inputs["g"][:, :T], device, jnp.float32)
    beta = _to_jax_tensor(gdn_sane_inputs["beta"][:, :T], device, jnp.float32)
    tau = _to_jax_tensor(gdn_sane_inputs["tau"][:, : (T // 16), :], device, jnp.float32)
    mask = _to_jax_tensor(gdn_sane_inputs["mask"][:, : (T // 16)], device, jnp.float32)
    h0 = _to_jax_tensor(gdn_sane_inputs["h0"], device, jnp.float32)

    out_cuda, state_cuda = gdn_cuda_sane_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_sane_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert out_cuda.shape == out_ref.shape
    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "cuda sane inference arbitrary length output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane inference arbitrary length state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_single_step(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """JAX CUDA SANE 单步 RNN：do_sane=1 对 native SANE，do_sane=0 对原 GDN。"""
    device = gdn_sane_jax_cuda_device
    q = _to_jax_tensor(gdn_sane_inputs["q"][:, 0], device, jnp.bfloat16)
    k = _to_jax_tensor(gdn_sane_inputs["k"][:, 0], device, jnp.bfloat16)
    v = _to_jax_tensor(gdn_sane_inputs["v"][:, 0], device, jnp.bfloat16)
    g = _to_jax_tensor(gdn_sane_inputs["g"][:, 0], device, jnp.float32)
    beta = _to_jax_tensor(gdn_sane_inputs["beta"][:, 0], device, jnp.float32)
    tau = _to_jax_tensor(gdn_sane_inputs["tau"][:, 0, :], device, jnp.float32)
    h0 = _to_jax_tensor(gdn_sane_inputs["h0"], device, jnp.float32)
    B = gdn_sane_inputs["q"].shape[0]

    do_sane = jnp.full((B,), 1.0, dtype=jnp.float32)
    out_cuda, state_cuda = gdn_cuda_sane_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_sane_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "cuda sane single_step do_sane=1 output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane single_step do_sane=1 state",
        atol=1e-2,
        rtol=1e-2,
    )

    do_sane = jnp.full((B,), 0.0, dtype=jnp.float32)
    out_cuda, state_cuda = gdn_cuda_sane_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "cuda sane single_step do_sane=0 vs non-sane output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane single_step do_sane=0 vs non-sane state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_head_first(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """JAX CUDA SANE 训练算子 head_first 布局与默认布局结果一致。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_hf, state_hf = gdn_cuda_sane(
        jnp.transpose(q, (0, 2, 1, 3)),
        jnp.transpose(k, (0, 2, 1, 3)),
        jnp.transpose(v, (0, 2, 1, 3)),
        jnp.transpose(g, (0, 2, 1)),
        jnp.transpose(beta, (0, 2, 1)),
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )

    assert_allclose_with_stats(
        out_ref, out_hf, "cuda sane head_first vs default output", atol=1e-5, rtol=1e-4
    )
    assert_allclose_with_stats(
        state_ref,
        state_hf,
        "cuda sane head_first vs default state",
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_rejects_arbitrary_length(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """JAX CUDA SANE 训练算子拒绝不被 chunk_size 整除的序列长度。"""
    device = gdn_sane_jax_cuda_device
    q = _to_jax_tensor(gdn_sane_inputs["q"][:, :120], device)
    k = _to_jax_tensor(gdn_sane_inputs["k"][:, :120], device)
    v = _to_jax_tensor(gdn_sane_inputs["v"][:, :120], device)
    g = _to_jax_tensor(gdn_sane_inputs["g"][:, :120], device)
    beta = _to_jax_tensor(gdn_sane_inputs["beta"][:, :120], device)
    tau = _to_jax_tensor(gdn_sane_inputs["tau"], device)

    with pytest.raises(ValueError):
        gdn_cuda_sane(q, k, v, g, beta, tau, output_final_state=False)


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_forward_chunk_size(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """JAX CUDA SANE 训练算子非默认 chunk_size 前向/最终 state 与 native 对齐。"""
    chunk_size = 8
    q, k, v, g, beta, _, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        gdn_sane_inputs, chunk_size, gdn_sane_jax_cuda_device
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_cuda, state_cuda = gdn_cuda_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "cuda sane chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_backward_chunk_size(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """JAX CUDA SANE 训练算子非默认 chunk_size 反向梯度（含 dtau）与 native 对齐。"""
    chunk_size = 8
    q, k, v, g, beta, _, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        gdn_sane_inputs, chunk_size, gdn_sane_jax_cuda_device
    )

    ref_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, g, beta, tau, mask, h0, chunk_size=chunk_size
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    cuda_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_cuda_sane, q, k, v, g, beta, tau, mask, h0, chunk_size=chunk_size
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, gr, gc in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(
            gr,
            gc,
            f"grad_{name} cuda sane chunk_size=8 vs native",
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_gdn_cuda_sane_inference_chunk_size(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """JAX CUDA SANE 推理算子非默认 chunk_size 与 native 对齐。"""
    chunk_size = 8
    q, k, v, g, beta, _, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        gdn_sane_inputs, chunk_size, gdn_sane_jax_cuda_device
    )

    out_ref, state_ref = gdn_native_sane_inference(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_cuda, state_cuda = gdn_cuda_sane_inference(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "cuda sane inference chunk_size=8 vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane inference chunk_size=8 vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_irregular_padding(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """不规则 padding：实际长度 34，pad 到 48，padding chunk mask=0。

    padding 位取 q/k/v=0、g=0、beta=0（k_hat=0 且 exp(g)=1，state 冻结），
    最终 state 应等于前 32 个 token 训练（mask 全 1）后再跑 token 32/33。
    """
    device = gdn_sane_jax_cuda_device
    B, T, H, _ = gdn_sane_inputs["q"].shape
    actual_len = 34
    pad_len = ((actual_len + 15) // 16) * 16
    assert pad_len <= T

    q_np = gdn_sane_inputs["q"][:, :pad_len].copy()
    k_np = gdn_sane_inputs["k"][:, :pad_len].copy()
    v_np = gdn_sane_inputs["v"][:, :pad_len].copy()
    g_np = gdn_sane_inputs["g"][:, :pad_len].copy()
    beta_np = gdn_sane_inputs["beta"][:, :pad_len].copy()
    for arr in (q_np, k_np, v_np, g_np, beta_np):
        arr[:, actual_len:] = 0.0

    tau_np = gdn_sane_inputs["tau"][:, : pad_len // 16].copy()
    mask_np = np.ones((B, pad_len // 16), dtype=np.float32)
    mask_np[:, actual_len // 16 :] = 0.0

    q = _to_jax_tensor(q_np, device, jnp.bfloat16)
    k = _to_jax_tensor(k_np, device, jnp.bfloat16)
    v = _to_jax_tensor(v_np, device, jnp.bfloat16)
    g = _to_jax_tensor(g_np, device, jnp.float32)
    beta = _to_jax_tensor(beta_np, device, jnp.float32)
    tau = _to_jax_tensor(tau_np, device, jnp.float32)
    mask = _to_jax_tensor(mask_np, device, jnp.float32)
    h0 = _to_jax_tensor(gdn_sane_inputs["h0"], device, jnp.float32)

    _, state_cuda = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    # 参考：native SANE 训练前 32 个 token（mask 全 1），再用非 SANE native
    # recurrent 跑 token 32/33（chunk 边界 mask=0，不触发 SANE）。
    _, state_ref = gdn_native_sane(
        q[:, :32],
        k[:, :32],
        v[:, :32],
        g[:, :32],
        beta[:, :32],
        tau[:, :2],
        mask=jnp.ones((B, 2), dtype=jnp.float32),
        initial_state=h0,
        output_final_state=True,
    )
    _, state_ref = gdn_native_recurrent(
        q[:, 32:actual_len],
        k[:, 32:actual_len],
        v[:, 32:actual_len],
        g[:, 32:actual_len],
        beta[:, 32:actual_len],
        initial_state=state_ref,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "cuda sane irregular_padding state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_sharding_structure(gdn_sane_inputs, gdn_sane_jax_cuda_device):
    """1-device mesh 结构验证：jit + NamedSharding 编译通过且数值一致。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    mesh = jax.make_mesh((1,), ("data",))
    sharding_b = NamedSharding(mesh, PartitionSpec("data", None, None, None))
    sharding_g = NamedSharding(mesh, PartitionSpec("data", None, None))
    sharding_m = NamedSharding(mesh, PartitionSpec("data", None))

    qs = jax.device_put(q, sharding_b)
    ks = jax.device_put(k, sharding_b)
    vs = jax.device_put(v, sharding_b)
    gs = jax.device_put(g, sharding_g)
    bs = jax.device_put(beta, sharding_g)
    ts = jax.device_put(tau, sharding_g)
    ms = jax.device_put(mask, sharding_m)
    hs = jax.device_put(h0, sharding_b)

    out_jit, state_jit = jax.jit(
        lambda q, k, v, g, beta, tau, mask, h0: gdn_cuda_sane(
            q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
        )
    )(qs, ks, vs, gs, bs, ts, ms, hs)
    out_ref, state_ref = gdn_cuda_sane(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_jit, "cuda sane sharded vs unsharded output", atol=1e-5, rtol=1e-4
    )
    assert_allclose_with_stats(
        state_ref,
        state_jit,
        "cuda sane sharded vs unsharded state",
        atol=1e-5,
        rtol=1e-4,
    )


@pytest.mark.jax
def test_gdn_cuda_sane_no_mask_fwd_matches_native(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """mask=None 时 no-mask 内核的无条件 SANE 前向与 native 对拍（bf16）。"""
    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )
    out_cuda, state_cuda = gdn_cuda_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )

    assert state_ref is None
    assert state_cuda is None
    assert_allclose_with_stats(
        out_ref, out_cuda, "no_mask cuda vs native output", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
@pytest.mark.slow
def test_gdn_cuda_sane_no_mask_bwd_matches_native(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """no-mask 内核反向（含 tau 梯度）与 native 无条件分支对拍（fp32）。"""
    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="float32"
    )

    ref_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, g, beta, tau, None, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)
    cuda_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            gdn_cuda_sane, q, k, v, g, beta, tau, None, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, gr, gt in zip(names, ref_grads, cuda_grads):
        assert_allclose_with_stats(gr, gt, f"no_mask bwd {name}", atol=1e-2, rtol=1e-2)


@pytest.mark.jax
def test_gdn_cuda_sane_output_final_state_false_ignores_mask(
    gdn_sane_inputs, gdn_sane_jax_cuda_device
):
    """output_final_state=False 时即便提供 mask 也走无条件 SANE 的 no-mask 内核。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_sane_jax_cuda_device, dtype="bfloat16"
    )

    out_with_mask, _ = gdn_cuda_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=False,
        chunk_size=16,
    )
    out_no_mask, _ = gdn_cuda_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=False,
        chunk_size=16,
    )
    out_ref, _ = gdn_native_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=False,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_no_mask,
        out_with_mask,
        "output_final_state=False ignores mask",
        atol=1e-3,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        out_ref,
        out_with_mask,
        "output_final_state=False vs native",
        atol=1e-2,
        rtol=1e-2,
    )

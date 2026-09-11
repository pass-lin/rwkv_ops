"""Gated DeltaNet recurrent SANE Pallas kernel 的 JAX 后端测试。"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax.experimental.pallas")

from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent,
)
from rwkv_ops.delta_net_recurrent_sane.jax_pallas_kernel import (
    delta_net_recurrent_sane as gdn_pallas_recurrent,
    delta_net_recurrent_sane_inference as gdn_pallas_inference,
    delta_net_recurrent_sane_single_step as gdn_pallas_single_step,
)
from rwkv_ops.delta_net_recurrent_sane.native_keras_op import (
    delta_net_recurrent_sane as gdn_native_sane,
    delta_net_recurrent_sane_inference as gdn_native_sane_inference,
    delta_net_recurrent_sane_single_step as gdn_native_sane_single_step,
)


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def dn_sane_jax_device():
    """JAX Pallas SANE kernel 需要 GPU/TPU，否则跳过整个文件。"""
    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("Gated DeltaNet recurrent SANE Pallas kernel requires JAX GPU/TPU.")
    return jax.devices()[0]


def _prepare_sane_inputs(delta_net_sane_inputs, device, dtype="float32"):
    """把 delta_net_sane_inputs fixture 转成 JAX 测试张量。

    Args:
        delta_net_sane_inputs: dict, 来自 conftest 的 numpy 输入。
        device: jax.Device。
        dtype: str, q/k/v 的目标 dtype。

    Returns:
        tuple: (q, k, v, beta, tau, mask, h0)。
    """
    q = _to_jax_tensor(delta_net_sane_inputs["q"], device, dtype)
    k = _to_jax_tensor(delta_net_sane_inputs["k"], device, dtype)
    v = _to_jax_tensor(delta_net_sane_inputs["v"], device, dtype)
    beta = _to_jax_tensor(delta_net_sane_inputs["beta"], device, jnp.float32)
    tau = _to_jax_tensor(delta_net_sane_inputs["tau"], device, jnp.float32)
    mask = _to_jax_tensor(delta_net_sane_inputs["mask"], device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_sane_inputs["h0"], device, jnp.float32)
    return q, k, v, beta, tau, mask, h0


def _sane_loss_fn(
    op, q, k, v, beta, tau, mask, h0, output_final_state=True, chunk_size=16
):
    """统一的 SANE 损失函数，用于反向梯度测试。"""
    out, state = op(
        q,
        k,
        v,
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


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_forward_matches_native(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """JAX Pallas SANE 训练算子前向/最终 state 与 native Keras 参考对齐。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    out_pallas, state_pallas = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "pallas sane vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas sane vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_backward_matches_native(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """JAX Pallas SANE 训练算子反向梯度（含 dtau）与 native 参考对齐。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    ref_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, beta, tau, h0)

    pallas_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_pallas_recurrent, q, k, v, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, beta, tau, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, gr, gp in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            gr,
            gp,
            f"grad_{name} pallas vs native",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
def test_dn_pallas_sane_no_mask_warning_and_none_state(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """mask=None 且 output_final_state=True 时发出 UserWarning 并返回 None state。"""
    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, final_state = gdn_pallas_recurrent(
            q,
            k,
            v,
            beta,
            tau,
            mask=None,
            initial_state=h0,
            output_final_state=True,
        )
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert final_state is None
    assert out.shape == q.shape[:-1] + (v.shape[-1],)


@pytest.mark.jax
def test_dn_pallas_sane_all_one_mask_equals_no_mask(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """全 1 mask 与 mask=None 的输出一致，且后者返回 None state。"""
    B, T, _, _ = delta_net_sane_inputs["q"].shape
    all_one_mask = jnp.ones((B, T // 16), dtype=jnp.float32)

    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = gdn_pallas_recurrent(
            q,
            k,
            v,
            beta,
            tau,
            mask=None,
            initial_state=h0,
            output_final_state=True,
        )

    y_all_one, s_all_one = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=all_one_mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert s_no_mask is None
    assert s_all_one is not None
    assert_allclose_with_stats(
        y_no_mask,
        y_all_one,
        "no_mask vs all_one_mask output",
        atol=1e-5,
        rtol=1e-5,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_all_zero_mask_matches_non_sane(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """全 0 mask 等价于不使用 SANE 的 GDN recurrent。"""
    B, T, _, _ = delta_net_sane_inputs["q"].shape
    zero_mask = jnp.zeros((B, T // 16), dtype=jnp.float32)

    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_pallas, state_pallas = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=zero_mask,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "all_zero_mask vs non-sane output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "all_zero_mask vs non-sane state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_dn_pallas_sane_inference_matches_native(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """JAX Pallas SANE 推理算子前向/最终 state 与 native 参考对齐。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_pallas, state_pallas = gdn_pallas_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = gdn_native_sane_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "pallas sane inference vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas sane inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_dn_pallas_sane_inference_arbitrary_length(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """JAX Pallas SANE 推理算子支持 T 不被 16 整除的任意长度。"""
    T = 34
    q_np = delta_net_sane_inputs["q"][:, :T]
    k_np = delta_net_sane_inputs["k"][:, :T]
    v_np = delta_net_sane_inputs["v"][:, :T]
    beta_np = delta_net_sane_inputs["beta"][:, :T]
    tau_np = delta_net_sane_inputs["tau"][:, : (T // 16), :]
    mask_np = delta_net_sane_inputs["mask"][:, : (T // 16)]
    h0_np = delta_net_sane_inputs["h0"]

    q = _to_jax_tensor(q_np, dn_sane_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(k_np, dn_sane_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(v_np, dn_sane_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(beta_np, dn_sane_jax_device, jnp.float32)
    tau = _to_jax_tensor(tau_np, dn_sane_jax_device, jnp.float32)
    mask = _to_jax_tensor(mask_np, dn_sane_jax_device, jnp.float32)
    h0 = _to_jax_tensor(h0_np, dn_sane_jax_device, jnp.float32)

    out_pallas, state_pallas = gdn_pallas_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = gdn_native_sane_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert out_pallas.shape == out_ref.shape
    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "pallas sane inference arbitrary length output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas sane inference arbitrary length state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_dn_pallas_sane_single_step(delta_net_sane_inputs, dn_sane_jax_device):
    """JAX Pallas SANE 单步 RNN 与 native 参考对齐（do_sane=1/0 两种情形）。"""
    for do_sane_val in (1.0, 0.0):
        q = _to_jax_tensor(
            delta_net_sane_inputs["q"][:, 0], dn_sane_jax_device, jnp.bfloat16
        )
        k = _to_jax_tensor(
            delta_net_sane_inputs["k"][:, 0], dn_sane_jax_device, jnp.bfloat16
        )
        v = _to_jax_tensor(
            delta_net_sane_inputs["v"][:, 0], dn_sane_jax_device, jnp.bfloat16
        )
        beta = _to_jax_tensor(
            delta_net_sane_inputs["beta"][:, 0], dn_sane_jax_device, jnp.float32
        )
        tau = _to_jax_tensor(
            delta_net_sane_inputs["tau"][:, 0, :], dn_sane_jax_device, jnp.float32
        )
        do_sane = jnp.full(
            (delta_net_sane_inputs["q"].shape[0],), do_sane_val, dtype=jnp.float32
        )
        h0 = _to_jax_tensor(
            delta_net_sane_inputs["h0"], dn_sane_jax_device, jnp.float32
        )

        out_pallas, state_pallas = gdn_pallas_single_step(
            q, k, v, beta, tau, do_sane, initial_state=h0, output_final_state=True
        )
        out_ref, state_ref = gdn_native_sane_single_step(
            q, k, v, beta, tau, do_sane, initial_state=h0, output_final_state=True
        )

        assert_allclose_with_stats(
            out_ref,
            out_pallas,
            f"pallas sane single_step do_sane={do_sane_val} output",
            atol=1e-4,
            rtol=1e-3,
        )
        assert_allclose_with_stats(
            state_ref,
            state_pallas,
            f"pallas sane single_step do_sane={do_sane_val} state",
            atol=1e-4,
            rtol=1e-3,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_bfloat16_io(delta_net_sane_inputs, dn_sane_jax_device):
    """bfloat16 I/O 下 Pallas SANE 训练算子仍与 native 参考对齐。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    out_pallas, state_pallas = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert out_pallas.dtype == jnp.bfloat16
    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "bf16 pallas sane vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "bf16 pallas sane vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_dn_pallas_sane_head_sharding(delta_net_sane_inputs, dn_sane_jax_device):
    """验证 head 维度可以沿 'h' 轴分片（TP）。单 GPU 下用 1-device mesh 模拟。"""
    devices = jax.devices()
    mesh = Mesh(devices, ("h",))

    # Pallas 内部为 head-first [B, N, T, K]/[B, N, T, V]。
    q_spec = PartitionSpec(None, "h", None, None)
    v_spec = PartitionSpec(None, "h", None, None)
    gb_spec = PartitionSpec(None, "h", None)
    tau_spec = PartitionSpec(None, "h", None)
    mask_spec = PartitionSpec(None, None)
    h0_spec = PartitionSpec(None, "h", None, None)

    q = _to_jax_tensor(delta_net_sane_inputs["q"], dn_sane_jax_device, jnp.bfloat16)
    k = _to_jax_tensor(delta_net_sane_inputs["k"], dn_sane_jax_device, jnp.bfloat16)
    v = _to_jax_tensor(delta_net_sane_inputs["v"], dn_sane_jax_device, jnp.bfloat16)
    beta = _to_jax_tensor(
        delta_net_sane_inputs["beta"], dn_sane_jax_device, jnp.float32
    )
    tau = _to_jax_tensor(delta_net_sane_inputs["tau"], dn_sane_jax_device, jnp.float32)
    mask = _to_jax_tensor(
        delta_net_sane_inputs["mask"], dn_sane_jax_device, jnp.float32
    )
    h0 = _to_jax_tensor(delta_net_sane_inputs["h0"], dn_sane_jax_device, jnp.float32)

    in_shardings = (
        NamedSharding(mesh, q_spec),  # q
        NamedSharding(mesh, q_spec),  # k
        NamedSharding(mesh, v_spec),  # v
        NamedSharding(mesh, gb_spec),  # g
        NamedSharding(mesh, gb_spec),  # beta
        NamedSharding(mesh, tau_spec),  # tau
        NamedSharding(mesh, mask_spec),  # mask
        NamedSharding(mesh, h0_spec),  # h0
    )

    def run(q, k, v, beta, tau, mask, h0):
        return gdn_pallas_recurrent(
            q,
            k,
            v,
            beta,
            tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
        )

    run_sharded = jax.jit(run, in_shardings=in_shardings)
    out_ref, state_ref = run(q, k, v, beta, tau, mask, h0)
    out_s, state_s = run_sharded(q, k, v, beta, tau, mask, h0)

    assert_allclose_with_stats(
        out_ref, out_s, "pallas sane head tp output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_s, "pallas sane head tp state", atol=1e-4, rtol=1e-3
    )
    assert out_s.sharding.mesh.axis_names == ("h",)


def _make_tau_mask_for_chunk_size(delta_net_sane_inputs, chunk_size, device):
    """为指定 chunk_size 重新生成 tau 与 mask。"""
    B, T, H, _ = delta_net_sane_inputs["q"].shape
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
def test_dn_pallas_sane_forward_chunk_size(delta_net_sane_inputs, dn_sane_jax_device):
    """JAX Pallas SANE 训练算子非默认 chunk_size 前向/最终 state 与 native 对齐。"""
    chunk_size = 32
    q, k, v, beta, _, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        delta_net_sane_inputs, chunk_size, dn_sane_jax_device
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_pallas, state_pallas = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "pallas sane chunk_size=32 vs native output",
        atol=3e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas sane chunk_size=32 vs native state",
        atol=3e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_backward_chunk_size(delta_net_sane_inputs, dn_sane_jax_device):
    """JAX Pallas SANE 训练算子非默认 chunk_size 反向梯度（含 dtau）与 native 对齐。"""
    chunk_size = 8
    q, k, v, beta, _, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        delta_net_sane_inputs, chunk_size, dn_sane_jax_device
    )

    ref_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, beta, tau, mask, h0, chunk_size=chunk_size
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, beta, tau, h0)

    pallas_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_pallas_recurrent, q, k, v, beta, tau, mask, h0, chunk_size=chunk_size
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, beta, tau, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, gr, gp in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            gr,
            gp,
            f"grad_{name} pallas sane chunk_size=32 vs native",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
def test_dn_pallas_sane_inference_chunk_size(delta_net_sane_inputs, dn_sane_jax_device):
    """JAX Pallas SANE 推理算子非默认 chunk_size 与 native 对齐。"""
    chunk_size = 8
    q, k, v, beta, _, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )
    tau, mask = _make_tau_mask_for_chunk_size(
        delta_net_sane_inputs, chunk_size, dn_sane_jax_device
    )

    out_ref, state_ref = gdn_native_sane_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    out_pallas, state_pallas = gdn_pallas_inference(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )

    assert_allclose_with_stats(
        out_ref,
        out_pallas,
        "pallas sane inference chunk_size=32 vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_pallas,
        "pallas sane inference chunk_size=32 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_dn_pallas_sane_no_mask_fwd_matches_native(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """mask=None 时 no-mask 内核的无条件 SANE 前向与 native 对拍（bf16）。"""
    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_ref, state_ref = gdn_native_sane(
        q,
        k,
        v,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )
    out_pallas, state_pallas = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=None,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )

    assert state_ref is None
    assert state_pallas is None
    assert_allclose_with_stats(
        out_ref, out_pallas, "no_mask pallas vs native output", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
@pytest.mark.slow
def test_dn_pallas_sane_no_mask_bwd_matches_native(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """no-mask 内核反向（含 tau 梯度）与 native 无条件分支对拍（fp32）。"""
    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="float32"
    )

    ref_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_native_sane, q, k, v, beta, tau, None, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, beta, tau, h0)
    pallas_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            gdn_pallas_recurrent, q, k, v, beta, tau, None, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, beta, tau, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, gr, gt in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(gr, gt, f"no_mask bwd {name}", atol=1e-2, rtol=1e-2)


@pytest.mark.jax
def test_dn_pallas_sane_output_final_state_false_ignores_mask(
    delta_net_sane_inputs, dn_sane_jax_device
):
    """output_final_state=False 时即便提供 mask 也走无条件 SANE 的 no-mask 内核。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, dn_sane_jax_device, dtype="bfloat16"
    )

    out_with_mask, _ = gdn_pallas_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=False,
        chunk_size=16,
    )
    out_no_mask, _ = gdn_pallas_recurrent(
        q,
        k,
        v,
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

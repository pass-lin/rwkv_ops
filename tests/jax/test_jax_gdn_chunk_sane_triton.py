"""Gated DeltaNet chunkwise SANE JAX-Triton 数值测试。"""

import warnings

import jax
import jax.numpy as jnp
import pytest

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax_triton")

from rwkv_ops.gdn_chunk.native_keras_op import gated_delta_net_chunk as gdn_native_chunk
from rwkv_ops.gdn_chunk_sane.native_keras_op import (
    gated_delta_net_chunk_sane as native_chunk_sane,
)
from rwkv_ops.gdn_chunk_sane.jax_triton_kernel import (
    gated_delta_net_chunk_sane as triton_chunk_sane,
)


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def gdn_chunk_sane_jax_triton_device():
    """JAX-Triton SANE chunk kernel 需要 JAX GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("Gated DeltaNet chunk SANE JAX-Triton kernel requires JAX GPU.")
    return jax.devices()[0]


def _prepare_sane_inputs(gdn_sane_inputs, device, dtype="bfloat16"):
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


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_fwd_vs_native(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """JAX-Triton SANE chunkwise 前向/最终 state 与 native 参考对拍。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    out_ref, state_ref = native_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_ref, out_triton, "chunk sane triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref,
        state_triton,
        "chunk sane triton vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_bwd_vs_native(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """JAX-Triton SANE chunkwise 反向（含 dtau）与 native Keras autograd 对拍。"""
    q, k, v, g, beta, tau, mask, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    ref_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            native_chunk_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    triton_grads = jax.grad(
        lambda q, k, v, g, beta, tau, h0: _sane_loss_fn(
            triton_chunk_sane, q, k, v, g, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5, 6),
    )(q, k, v, g, beta, tau, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, gr, gt in zip(names, ref_grads, triton_grads):
        assert_allclose_with_stats(gr, gt, f"bwd {name}", atol=2e-1, rtol=2e-1)


@pytest.mark.jax
def test_chunk_sane_triton_no_mask_warning_and_none_state(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """mask=None 且 output_final_state=True 时发出 UserWarning 并返回 None state。"""
    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, final_state = triton_chunk_sane(
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
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert final_state is None
    assert out.shape == q.shape[:-1] + (v.shape[-1],)


@pytest.mark.jax
def test_chunk_sane_triton_all_one_mask_equals_no_mask(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """全 1 mask 与 mask=None 的输出一致，且 final_state 保留。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    all_one_mask = jnp.ones((B, T // 16), dtype=jnp.float32)

    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = triton_chunk_sane(
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

    y_all_one, s_all_one = triton_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=all_one_mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )

    assert s_no_mask is None
    assert s_all_one is not None
    assert_allclose_with_stats(
        y_no_mask, y_all_one, "no_mask vs all_one_mask output", atol=2e-5, rtol=1e-5
    )


@pytest.mark.jax
def test_chunk_sane_triton_all_zero_mask_matches_non_sane(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """全 0 mask 等价于不使用 SANE 的 GDN chunk。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    zero_mask = jnp.zeros((B, T // 16), dtype=jnp.float32)

    q, k, v, g, beta, tau, _, h0 = _prepare_sane_inputs(
        gdn_sane_inputs, gdn_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    out_triton, state_triton = triton_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=zero_mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )
    out_ref, state_ref = gdn_native_chunk(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=16
    )

    assert_allclose_with_stats(
        out_ref, out_triton, "all_zero_mask vs non-sane output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_triton, "all_zero_mask vs non-sane state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_chunk_sane_triton_irregular_T_raises(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """T 不被 chunk_size 整除时 Triton 后端显式抛 ValueError。"""
    T = 34
    q = _to_jax_tensor(
        gdn_sane_inputs["q"][:, :T], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    k = _to_jax_tensor(
        gdn_sane_inputs["k"][:, :T], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    v = _to_jax_tensor(
        gdn_sane_inputs["v"][:, :T], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    g = _to_jax_tensor(
        gdn_sane_inputs["g"][:, :T], gdn_chunk_sane_jax_triton_device, jnp.float32
    )
    beta = _to_jax_tensor(
        gdn_sane_inputs["beta"][:, :T], gdn_chunk_sane_jax_triton_device, jnp.float32
    )
    tau = _to_jax_tensor(
        gdn_sane_inputs["tau"][:, : (T // 16), :],
        gdn_chunk_sane_jax_triton_device,
        jnp.float32,
    )

    with pytest.raises(ValueError, match="divisible"):
        triton_chunk_sane(
            q, k, v, g, beta, tau, mask=None, output_final_state=False, chunk_size=16
        )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_different_chunk_size(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """不同 chunk_size 下 JAX-Triton SANE 与 native 均等价。"""
    q_np, k_np, v_np = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g_np, beta_np, h0_np = (
        gdn_sane_inputs["g"],
        gdn_sane_inputs["beta"],
        gdn_sane_inputs["h0"],
    )

    q = _to_jax_tensor(q_np, gdn_chunk_sane_jax_triton_device, jnp.bfloat16)
    k = _to_jax_tensor(k_np, gdn_chunk_sane_jax_triton_device, jnp.bfloat16)
    v = _to_jax_tensor(v_np, gdn_chunk_sane_jax_triton_device, jnp.bfloat16)
    g = _to_jax_tensor(g_np, gdn_chunk_sane_jax_triton_device, jnp.float32)
    beta = _to_jax_tensor(beta_np, gdn_chunk_sane_jax_triton_device, jnp.float32)
    h0 = _to_jax_tensor(h0_np, gdn_chunk_sane_jax_triton_device, jnp.float32)

    for chunk_size in (32, 64):
        C = q_np.shape[1] // chunk_size
        tau_chunk = _to_jax_tensor(
            gdn_sane_inputs["tau"][:, :C, :],
            gdn_chunk_sane_jax_triton_device,
            jnp.float32,
        )

        out_ref, state_ref = native_chunk_sane(
            q,
            k,
            v,
            g,
            beta,
            tau_chunk,
            mask=None,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        out_triton, state_triton = triton_chunk_sane(
            q,
            k,
            v,
            g,
            beta,
            tau_chunk,
            mask=None,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
        )

        assert_allclose_with_stats(
            out_ref, out_triton, f"chunk_size={chunk_size} output", atol=1e-2, rtol=1e-2
        )
        assert_allclose_with_stats(
            state_ref,
            state_triton,
            f"chunk_size={chunk_size} state",
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_factory_dispatch(
    gdn_sane_inputs, gdn_chunk_sane_jax_triton_device
):
    """get_gated_delta_net_chunk_sane 工厂正确分发 Triton / native 后端。"""
    from rwkv_ops import get_gated_delta_net_chunk_sane

    q = _to_jax_tensor(
        gdn_sane_inputs["q"], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    k = _to_jax_tensor(
        gdn_sane_inputs["k"], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    v = _to_jax_tensor(
        gdn_sane_inputs["v"], gdn_chunk_sane_jax_triton_device, jnp.bfloat16
    )
    g = _to_jax_tensor(
        gdn_sane_inputs["g"], gdn_chunk_sane_jax_triton_device, jnp.float32
    )
    beta = _to_jax_tensor(
        gdn_sane_inputs["beta"], gdn_chunk_sane_jax_triton_device, jnp.float32
    )
    tau = _to_jax_tensor(
        gdn_sane_inputs["tau"], gdn_chunk_sane_jax_triton_device, jnp.float32
    )
    h0 = _to_jax_tensor(
        gdn_sane_inputs["h0"], gdn_chunk_sane_jax_triton_device, jnp.float32
    )

    native_op = get_gated_delta_net_chunk_sane(KERNEL_TYPE="native", chunk_size=16)
    triton_op = get_gated_delta_net_chunk_sane(KERNEL_TYPE="triton", chunk_size=16)

    out_native, _ = native_op(
        q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=False
    )
    out_triton, _ = triton_op(
        q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=False
    )

    assert_allclose_with_stats(
        out_native, out_triton, "factory dispatch output", atol=1e-2, rtol=1e-2
    )

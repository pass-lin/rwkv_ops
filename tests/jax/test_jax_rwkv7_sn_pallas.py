"""
RWKV-7 State Neutralization JAX Pallas kernel 数值测试。

运行方式：
    KERAS_BACKEND=jax pytest tests/jax/test_jax_rwkv7_sn_pallas.py -v
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax.experimental.pallas")


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _prepare_inputs(rwkv7_sn_inputs, head_first, dtype="bfloat16"):
    B, T, H, _ = rwkv7_sn_inputs["r"].shape
    if head_first:
        r = _to_jax(np.transpose(rwkv7_sn_inputs["r"], (0, 2, 1, 3)), dtype)
        k = _to_jax(np.transpose(rwkv7_sn_inputs["k"], (0, 2, 1, 3)), dtype)
        v = _to_jax(np.transpose(rwkv7_sn_inputs["v"], (0, 2, 1, 3)), dtype)
        a = _to_jax(np.transpose(rwkv7_sn_inputs["a"], (0, 2, 1, 3)), dtype)
        b = _to_jax(np.transpose(rwkv7_sn_inputs["b"], (0, 2, 1, 3)), dtype)
        w = _to_jax(np.transpose(rwkv7_sn_inputs["w"], (0, 2, 1, 3)), dtype)
    else:
        r = _to_jax(rwkv7_sn_inputs["r"], dtype)
        k = _to_jax(rwkv7_sn_inputs["k"], dtype)
        v = _to_jax(rwkv7_sn_inputs["v"], dtype)
        a = _to_jax(rwkv7_sn_inputs["a"], dtype)
        b = _to_jax(rwkv7_sn_inputs["b"], dtype)
        w = _to_jax(rwkv7_sn_inputs["w"], dtype)
    tau = _to_jax(rwkv7_sn_inputs["tau"], "float32")
    mask = jnp.ones((B, T // 16), dtype=jnp.float32)
    h0 = _to_jax(rwkv7_sn_inputs["h0"], "float32")
    return r, k, v, a, b, w, tau, mask, h0


def _test_is_close(name, ref, tgt, atol, rtol):
    ref_f = np.asarray(ref, dtype=np.float32)
    tgt_f = np.asarray(tgt, dtype=np.float32)
    diff = np.abs(ref_f - tgt_f)
    exact_rate = np.sum(diff < 1e-7) / ref_f.size * 100
    avg_err = diff.mean()
    max_diff = diff.max()
    print("-" * 80)
    print(
        f"[{name}] exact={exact_rate:.2f}%, avg_err={avg_err:.6e}, max_diff={max_diff:.6e}"
    )
    assert_allclose_with_stats(ref, tgt, name, atol=atol, rtol=rtol)


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_pallas_forward_state(
    rwkv7_sn_jax_pallas_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, mask_ref, h0_ref = (
        _prepare_inputs(rwkv7_sn_inputs, head_first, "bfloat16")
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, mask_c, h0_c = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    y_ref, s_ref = rwkv7_sn_native_op(
        r=r_ref,
        w=w_ref,
        k=k_ref,
        v=v_ref,
        a=a_ref,
        b=b_ref,
        tau=tau_ref,
        mask=mask_ref,
        initial_state=h0_ref,
        output_final_state=True,
        head_first=head_first,
    )
    y_c, s_c = rwkv7_sn_jax_pallas_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        mask=mask_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=head_first,
    )

    _test_is_close(f"y_head_first={head_first}", y_ref, y_c, atol=1e-4, rtol=1e-2)
    _test_is_close(
        f"final_state_head_first={head_first}", s_ref, s_c, atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_pallas_backward(
    rwkv7_sn_jax_pallas_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    r, k, v, a, b, w, tau, mask, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    def loss(op, params):
        w, r, k, v, a, b, tau, mask, h0 = params
        y, state = op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    ref_grads = jax.grad(lambda *p: loss(rwkv7_sn_native_op, p), argnums=range(9))(
        w, r, k, v, a, b, tau, mask, h0
    )
    pallas_grads = jax.grad(
        lambda *p: loss(rwkv7_sn_jax_pallas_op, p), argnums=range(9)
    )(w, r, k, v, a, b, tau, mask, h0)

    names = ["w", "r", "k", "v", "a", "b", "tau", "mask", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_pallas_no_mask_forward_state(
    rwkv7_sn_jax_pallas_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, _, h0_ref = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, _, h0_c = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    y_ref = rwkv7_sn_native_op(
        r=r_ref,
        w=w_ref,
        k=k_ref,
        v=v_ref,
        a=a_ref,
        b=b_ref,
        tau=tau_ref,
        initial_state=h0_ref,
        output_final_state=False,
        head_first=head_first,
    )
    y_c = rwkv7_sn_jax_pallas_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        initial_state=h0_c,
        output_final_state=False,
        head_first=head_first,
    )

    _test_is_close(
        f"y_no_mask_head_first={head_first}", y_ref, y_c, atol=1e-4, rtol=1e-2
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_pallas_no_mask_backward(
    rwkv7_sn_jax_pallas_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    r, k, v, a, b, w, tau, _, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    def loss(op, params):
        w, r, k, v, a, b, tau, h0 = params
        y = op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=False,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2)

    ref_grads = jax.grad(lambda *p: loss(rwkv7_sn_native_op, p), argnums=range(8))(
        w, r, k, v, a, b, tau, h0
    )
    pallas_grads = jax.grad(
        lambda *p: loss(rwkv7_sn_jax_pallas_op, p), argnums=range(8)
    )(w, r, k, v, a, b, tau, h0)

    names = ["w", "r", "k", "v", "a", "b", "tau", "h0"]
    for name, g_ref, g_c in zip(names, ref_grads, pallas_grads):
        assert_allclose_with_stats(
            g_ref,
            g_c,
            f"grad_no_mask_{name}_head_first={head_first}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.jax
def test_rwkv7_sn_pallas_no_mask_y_matches_all_one(
    rwkv7_sn_jax_pallas_op, rwkv7_sn_inputs
):
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    n_chunks = T // 16
    mask = jnp.ones((B, n_chunks), dtype=jnp.float32)
    r, k, v, a, b, w, tau, _, h0 = _prepare_inputs(rwkv7_sn_inputs, head_first=False)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = rwkv7_sn_jax_pallas_op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=True,
        )
        assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)

    y_all_one, s_all_one = rwkv7_sn_jax_pallas_op(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )

    assert s_no_mask is None
    pred_diff = float(jnp.max(jnp.abs(y_all_one - y_no_mask)))
    assert pred_diff < 1e-5, f"全 1 Mask 输出不一致 (max_diff={pred_diff:.3e})"
    assert s_all_one is not None


@pytest.mark.jax
def test_rwkv7_sn_pallas_head_sharding(rwkv7_sn_jax_pallas_op, rwkv7_sn_inputs):
    """验证 head 维度可以沿 'h' 轴分片（TP）。单 GPU 下用 1-device mesh 模拟。"""
    devices = jax.devices()
    mesh = Mesh(devices, ("h",))

    # pallas 内部是 head-first [B, N, T, H]
    q_spec = PartitionSpec(None, "h", None, None)
    tau_spec = PartitionSpec(None, "h", None)
    h0_spec = PartitionSpec(None, "h", None, None)
    mask_spec = PartitionSpec(None, None)

    r, k, v, a, b, w, tau, mask, h0 = _prepare_inputs(
        rwkv7_sn_inputs, head_first=False, dtype="bfloat16"
    )

    in_shardings = (
        NamedSharding(mesh, q_spec),  # r
        NamedSharding(mesh, q_spec),  # w
        NamedSharding(mesh, q_spec),  # k
        NamedSharding(mesh, q_spec),  # v
        NamedSharding(mesh, q_spec),  # a
        NamedSharding(mesh, q_spec),  # b
        NamedSharding(mesh, tau_spec),  # tau
        NamedSharding(mesh, mask_spec),  # mask
        NamedSharding(mesh, h0_spec),  # h0
    )

    def run(r, w, k, v, a, b, tau, mask, h0):
        return rwkv7_sn_jax_pallas_op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
            head_first=False,
        )

    run_sharded = jax.jit(run, in_shardings=in_shardings)
    y_ref, s_ref = run(r, w, k, v, a, b, tau, mask, h0)
    y_s, s_s = run_sharded(r, w, k, v, a, b, tau, mask, h0)

    _test_is_close("y_head_tp", y_ref, y_s, atol=1e-4, rtol=1e-2)
    _test_is_close("final_state_head_tp", s_ref, s_s, atol=1e-5, rtol=1e-3)

    assert y_s.sharding.mesh.axis_names == ("h",)

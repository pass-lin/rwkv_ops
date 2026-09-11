"""DeltaNet chunkwise SANE JAX-Triton 数值测试。"""

import warnings

import jax
import jax.numpy as jnp
import pytest
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax_triton")

from rwkv_ops.delta_net_chunk.native_keras_op import (
    delta_net_chunk as delta_native_chunk,
)
from rwkv_ops.delta_net_chunk_sane import get_delta_net_chunk_sane
from rwkv_ops.delta_net_chunk_sane.jax_triton_kernel import (
    delta_net_chunk_sane as triton_chunk_sane,
)
from rwkv_ops.delta_net_chunk_sane.native_keras_op import (
    delta_net_chunk_sane as native_chunk_sane,
)


def _to_jax_tensor(arr, device, dtype=jnp.float32):
    """把 numpy 数组转成指定 dtype 的 JAX 设备数组。"""
    return jax.device_put(jnp.asarray(arr, dtype=dtype), device)


@pytest.fixture(scope="session")
def delta_chunk_sane_jax_triton_device():
    """JAX-Triton SANE chunk kernel 需要 JAX GPU，否则跳过整个文件。"""
    if jax.devices()[0].platform != "gpu":
        pytest.skip("DeltaNet chunk SANE JAX-Triton kernel requires JAX GPU.")
    return jax.devices()[0]


def _prepare_sane_inputs(delta_net_sane_inputs, device, dtype="bfloat16"):
    """把 delta_net_sane_inputs fixture 转成 JAX 测试张量。"""
    q = _to_jax_tensor(delta_net_sane_inputs["q"], device, dtype)
    k = _to_jax_tensor(delta_net_sane_inputs["k"], device, dtype)
    v = _to_jax_tensor(delta_net_sane_inputs["v"], device, dtype)
    beta = _to_jax_tensor(delta_net_sane_inputs["beta"], device, jnp.float32)
    tau = _to_jax_tensor(delta_net_sane_inputs["tau"], device, jnp.float32)
    mask = _to_jax_tensor(delta_net_sane_inputs["mask"], device, jnp.float32)
    h0 = _to_jax_tensor(delta_net_sane_inputs["h0"], device, jnp.float32)
    return q, k, v, beta, tau, mask, h0


def _sane_loss_fn(op, q, k, v, beta, tau, mask, h0, chunk_size=16):
    """统一的 SANE 损失函数，用于反向梯度测试。"""
    out, state = op(
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
    loss = jnp.mean(jnp.asarray(out, jnp.float32) ** 2)
    if state is not None:
        loss = loss + jnp.mean(jnp.asarray(state, jnp.float32) ** 2)
    return loss


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_fwd_vs_native(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """JAX-Triton SANE chunkwise 前向/最终 state 与 native 参考对拍。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    out_ref, state_ref = native_chunk_sane(
        q,
        k,
        v,
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
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """JAX-Triton SANE chunkwise 反向（含 dtau）与 native Keras autograd 对拍。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    ref_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            native_chunk_sane, q, k, v, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, beta, tau, h0)

    triton_grads = jax.grad(
        lambda q, k, v, beta, tau, h0: _sane_loss_fn(
            triton_chunk_sane, q, k, v, beta, tau, mask, h0
        ),
        argnums=(0, 1, 2, 3, 4, 5),
    )(q, k, v, beta, tau, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    atols = [1e-2, 1e-2, 1e-2, 2e-3, 2e-3, 1e-4]
    rtols = [1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-3]
    for name, gr, gt, atol, rtol in zip(names, ref_grads, triton_grads, atols, rtols):
        assert_allclose_with_stats(gr, gt, f"bwd {name}", atol=atol, rtol=rtol)


@pytest.mark.jax
def test_chunk_sane_triton_no_mask_warning_and_none_state(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """mask=None 且 output_final_state=True 时发出 UserWarning 并返回 None state。"""
    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, final_state = triton_chunk_sane(
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
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert final_state is None
    assert out.shape == q.shape[:-1] + (v.shape[-1],)


@pytest.mark.jax
def test_chunk_sane_triton_all_one_mask_equals_no_mask(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """全 1 mask 与 mask=None 的输出一致，且 final_state 保留。"""
    B, T, _, _ = delta_net_sane_inputs["q"].shape
    all_one_mask = jnp.ones((B, T // 16), dtype=jnp.float32)

    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = triton_chunk_sane(
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

    y_all_one, s_all_one = triton_chunk_sane(
        q,
        k,
        v,
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
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """全 0 mask 等价于不使用 SANE 的 DeltaNet chunk。"""
    B, T, _, _ = delta_net_sane_inputs["q"].shape
    zero_mask = jnp.zeros((B, T // 16), dtype=jnp.float32)

    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    out_triton, state_triton = triton_chunk_sane(
        q,
        k,
        v,
        beta,
        tau,
        mask=zero_mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )
    out_ref, state_ref = delta_native_chunk(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_ref, out_triton, "all_zero_mask output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_triton, "all_zero_mask state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_chunk_sane_triton_irregular_T_raises(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """T 不被 chunk_size 整除时 Triton 后端显式抛 ValueError。"""
    T = 34
    q, k, v, beta, _, _, _ = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )
    tau = _to_jax_tensor(
        delta_net_sane_inputs["tau"][:, : (T // 16), :],
        delta_chunk_sane_jax_triton_device,
        jnp.float32,
    )
    q, k, v, beta = q[:, :T], k[:, :T], v[:, :T], beta[:, :T]

    with pytest.raises(ValueError, match="divisible"):
        triton_chunk_sane(
            q,
            k,
            v,
            beta,
            tau,
            mask=None,
            output_final_state=False,
            chunk_size=16,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_sane_triton_different_chunk_size(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """不同 chunk_size 下 JAX-Triton SANE 与 native 均等价。"""
    q, k, v, beta, tau, _, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )

    for chunk_size in (32, 64):
        C = q.shape[1] // chunk_size
        tau_chunk = tau[:, :C, :]
        out_ref, state_ref = native_chunk_sane(
            q,
            k,
            v,
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
            beta,
            tau_chunk,
            mask=None,
            initial_state=h0,
            output_final_state=True,
            chunk_size=chunk_size,
        )

        assert_allclose_with_stats(
            out_ref,
            out_triton,
            f"chunk_size={chunk_size} output",
            atol=1e-2,
            rtol=1e-2,
        )
        assert_allclose_with_stats(
            state_ref,
            state_triton,
            f"chunk_size={chunk_size} state",
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_chunk_sane_triton_head_sharding(
    delta_net_sane_inputs, delta_chunk_sane_jax_triton_device
):
    """单 device mesh 下 head 轴分片结构验证（编译通过 + 数值一致）。"""
    q, k, v, beta, tau, mask, h0 = _prepare_sane_inputs(
        delta_net_sane_inputs, delta_chunk_sane_jax_triton_device, dtype="bfloat16"
    )
    op = get_delta_net_chunk_sane(KERNEL_TYPE="triton", chunk_size=16)

    mesh = jax.make_mesh((1,), ("h",))
    q_sh = NamedSharding(mesh, P(None, None, "h", None))
    h_sh = NamedSharding(mesh, P(None, "h", None, None))
    t_sh = NamedSharding(mesh, P(None, None, "h"))
    m_sh = NamedSharding(mesh, P(None, None))

    def run(q, k, v, beta, tau, mask, h0):
        return op(
            q,
            k,
            v,
            beta,
            tau,
            mask=mask,
            initial_state=h0,
            output_final_state=True,
        )

    in_shardings = (q_sh, q_sh, q_sh, t_sh, t_sh, m_sh, h_sh)
    out_ref, state_ref = run(q, k, v, beta, tau, mask, h0)
    out_sh, state_sh = jax.jit(
        run, in_shardings=in_shardings, out_shardings=(q_sh, h_sh)
    )(q, k, v, beta, tau, mask, h0)

    assert out_sh.sharding.mesh.axis_names == ("h",)
    assert state_sh.sharding.mesh.axis_names == ("h",)
    assert out_sh.sharding.spec == q_sh.spec
    assert state_sh.sharding.spec == h_sh.spec
    assert_allclose_with_stats(
        out_ref, out_sh, "head sharding output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_sh, "head sharding state", atol=1e-2, rtol=1e-2
    )

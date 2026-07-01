"""
RWKV-7 State Norm JAX CUDA kernel 数值测试。

运行方式：
    KERAS_BACKEND=jax pytest tests/jax/test_rwkv7_sn.py -v
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.conftest import assert_allclose_with_stats


def _to_jax(arr, dtype):
    return jnp.asarray(arr, dtype=getattr(jnp, dtype))


def _prepare_inputs(rwkv7_sn_inputs, head_first, dtype="bfloat16"):
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
    h0 = _to_jax(rwkv7_sn_inputs["h0"], "float32")
    return r, k, v, a, b, w, tau, h0


@pytest.mark.jax
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_forward_state(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first
):
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, h0_ref = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, h0_c = _prepare_inputs(
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
        initial_state=h0_ref,
        output_final_state=True,
        head_first=head_first,
    )
    y_c, s_c = rwkv7_sn_jax_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=head_first,
    )

    assert_allclose_with_stats(
        y_ref, y_c, f"y_head_first={head_first}", atol=1.0, rtol=1e-1
    )
    assert_allclose_with_stats(
        s_ref, s_c, f"final_state_head_first={head_first}", atol=1.0, rtol=1e-1
    )


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("head_first", [False, True])
def test_rwkv7_sn_backward_directional(
    rwkv7_sn_jax_op, rwkv7_sn_native_op, rwkv7_sn_inputs, head_first, rng
):
    """CUDA custom_vjp 反向与 native 有限差分对比（包含 tau 梯度）。"""
    r_ref, k_ref, v_ref, a_ref, b_ref, w_ref, tau_ref, h0_ref = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "float32"
    )
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, h0_c = _prepare_inputs(
        rwkv7_sn_inputs, head_first, "bfloat16"
    )

    key = jax.random.PRNGKey(int(rng.integers(0, 2**31)))
    keys = jax.random.split(key, 8)
    dirs = [
        jax.random.normal(k, p.shape, dtype=jnp.float32)
        for k, p in zip(
            keys, [w_ref, r_ref, k_ref, v_ref, a_ref, b_ref, tau_ref, h0_ref]
        )
    ]

    def loss(op, params):
        w, r, k, v, a, b, tau, h0 = params
        y, state = op(
            r=r,
            w=w,
            k=k,
            v=v,
            a=a,
            b=b,
            tau=tau,
            initial_state=h0,
            output_final_state=True,
            head_first=head_first,
        )
        return jnp.mean(jnp.asarray(y, jnp.float32) ** 2) + jnp.mean(
            jnp.asarray(state, jnp.float32) ** 2
        )

    def directional_fd(op, params, dirs, eps=1e-3):
        plus = [p + eps * d for p, d in zip(params, dirs)]
        minus = [p - eps * d for p, d in zip(params, dirs)]
        return (loss(op, plus) - loss(op, minus)) / (2 * eps)

    ref_val = directional_fd(
        rwkv7_sn_native_op,
        [w_ref, r_ref, k_ref, v_ref, a_ref, b_ref, tau_ref, h0_ref],
        dirs,
    )

    def cuda_loss_fn(w, r, k, v, a, b, tau, h0):
        return loss(rwkv7_sn_jax_op, [w, r, k, v, a, b, tau, h0])

    grad_c = jax.grad(cuda_loss_fn, argnums=range(8))(
        w_c, r_c, k_c, v_c, a_c, b_c, tau_c, h0_c
    )
    cuda_val = sum(
        jnp.sum(jnp.asarray(g, jnp.float32) * d) for g, d in zip(grad_c, dirs)
    )

    rel = float(jnp.abs(ref_val - cuda_val) / (jnp.abs(ref_val) + 1e-8))
    print(
        f"[directional_derivative sn head_first={head_first}] ref={float(ref_val):.6e}, "
        f"cuda={float(cuda_val):.6e}, rel_diff={rel:.3e}"
    )
    assert rel < 1e-1, f"方向导数相对差异过大: {rel}"


@pytest.mark.jax
@pytest.mark.slow
def test_rwkv7_sn_rnn(
    rwkv7_sn_jax_op, rwkv7_sn_rnn_op, rwkv7_sn_rnn_native_op, rwkv7_sn_inputs
):
    """验证单步 RNN 与 native 单步在 16 步内一致。"""
    inputs = {k: v.copy() for k, v in rwkv7_sn_inputs.items()}
    B, T, H, K = inputs["r"].shape

    prefill_len = 16
    pre_inputs = {k: v[:, :prefill_len] for k, v in inputs.items()}
    pre_inputs["tau"] = inputs["tau"][:, : prefill_len // 16]
    r_c, k_c, v_c, a_c, b_c, w_c, tau_c, h0_c = _prepare_inputs(
        pre_inputs, head_first=False, dtype="bfloat16"
    )

    _, state = rwkv7_sn_jax_op(
        r=r_c,
        w=w_c,
        k=k_c,
        v=v_c,
        a=a_c,
        b=b_c,
        tau=tau_c,
        initial_state=h0_c,
        output_final_state=True,
        head_first=False,
    )

    native_state = state
    cuda_state = state

    for step in range(prefill_len):
        rr = inputs["r"][:, prefill_len + step : prefill_len + step + 1]
        kk = inputs["k"][:, prefill_len + step : prefill_len + step + 1]
        vv = inputs["v"][:, prefill_len + step : prefill_len + step + 1]
        aa = inputs["a"][:, prefill_len + step : prefill_len + step + 1]
        bb = inputs["b"][:, prefill_len + step : prefill_len + step + 1]
        ww = inputs["w"][:, prefill_len + step : prefill_len + step + 1]
        tau = inputs["tau"][:, prefill_len // 16]
        do_sn = step == 15

        r_s = _to_jax(rr, "bfloat16")
        k_s = _to_jax(kk, "bfloat16")
        v_s = _to_jax(vv, "bfloat16")
        a_s = _to_jax(aa, "bfloat16")
        b_s = _to_jax(bb, "bfloat16")
        w_s = _to_jax(ww, "bfloat16")
        tau_s = _to_jax(tau, "float32")

        cuda_y, cuda_state = rwkv7_sn_rnn_op(
            r=r_s,
            w=w_s,
            k=k_s,
            v=v_s,
            a=a_s,
            b=b_s,
            tau=tau_s,
            do_sn=do_sn,
            initial_state=cuda_state,
            output_final_state=True,
            head_first=False,
        )

        native_y, native_state = rwkv7_sn_rnn_native_op(
            r=r_s,
            w=w_s,
            k=k_s,
            v=v_s,
            a=a_s,
            b=b_s,
            tau=tau_s,
            do_sn=do_sn,
            initial_state=native_state,
            output_final_state=True,
            head_first=False,
        )

        assert_allclose_with_stats(
            native_y, cuda_y, f"rnn_y_step_{step}", atol=1.0, rtol=1e-1
        )
        assert_allclose_with_stats(
            native_state,
            cuda_state,
            f"rnn_state_step_{step}",
            atol=1.0,
            rtol=1e-1,
        )

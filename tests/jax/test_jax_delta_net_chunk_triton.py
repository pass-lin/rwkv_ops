"""DeltaNet chunkwise JAX-Triton 数值测试。"""

import jax
import jax.numpy as jnp
import pytest

from tests.conftest import assert_allclose_with_stats


@pytest.mark.jax
def test_chunk_triton_fwd_vs_native(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """JAX-Triton chunkwise 前向与 native 参考实现对拍。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)
    h0_j = jnp.asarray(h0)

    out_native, state_native = delta_net_chunk_jax_native_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = delta_net_chunk_jax_triton_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "chunk triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_native, state_triton, "chunk triton vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_chunk_triton_no_final_state(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """output_final_state=False 时不返回 state。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)

    out_native, state_native = delta_net_chunk_jax_native_op(
        q_j, k_j, v_j, beta_j, output_final_state=False, chunk_size=16
    )
    out_triton, state_triton = delta_net_chunk_jax_triton_op(
        q_j, k_j, v_j, beta_j, output_final_state=False, chunk_size=16
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native, out_triton, "no-state chunk output", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_chunk_triton_no_initial_state(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """initial_state=None 时 JAX-Triton 与 native 前向一致。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)

    out_native, state_native = delta_net_chunk_jax_native_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=None,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = delta_net_chunk_jax_triton_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=None,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "no-h0 chunk output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_native, state_triton, "no-h0 chunk state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
def test_chunk_triton_broadcast_initial_state(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """initial_state 形状 [1, H, K, V] 广播时 JAX-Triton 与 native 前向一致。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)
    h0_j = jnp.asarray(h0[:1])

    out_native, state_native = delta_net_chunk_jax_native_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = delta_net_chunk_jax_triton_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "broadcast-h0 chunk output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_native, state_triton, "broadcast-h0 chunk state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_triton_different_chunk_size(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """不同 chunk_size 下 JAX-Triton 与 native 均等价。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)
    h0_j = jnp.asarray(h0)

    for chunk_size in (32, 64):
        out_native, state_native = delta_net_chunk_jax_native_op(
            q_j,
            k_j,
            v_j,
            beta_j,
            initial_state=h0_j,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        out_triton, state_triton = delta_net_chunk_jax_triton_op(
            q_j,
            k_j,
            v_j,
            beta_j,
            initial_state=h0_j,
            output_final_state=True,
            chunk_size=chunk_size,
        )

        assert_allclose_with_stats(
            out_native,
            out_triton,
            f"chunk_size={chunk_size} output",
            atol=1e-2,
            rtol=1e-2,
        )
        assert_allclose_with_stats(
            state_native,
            state_triton,
            f"chunk_size={chunk_size} state",
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_triton_bwd_vs_native(
    delta_net_inputs, delta_net_chunk_jax_native_op, delta_net_chunk_jax_triton_op
):
    """JAX-Triton chunkwise 反向与 native Keras autograd 对拍。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    def _run_and_grad(op):
        q_j = jnp.asarray(q)
        k_j = jnp.asarray(k)
        v_j = jnp.asarray(v)
        beta_j = jnp.asarray(beta)
        h0_j = jnp.asarray(h0)

        def loss_fn(q_, k_, v_, beta_, h0_):
            out, state = op(
                q_,
                k_,
                v_,
                beta_,
                initial_state=h0_,
                output_final_state=True,
                chunk_size=16,
            )
            return jnp.mean(out**2) + jnp.mean(state**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q_j, k_j, v_j, beta_j, h0_j)
        return grads

    grads_native = _run_and_grad(delta_net_chunk_jax_native_op)
    grads_triton = _run_and_grad(delta_net_chunk_jax_triton_op)

    names = ["q", "k", "v", "beta", "h0"]
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.jax
def test_chunk_triton_fwd_vs_native_chunk_size_32(delta_net_inputs):
    """JAX-Triton chunkwise 前向在 chunk_size=32 下与 native 参考实现对拍。"""
    pytest.importorskip("triton")
    from rwkv_ops import get_delta_net_chunk

    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)
    h0_j = jnp.asarray(h0)

    native_op = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_op = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    out_native, state_native = native_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
    )
    out_triton, state_triton = triton_op(
        q_j,
        k_j,
        v_j,
        beta_j,
        initial_state=h0_j,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_native,
        out_triton,
        "chunk_size=32 triton vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_native,
        state_triton,
        "chunk_size=32 triton vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
def test_chunk_triton_no_final_state_chunk_size_32(delta_net_inputs):
    """output_final_state=False 且 chunk_size=32 时不返回 state。"""
    pytest.importorskip("triton")
    from rwkv_ops import get_delta_net_chunk

    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_j = jnp.asarray(q, dtype=jnp.bfloat16)
    k_j = jnp.asarray(k, dtype=jnp.bfloat16)
    v_j = jnp.asarray(v, dtype=jnp.bfloat16)
    beta_j = jnp.asarray(beta)

    native_op = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_op = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    out_native, state_native = native_op(
        q_j, k_j, v_j, beta_j, output_final_state=False
    )
    out_triton, state_triton = triton_op(
        q_j, k_j, v_j, beta_j, output_final_state=False
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native,
        out_triton,
        "chunk_size=32 no-state chunk output",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.jax
@pytest.mark.slow
def test_chunk_triton_bwd_vs_native_chunk_size_32(delta_net_inputs):
    """JAX-Triton chunkwise 反向在 chunk_size=32 下与 native Keras autograd 对拍。"""
    pytest.importorskip("triton")
    from rwkv_ops import get_delta_net_chunk

    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    def _run_and_grad(op):
        q_j = jnp.asarray(q)
        k_j = jnp.asarray(k)
        v_j = jnp.asarray(v)
        beta_j = jnp.asarray(beta)
        h0_j = jnp.asarray(h0)

        def loss_fn(q_, k_, v_, beta_, h0_):
            out, state = op(
                q_,
                k_,
                v_,
                beta_,
                initial_state=h0_,
                output_final_state=True,
            )
            return jnp.mean(out**2) + jnp.mean(state**2)

        grads = jax.grad(loss_fn, argnums=(0, 1, 2, 3, 4))(q_j, k_j, v_j, beta_j, h0_j)
        return grads

    native_op = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_op = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    grads_native = _run_and_grad(native_op)
    grads_triton = _run_and_grad(triton_op)

    names = ["q", "k", "v", "beta", "h0"]
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"chunk_size=32 bwd {name}",
            atol=7e-3,
            rtol=1e-2,
        )

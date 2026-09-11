"""DeltaNet chunkwise Triton 前向数值测试。"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from rwkv_ops import get_delta_net_chunk
from rwkv_ops.delta_net_chunk.native_keras_op import delta_net_chunk as native_chunk
from rwkv_ops.delta_net_chunk.torch_triton_kernel import (
    delta_net_chunk as triton_chunk,
)
from tests.conftest import assert_allclose_with_stats


@pytest.mark.torch
def test_chunk_triton_fwd_vs_native(delta_net_inputs, device):
    """Triton chunkwise 前向与 native 参考实现对拍（bf16 I/O）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    out_native, state_native = native_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "chunk triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_native, state_triton, "chunk triton vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
def test_chunk_triton_no_final_state(delta_net_inputs, device):
    """output_final_state=False 时不返回 state（bf16 I/O）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)

    out_native, state_native = native_chunk(
        q_t, k_t, v_t, beta_t, output_final_state=False, chunk_size=16
    )
    out_triton, state_triton = triton_chunk(
        q_t, k_t, v_t, beta_t, output_final_state=False, chunk_size=16
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native, out_triton, "no-state chunk output", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
def test_chunk_triton_no_initial_state(delta_net_inputs, device):
    """initial_state=None 时 Triton 与 native 前向一致（bf16 I/O）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)

    out_native, state_native = native_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=None,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
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


@pytest.mark.torch
def test_chunk_triton_broadcast_initial_state(delta_net_inputs, device):
    """initial_state 形状 [1, H, K, V] 广播时 Triton 与 native 前向一致。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0[:1]).to(device)

    out_native, state_native = native_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "broadcast-h0 chunk output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_native, state_triton, "broadcast-h0 chunk state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_different_chunk_size(delta_net_inputs, device):
    """不同 chunk_size 下 Triton 与 native 均等价。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    for chunk_size in (32, 64):
        out_native, state_native = native_chunk(
            q_t,
            k_t,
            v_t,
            beta_t,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        out_triton, state_triton = triton_chunk(
            q_t,
            k_t,
            v_t,
            beta_t,
            initial_state=h0_t,
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


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_bwd_vs_native(delta_net_inputs, device):
    """Triton chunkwise 反向与 native Keras autograd 对拍。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    def _run_and_grad(op):
        q_t = torch.from_numpy(q).to(device).to(torch.bfloat16).requires_grad_(True)
        k_t = torch.from_numpy(k).to(device).to(torch.bfloat16).requires_grad_(True)
        v_t = torch.from_numpy(v).to(device).to(torch.bfloat16).requires_grad_(True)
        beta_t = torch.from_numpy(beta).to(device).requires_grad_(True)
        h0_t = torch.from_numpy(h0).to(device).requires_grad_(True)

        out, state = op(
            q_t,
            k_t,
            v_t,
            beta_t,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=16,
        )
        loss = out.pow(2).mean() + state.pow(2).mean()
        loss.backward()
        return (
            q_t.grad,
            k_t.grad,
            v_t.grad,
            beta_t.grad,
            h0_t.grad,
        )

    grads_native = _run_and_grad(native_chunk)
    grads_triton = _run_and_grad(triton_chunk)

    names = ["q", "k", "v", "beta", "h0"]
    # bf16 叶子的梯度为 bf16，按 bf16 红线 1e-2；fp32 叶子梯度按 fp32 红线收紧，
    # dbeta 是 bf16 输入图下游的 fp32 叶子，梯度带 bf16 噪声，放宽到 2e-3
    tols = {
        "q": (1e-2, 1e-2),
        "k": (1e-2, 1e-2),
        "v": (1e-2, 1e-2),
        "beta": (2e-3, 1e-2),
        "h0": (1e-4, 1e-3),
    }
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        atol, rtol = tols[name]
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=atol,
            rtol=rtol,
        )


@pytest.mark.torch
def test_chunk_triton_fwd_chunk_size_32(delta_net_inputs, device):
    """chunk_size=32 时 Triton chunkwise 前向与 native 参考实现对拍（bf16 I/O）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    native_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    out_native, state_native = native_chunk_32(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
    )
    out_triton, state_triton = triton_chunk_32(
        q_t,
        k_t,
        v_t,
        beta_t,
        initial_state=h0_t,
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


@pytest.mark.torch
def test_chunk_triton_no_final_state_chunk_size_32(delta_net_inputs, device):
    """chunk_size=32 且 output_final_state=False 时不返回 state（bf16 I/O）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    beta_t = torch.from_numpy(beta).to(device)

    native_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    out_native, state_native = native_chunk_32(
        q_t,
        k_t,
        v_t,
        beta_t,
        output_final_state=False,
    )
    out_triton, state_triton = triton_chunk_32(
        q_t,
        k_t,
        v_t,
        beta_t,
        output_final_state=False,
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


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_bwd_chunk_size_32(delta_net_inputs, device):
    """chunk_size=32 时 Triton chunkwise 反向与 native Keras autograd 对拍。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    def _run_and_grad(op):
        q_t = torch.from_numpy(q).to(device).to(torch.bfloat16).requires_grad_(True)
        k_t = torch.from_numpy(k).to(device).to(torch.bfloat16).requires_grad_(True)
        v_t = torch.from_numpy(v).to(device).to(torch.bfloat16).requires_grad_(True)
        beta_t = torch.from_numpy(beta).to(device).requires_grad_(True)
        h0_t = torch.from_numpy(h0).to(device).requires_grad_(True)

        out, state = op(
            q_t,
            k_t,
            v_t,
            beta_t,
            initial_state=h0_t,
            output_final_state=True,
        )
        loss = out.pow(2).mean() + state.pow(2).mean()
        loss.backward()
        return (
            q_t.grad,
            k_t.grad,
            v_t.grad,
            beta_t.grad,
            h0_t.grad,
        )

    native_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="native", chunk_size=32)
    triton_chunk_32 = get_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=32)

    grads_native = _run_and_grad(native_chunk_32)
    grads_triton = _run_and_grad(triton_chunk_32)

    names = ["q", "k", "v", "beta", "h0"]
    # bf16 叶子的梯度为 bf16，按 bf16 红线 1e-2；fp32 叶子梯度按 fp32 红线收紧，
    # dbeta 是 bf16 输入图下游的 fp32 叶子，梯度带 bf16 噪声，放宽到 2e-3
    tols = {
        "q": (1e-2, 1e-2),
        "k": (1e-2, 1e-2),
        "v": (1e-2, 1e-2),
        "beta": (2e-3, 1e-2),
        "h0": (1e-4, 1e-3),
    }
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        atol, rtol = tols[name]
        assert_allclose_with_stats(
            ref,
            tgt,
            f"chunk_size=32 bwd {name}",
            atol=atol,
            rtol=rtol,
        )

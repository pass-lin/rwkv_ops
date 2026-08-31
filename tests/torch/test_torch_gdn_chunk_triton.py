"""Gated DeltaNet chunkwise Triton 前向数值测试。"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from rwkv_ops import get_gated_delta_net_chunk
from rwkv_ops.gdn_chunk.native_keras_op import gated_delta_net_chunk as native_chunk
from rwkv_ops.gdn_chunk.torch_triton_kernel import (
    gated_delta_net_chunk as triton_chunk,
)
from tests.conftest import assert_allclose_with_stats


@pytest.mark.torch
def test_chunk_triton_fwd_vs_native(gdn_inputs, device):
    """Triton chunkwise 前向与 native 参考实现对拍（bf16 I/O）。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    out_native, state_native = native_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
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
def test_chunk_triton_no_final_state(gdn_inputs, device):
    """output_final_state=False 时不返回 state（bf16 I/O）。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta = gdn_inputs["g"], gdn_inputs["beta"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)

    out_native, state_native = native_chunk(
        q_t, k_t, v_t, g_t, beta_t, output_final_state=False, chunk_size=16
    )
    out_triton, state_triton = triton_chunk(
        q_t, k_t, v_t, g_t, beta_t, output_final_state=False, chunk_size=16
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native, out_triton, "no-state chunk output", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_different_chunk_size(gdn_inputs, device):
    """不同 chunk_size 下 Triton 与 native 均等价。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    for chunk_size in (32, 64):
        out_native, state_native = native_chunk(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        out_triton, state_triton = triton_chunk(
            q_t,
            k_t,
            v_t,
            g_t,
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
def test_chunk_triton_bwd_vs_native(gdn_inputs, device):
    """Triton chunkwise 反向与 native Keras autograd 对拍。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    def _run_and_grad(op):
        q_t = torch.from_numpy(q).to(device).to(torch.bfloat16).requires_grad_(True)
        k_t = torch.from_numpy(k).to(device).to(torch.bfloat16).requires_grad_(True)
        v_t = torch.from_numpy(v).to(device).to(torch.bfloat16).requires_grad_(True)
        g_t = torch.from_numpy(g).to(device).requires_grad_(True)
        beta_t = torch.from_numpy(beta).to(device).requires_grad_(True)
        h0_t = torch.from_numpy(h0).to(device).requires_grad_(True)

        out, state = op(
            q_t,
            k_t,
            v_t,
            g_t,
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
            g_t.grad,
            beta_t.grad,
            h0_t.grad,
        )

    grads_native = _run_and_grad(native_chunk)
    grads_triton = _run_and_grad(triton_chunk)

    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=1e-1,
            rtol=1e-1,
        )


@pytest.mark.torch
def test_chunk_triton_fwd_chunk_size_8(gdn_inputs, device):
    """chunk_size=8 时 Triton chunkwise 前向与 native 参考实现对拍（bf16 I/O）。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)
    h0_t = torch.from_numpy(h0).to(device)

    native_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="native", chunk_size=8)
    triton_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=8)

    out_native, state_native = native_chunk_8(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
    )
    out_triton, state_triton = triton_chunk_8(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_native,
        out_triton,
        "chunk_size=8 triton vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_native,
        state_triton,
        "chunk_size=8 triton vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.torch
def test_chunk_triton_no_final_state_chunk_size_8(gdn_inputs, device):
    """chunk_size=8 且 output_final_state=False 时不返回 state（bf16 I/O）。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta = gdn_inputs["g"], gdn_inputs["beta"]

    q_t = torch.from_numpy(q).to(device).to(torch.bfloat16)
    k_t = torch.from_numpy(k).to(device).to(torch.bfloat16)
    v_t = torch.from_numpy(v).to(device).to(torch.bfloat16)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)

    native_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="native", chunk_size=8)
    triton_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=8)

    out_native, state_native = native_chunk_8(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        output_final_state=False,
    )
    out_triton, state_triton = triton_chunk_8(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        output_final_state=False,
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native,
        out_triton,
        "chunk_size=8 no-state chunk output",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_bwd_chunk_size_8(gdn_inputs, device):
    """chunk_size=8 时 Triton chunkwise 反向与 native Keras autograd 对拍。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    def _run_and_grad(op):
        q_t = torch.from_numpy(q).to(device).to(torch.bfloat16).requires_grad_(True)
        k_t = torch.from_numpy(k).to(device).to(torch.bfloat16).requires_grad_(True)
        v_t = torch.from_numpy(v).to(device).to(torch.bfloat16).requires_grad_(True)
        g_t = torch.from_numpy(g).to(device).requires_grad_(True)
        beta_t = torch.from_numpy(beta).to(device).requires_grad_(True)
        h0_t = torch.from_numpy(h0).to(device).requires_grad_(True)

        out, state = op(
            q_t,
            k_t,
            v_t,
            g_t,
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
            g_t.grad,
            beta_t.grad,
            h0_t.grad,
        )

    native_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="native", chunk_size=8)
    triton_chunk_8 = get_gated_delta_net_chunk(KERNEL_TYPE="triton", chunk_size=8)

    grads_native = _run_and_grad(native_chunk_8)
    grads_triton = _run_and_grad(triton_chunk_8)

    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"chunk_size=8 bwd {name}",
            atol=1e-1,
            rtol=1e-1,
        )

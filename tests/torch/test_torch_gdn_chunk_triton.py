"""Gated DeltaNet chunkwise Triton 前向数值测试。"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from rwkv_ops.gdn_chunk.native_keras_op import gated_delta_net_chunk as native_chunk
from rwkv_ops.gdn_chunk.torch_triton_kernel import (
    gated_delta_net_chunk as triton_chunk,
)
from tests.conftest import assert_allclose_with_stats


@pytest.mark.torch
def test_chunk_triton_fwd_vs_native(gdn_inputs, device):
    """Triton chunkwise 前向与 native 参考实现对拍。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q_t = torch.from_numpy(q).to(device)
    k_t = torch.from_numpy(k).to(device)
    v_t = torch.from_numpy(v).to(device)
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
        chunk_size=64,
    )
    out_triton, state_triton = triton_chunk(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=64,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "chunk triton vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_native, state_triton, "chunk triton vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_chunk_triton_no_final_state(gdn_inputs, device):
    """output_final_state=False 时不返回 state。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta = gdn_inputs["g"], gdn_inputs["beta"]

    q_t = torch.from_numpy(q).to(device)
    k_t = torch.from_numpy(k).to(device)
    v_t = torch.from_numpy(v).to(device)
    g_t = torch.from_numpy(g).to(device)
    beta_t = torch.from_numpy(beta).to(device)

    out_native, state_native = native_chunk(
        q_t, k_t, v_t, g_t, beta_t, output_final_state=False, chunk_size=64
    )
    out_triton, state_triton = triton_chunk(
        q_t, k_t, v_t, g_t, beta_t, output_final_state=False, chunk_size=64
    )

    assert state_native is None
    assert state_triton is None
    assert_allclose_with_stats(
        out_native, out_triton, "no-state chunk output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_triton_different_chunk_size(gdn_inputs, device):
    """不同 chunk_size 下 Triton 与 native 均等价。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q_t = torch.from_numpy(q).to(device)
    k_t = torch.from_numpy(k).to(device)
    v_t = torch.from_numpy(v).to(device)
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
            atol=1e-4,
            rtol=1e-3,
        )
        assert_allclose_with_stats(
            state_native,
            state_triton,
            f"chunk_size={chunk_size} state",
            atol=1e-4,
            rtol=1e-3,
        )

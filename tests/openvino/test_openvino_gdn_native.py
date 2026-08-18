"""GDN 原生 Keras 实现的数值等价性测试（OpenVINO 后端）。"""

import pytest

pytest.importorskip("openvino")

from rwkv_ops.gdn_chunk.native_keras_op import gated_delta_net_chunk
from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent,
    gated_delta_net_reference,
)
from tests.conftest import assert_allclose_with_stats


@pytest.mark.openvino
def test_gdn_chunk_matches_reference(gdn_inputs):
    """chunkwise 实现应与最简 recurrent reference 数值一致。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    out_chunk, state_chunk = gated_delta_net_chunk(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_ref, state_ref = gated_delta_net_reference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_chunk, "chunk vs reference output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_chunk, "chunk vs reference state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_gdn_recurrent_matches_reference(gdn_inputs):
    """fori_loop recurrent 实现应与最简 reference 一致。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    out_rec, state_rec = gated_delta_net_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gated_delta_net_reference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_rec, "recurrent vs reference output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_rec, "recurrent vs reference state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_gdn_chunk_matches_recurrent(gdn_inputs):
    """chunkwise 与 recurrent 两种实现应等价。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    out_chunk, state_chunk = gated_delta_net_chunk(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_rec, state_rec = gated_delta_net_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_chunk, "chunk vs recurrent output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_chunk, "chunk vs recurrent state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_gdn_no_final_state(gdn_inputs):
    """output_final_state=False 时不应返回 state。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta = gdn_inputs["g"], gdn_inputs["beta"]

    out_chunk, state_chunk = gated_delta_net_chunk(
        q, k, v, g, beta, output_final_state=False, chunk_size=64
    )
    out_rec, state_rec = gated_delta_net_recurrent(
        q, k, v, g, beta, output_final_state=False
    )

    assert state_chunk is None
    assert state_rec is None
    assert_allclose_with_stats(
        out_rec, out_chunk, "no-state chunk vs recurrent output", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
@pytest.mark.slow
def test_gdn_different_chunk_size(gdn_inputs):
    """chunk_size 改变时 chunkwise 结果应保持不变。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    out_c64, state_c64 = gated_delta_net_chunk(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_c32, state_c32 = gated_delta_net_chunk(
        q, k, v, g, beta, initial_state=h0, output_final_state=True, chunk_size=32
    )

    assert_allclose_with_stats(
        out_c64, out_c32, "chunk_size=64 vs 32 output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_c64, state_c32, "chunk_size=64 vs 32 state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_gdn_arbitrary_length(gdn_inputs):
    """recurrent / reference 支持任意长度（非 chunk 整除）。"""
    q, k, v = gdn_inputs["q"], gdn_inputs["k"], gdn_inputs["v"]
    g, beta, h0 = gdn_inputs["g"], gdn_inputs["beta"], gdn_inputs["h0"]

    q, k, v = q[:, :37], k[:, :37], v[:, :37]
    g, beta = g[:, :37], beta[:, :37]

    out_rec, state_rec = gated_delta_net_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gated_delta_net_reference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_rec, "arbitrary length output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_rec, "arbitrary length state", atol=1e-5, rtol=1e-3
    )

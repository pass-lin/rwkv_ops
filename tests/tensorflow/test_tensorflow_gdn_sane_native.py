"""GDN-SANE 原生 Keras 实现的运行测试（TensorFlow 后端）。"""

import warnings

import pytest

pytest.importorskip("tensorflow")

from rwkv_ops.gdn_chunk_sane.native_keras_op import gated_delta_net_chunk_sane
from rwkv_ops.gdn_recurrent_sane.native_keras_op import gated_delta_net_recurrent_sane
from tests.conftest import assert_allclose_with_stats


@pytest.mark.tensorflow
def test_gdn_chunk_sane_runs_with_mask(gdn_sane_inputs):
    """chunkwise SANE native 在带 mask 时能正常跑并返回 out/state。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau, mask = gdn_sane_inputs["tau"], gdn_sane_inputs["mask"]

    out, state = gated_delta_net_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
    )

    assert out.shape == v.shape
    assert state.shape == h0.shape


@pytest.mark.tensorflow
def test_gdn_chunk_sane_no_mask_warning_and_none_state(gdn_sane_inputs):
    """mask=None 时返回 None state 并发出 UserWarning。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau = gdn_sane_inputs["tau"]

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, state = gated_delta_net_chunk_sane(
            q,
            k,
            v,
            g,
            beta,
            tau,
            mask=None,
            initial_state=h0,
            output_final_state=True,
            chunk_size=64,
        )
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert state is None
    assert out.shape == v.shape


@pytest.mark.tensorflow
def test_gdn_chunk_sane_matches_recurrent_sane(gdn_sane_inputs):
    """chunkwise SANE 与 recurrent SANE native 实现等价。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau, mask = gdn_sane_inputs["tau"], gdn_sane_inputs["mask"]

    out_chunk, state_chunk = gated_delta_net_chunk_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
    )
    out_rec, state_rec = gated_delta_net_recurrent_sane(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
    )

    assert_allclose_with_stats(
        out_rec, out_chunk, "chunk vs recurrent output", atol=2e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_chunk, "chunk vs recurrent state", atol=2e-4, rtol=1e-3
    )

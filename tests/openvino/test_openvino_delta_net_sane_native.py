"""DeltaNet-SANE 原生 Keras 实现的数值等价性测试（OpenVINO 后端）。"""

import numpy as np
import pytest

pytest.importorskip("openvino")

from keras import ops

from rwkv_ops.delta_net_chunk.native_keras_op import delta_net_chunk
from rwkv_ops.delta_net_chunk_sane.native_keras_op import delta_net_chunk_sane
from rwkv_ops.delta_net_recurrent.native_keras_op import delta_net_recurrent
from rwkv_ops.delta_net_recurrent_sane.native_keras_op import (
    delta_net_recurrent_sane,
    delta_net_recurrent_sane_inference,
    delta_net_recurrent_sane_single_step,
)
from rwkv_ops.gdn_chunk_sane.native_keras_op import gated_delta_net_chunk_sane
from rwkv_ops.gdn_recurrent_sane.native_keras_op import gated_delta_net_recurrent_sane
from tests.conftest import assert_allclose_with_stats, to_numpy


@pytest.mark.openvino
def test_delta_net_chunk_sane_matches_recurrent_sane(delta_net_sane_inputs):
    """chunkwise SANE 与 recurrent SANE native 实现等价。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_chunk, state_chunk = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_chunk, "chunk sane vs recurrent sane output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec,
        state_chunk,
        "chunk sane vs recurrent sane state",
        atol=1e-2,
        rtol=1e-3,
    )


@pytest.mark.openvino
def test_delta_net_sane_identity_with_large_tau(delta_net_sane_inputs):
    """tau 远大于 state 幅度时 SANE 近似恒等，结果应与无 SANE 算子一致。

    共享 fixture 的 tau（约 8）不足以近似恒等，这里显式构造大 tau。
    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    mask = delta_net_sane_inputs["mask"]
    B, T, H, _ = q.shape
    tau = np.full((B, T // 16, H), 1e4, dtype=np.float32)
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_rec_sane, state_rec_sane = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_rec, out_rec_sane, "recurrent sane identity output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_rec_sane, "recurrent sane identity state", atol=1e-2, rtol=1e-3
    )

    out_chunk_sane, state_chunk_sane = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_chunk, out_chunk_sane, "chunk sane identity output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_chunk, state_chunk_sane, "chunk sane identity state", atol=1e-2, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_sane_mask_all_zero_matches_no_sane(delta_net_sane_inputs):
    """mask 全 0 时 SANE 被冻结，结果应与无 SANE 算子完全一致。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau = delta_net_sane_inputs["tau"]
    B, T, H, _ = q.shape
    mask_zero = np.zeros((B, T // 16), dtype=np.float32)

    out_rec_sane, state_rec_sane = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask_zero, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_rec, out_rec_sane, "recurrent sane mask=0 output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_rec_sane, "recurrent sane mask=0 state", atol=1e-5, rtol=1e-3
    )

    out_chunk_sane, state_chunk_sane = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask_zero, initial_state=h0, output_final_state=True
    )
    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_chunk, out_chunk_sane, "chunk sane mask=0 output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_chunk, state_chunk_sane, "chunk sane mask=0 state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_sane_mask_all_one_recurrent_vs_chunk(delta_net_sane_inputs):
    """mask 全 1 时所有 chunk 边界裁剪，recurrent 与 chunkwise 实现等价。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau = delta_net_sane_inputs["tau"]
    B, T, H, _ = q.shape
    mask_one = np.ones((B, T // 16), dtype=np.float32)

    out_rec, state_rec = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask_one, initial_state=h0, output_final_state=True
    )
    out_chunk, state_chunk = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask_one, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_chunk, "mask=1 chunk vs recurrent output", atol=1e-3, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_chunk, "mask=1 chunk vs recurrent state", atol=1e-3, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_sane_random_mask_recurrent_vs_chunk(delta_net_sane_inputs):
    """随机 mask 下 recurrent 与 chunkwise 实现等价。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]

    out_rec, state_rec = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_chunk, state_chunk = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec,
        out_chunk,
        "random mask chunk vs recurrent output",
        atol=1e-3,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_rec,
        state_chunk,
        "random mask chunk vs recurrent state",
        atol=1e-3,
        rtol=1e-3,
    )


@pytest.mark.openvino
def test_delta_net_recurrent_sane_no_mask_warning_and_none_state(delta_net_sane_inputs):
    """mask=None 且 output_final_state=True 时发出 UserWarning 且 state 为 None。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau = delta_net_sane_inputs["tau"]

    with pytest.warns(UserWarning, match="mask is None"):
        out, state = delta_net_recurrent_sane(
            q, k, v, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )
    assert state is None
    assert out.shape == v.shape


@pytest.mark.openvino
def test_delta_net_chunk_sane_no_mask_warning_and_none_state(delta_net_sane_inputs):
    """mask=None 且 output_final_state=True 时发出 UserWarning 且 state 为 None。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau = delta_net_sane_inputs["tau"]

    with pytest.warns(UserWarning, match="mask is None"):
        out, state = delta_net_chunk_sane(
            q, k, v, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )
    assert state is None
    assert out.shape == v.shape


@pytest.mark.openvino
def test_delta_net_sane_small_tau_clips_state(delta_net_sane_inputs):
    """tau 取小值时软裁剪生效，final_state 各元素绝对值不超过 tau。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    B, T, H, _ = q.shape
    tau_small = np.full((B, T // 16, H), 1.5, dtype=np.float32)
    mask_one = np.ones((B, T // 16), dtype=np.float32)

    _, state_rec = delta_net_recurrent_sane(
        q,
        k,
        v,
        beta,
        tau_small,
        mask=mask_one,
        initial_state=h0,
        output_final_state=True,
    )
    _, state_chunk = delta_net_chunk_sane(
        q,
        k,
        v,
        beta,
        tau_small,
        mask=mask_one,
        initial_state=h0,
        output_final_state=True,
    )
    _, state_no_sane = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    # 无 SANE 时 state 应明显超出 1.5，否则本测试没有裁剪意义。
    assert float(np.max(np.abs(to_numpy(state_no_sane)))) > 1.5
    assert float(np.max(np.abs(to_numpy(state_rec)))) <= 1.5 + 1e-4
    assert float(np.max(np.abs(to_numpy(state_chunk)))) <= 1.5 + 1e-4


@pytest.mark.openvino
def test_delta_net_sane_output_final_state_false(delta_net_sane_inputs):
    """output_final_state=False 时不返回 state，也不发警告。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta = delta_net_sane_inputs["beta"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]

    out_rec, state_rec = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, output_final_state=False
    )
    out_chunk, state_chunk = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask, output_final_state=False
    )
    assert state_rec is None
    assert state_chunk is None
    assert out_rec.shape == v.shape
    assert out_chunk.shape == v.shape


@pytest.mark.openvino
def test_delta_net_recurrent_sane_inference_matches_train(delta_net_sane_inputs):
    """推理封装应与训练实现输出一致。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]

    out_inf, state_inf = delta_net_recurrent_sane_inference(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_inf, "sane inference vs train output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_inf, "sane inference vs train state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_recurrent_sane_single_step_matches_recurrent(delta_net_sane_inputs):
    """单步 RNN 逐步跑一个 chunk，边界步执行 SANE，应与 recurrent 对齐。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau = delta_net_sane_inputs["tau"]
    B = q.shape[0]

    steps = 16
    state = h0
    outs = []
    for t in range(steps):
        do_sane = np.full((B,), (t + 1) % 16 == 0)
        out_t, state = delta_net_recurrent_sane_single_step(
            q[:, t],
            k[:, t],
            v[:, t],
            beta[:, t],
            tau[:, 0],
            do_sane,
            initial_state=state,
            output_final_state=True,
        )
        outs.append(out_t)
    out_step = ops.stack(outs, axis=1)

    out_rec, state_rec = delta_net_recurrent_sane(
        q[:, :steps],
        k[:, :steps],
        v[:, :steps],
        beta[:, :steps],
        tau[:, :1],
        mask=np.ones((B, 1), dtype=np.float32),
        initial_state=h0,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_rec, out_step, "sane single step vs recurrent output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state, "sane single step vs recurrent state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_recurrent_sane_arbitrary_length(delta_net_sane_inputs):
    """recurrent SANE 支持任意长度（T 不被 chunk_size 整除）。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]
    B = q.shape[0]

    q, k, v = q[:, :37], k[:, :37], v[:, :37]
    beta = beta[:, :37]
    tau = tau[:, :2]
    mask = mask[:, :2]

    out_sane, state_sane = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    assert out_sane.shape == v.shape
    assert state_sane.shape == h0.shape

    mask_zero = np.zeros((B, 2), dtype=np.float32)
    out_sane0, state_sane0 = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask_zero, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    assert_allclose_with_stats(
        out_rec, out_sane0, "T=37 sane mask=0 output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_sane0, "T=37 sane mask=0 state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.openvino
def test_delta_net_sane_zero_g_matches_gdn_sane(delta_net_sane_inputs):
    """无 g 的 DeltaNet SANE 应与 decay gate 恒为 0 的 GDN SANE 一致。"""
    q, k, v = (
        delta_net_sane_inputs["q"],
        delta_net_sane_inputs["k"],
        delta_net_sane_inputs["v"],
    )
    beta, h0 = delta_net_sane_inputs["beta"], delta_net_sane_inputs["h0"]
    tau, mask = delta_net_sane_inputs["tau"], delta_net_sane_inputs["mask"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")
    zeros_g = np.zeros(q.shape[:-1], dtype=np.float32)

    out_dn, state_dn = delta_net_recurrent_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_gdn, state_gdn = gated_delta_net_recurrent_sane(
        q,
        k,
        v,
        zeros_g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    assert_allclose_with_stats(
        out_gdn, out_dn, "zero-g gdn sane recurrent output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_gdn, state_dn, "zero-g gdn sane recurrent state", atol=1e-2, rtol=1e-3
    )

    out_dc, state_dc = delta_net_chunk_sane(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_gc, state_gc = gated_delta_net_chunk_sane(
        q,
        k,
        v,
        zeros_g,
        beta,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    assert_allclose_with_stats(
        out_gc, out_dc, "zero-g gdn sane chunk output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_gc, state_dc, "zero-g gdn sane chunk state", atol=1e-2, rtol=1e-3
    )

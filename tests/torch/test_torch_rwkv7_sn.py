"""
RWKV-7 State Norm Torch CUDA kernel 数值测试。

运行方式：
    KERAS_BACKEND=torch pytest tests/torch/test_rwkv7_sn.py -v
"""

import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


def _make_inputs(rwkv7_sn_inputs, device, dtype="bfloat16", grad=False):
    tensors = {
        name: _to_torch(rwkv7_sn_inputs[name], dtype, device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    tensors["tau"] = _to_torch(rwkv7_sn_inputs["tau"], "float32", device)
    tensors["h0"] = _to_torch(rwkv7_sn_inputs["h0"], "float32", device)
    if grad:
        for t in tensors.values():
            t.requires_grad_(True)
    return tensors


def _call_op(op, tensors, output_final_state=True):
    return op(
        r=tensors["r"],
        k=tensors["k"],
        v=tensors["v"],
        a=tensors["a"],
        b=tensors["b"],
        w=tensors["w"],
        tau=tensors["tau"],
        initial_state=tensors["h0"],
        output_final_state=output_final_state,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_forward_state(
    rwkv7_sn_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device
):
    ref = _make_inputs(rwkv7_sn_inputs, device, "float32")
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_sn_native_op, ref, output_final_state=True)
    y_tgt, s_tgt = _call_op(rwkv7_sn_op, tgt, output_final_state=True)

    assert_allclose_with_stats(y_ref, y_tgt, "y", atol=1.0, rtol=1e-1)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state", atol=1.0, rtol=1e-1)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_backward(rwkv7_sn_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device):
    def grads(op, tensors):
        t = {k: v.clone().requires_grad_(True) for k, v in tensors.items()}
        y, s = _call_op(op, t, output_final_state=True)
        loss = (y.float() ** 2).mean() + (s.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in t}

    ref = _make_inputs(rwkv7_sn_inputs, device, "float32")
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    g_ref = grads(rwkv7_sn_native_op, ref)
    g_tgt = grads(rwkv7_sn_op, tgt)

    for name in ["r", "k", "v", "a", "b", "w", "tau", "h0"]:
        assert_allclose_with_stats(
            g_ref[name], g_tgt[name], f"grad_{name}", atol=2e-2, rtol=2e-2
        )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_rnn(
    rwkv7_sn_op,
    rwkv7_sn_rnn_op,
    rwkv7_sn_rnn_native_op,
    rwkv7_sn_inputs,
    device,
):
    """
    验证单步 RNN 与 native 单步在 16 步内一致，并在第 15 步触发 SN。
    """
    inputs = {k: v.copy() for k, v in rwkv7_sn_inputs.items()}
    B, T, H, K = inputs["r"].shape

    # prefill 用训练 kernel（T 必须被 16 整除）
    prefill_len = 16
    pre_inputs = {k: v[:, :prefill_len] for k, v in inputs.items()}
    pre_inputs["tau"] = inputs["tau"][:, : prefill_len // 16]
    t = _make_inputs(pre_inputs, device, "bfloat16")
    _, state = _call_op(rwkv7_sn_op, t, output_final_state=True)

    native_state = state.detach().clone()
    cuda_state = state.detach().clone()

    for step in range(prefill_len):
        rr = inputs["r"][:, prefill_len + step : prefill_len + step + 1]
        kk = inputs["k"][:, prefill_len + step : prefill_len + step + 1]
        vv = inputs["v"][:, prefill_len + step : prefill_len + step + 1]
        aa = inputs["a"][:, prefill_len + step : prefill_len + step + 1]
        bb = inputs["b"][:, prefill_len + step : prefill_len + step + 1]
        ww = inputs["w"][:, prefill_len + step : prefill_len + step + 1]
        tau = inputs["tau"][:, prefill_len // 16]  # 同一 chunk 内 tau 相同
        do_sn = step == 15

        rr_t = _to_torch(rr, "bfloat16", device)
        ww_t = _to_torch(ww, "bfloat16", device)
        kk_t = _to_torch(kk, "bfloat16", device)
        vv_t = _to_torch(vv, "bfloat16", device)
        aa_t = _to_torch(aa, "bfloat16", device)
        bb_t = _to_torch(bb, "bfloat16", device)
        tau_t = _to_torch(tau, "float32", device)

        # CUDA 单步
        cuda_y, cuda_state = rwkv7_sn_rnn_op(
            r=rr_t,
            w=ww_t,
            k=kk_t,
            v=vv_t,
            a=aa_t,
            b=bb_t,
            tau=tau_t,
            do_sn=do_sn,
            initial_state=cuda_state,
            output_final_state=True,
            head_first=False,
        )

        # native 单步
        native_y, native_state = rwkv7_sn_rnn_native_op(
            r=rr_t,
            w=ww_t,
            k=kk_t,
            v=vv_t,
            a=aa_t,
            b=bb_t,
            tau=tau_t,
            do_sn=do_sn,
            initial_state=native_state,
            output_final_state=True,
            head_first=False,
        )

        assert_allclose_with_stats(
            native_y, cuda_y, f"rnn_y_step_{step}", atol=1.0, rtol=1e-1
        )
        assert_allclose_with_stats(
            native_state, cuda_state, f"rnn_state_step_{step}", atol=1.0, rtol=1e-1
        )

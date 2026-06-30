"""
RWKV-7 Torch CUDA 推理专用接口数值测试。
"""

import numpy as np
import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


def _make_inputs(rwkv7_inputs, device, dtype="bfloat16"):
    return {
        name: _to_torch(rwkv7_inputs[name], dtype, device)
        for name in ["r", "k", "v", "a", "b", "w"]
    } | {"h0": _to_torch(rwkv7_inputs["h0"], "float32", device)}


def _call_op(op, tensors, output_final_state=True, mask=None):
    return op(
        r=tensors["r"],
        k=tensors["k"],
        v=tensors["v"],
        a=tensors["a"],
        b=tensors["b"],
        w=tensors["w"],
        initial_state=tensors["h0"],
        output_final_state=output_final_state,
        mask=mask,
    )


@pytest.mark.torch
def test_rwkv7_inference_forward_state(
    rwkv7_inference_op, rwkv7_native_op, rwkv7_inputs, device
):
    ref = _make_inputs(rwkv7_inputs, device, "float32")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_native_op, ref, output_final_state=True)
    y_tgt, s_tgt = _call_op(rwkv7_inference_op, tgt, output_final_state=True)

    assert_allclose_with_stats(y_ref, y_tgt, "y", atol=1.0, rtol=1e-1)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state", atol=1.0, rtol=1e-1)


@pytest.mark.torch
def test_rwkv7_inference_masked(
    rwkv7_inference_op, rwkv7_native_op, rwkv7_inputs, device, rng
):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.ones((B, T), dtype=np.float32)
    freeze = rng.random((B, T)) < 0.3
    mask_np[freeze] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    ref = _make_inputs(rwkv7_inputs, device, "float32")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_native_op, ref, output_final_state=True, mask=mask)
    y_tgt, s_tgt = _call_op(rwkv7_inference_op, tgt, output_final_state=True, mask=mask)

    assert_allclose_with_stats(y_ref, y_tgt, "y_mask", atol=1.0, rtol=1e-1)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state_mask", atol=1.0, rtol=1e-1)


@pytest.mark.torch
def test_rwkv7_inference_mask_all_zero_frozen(rwkv7_inference_op, rwkv7_inputs, device):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = torch.zeros((B, T), dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")
    h0_frozen = tgt["h0"].clone()

    with torch.no_grad():
        _, state_frozen = _call_op(
            rwkv7_inference_op,
            {**tgt, "h0": h0_frozen},
            output_final_state=True,
            mask=mask,
        )

    diff = (state_frozen - h0_frozen).abs().max().item()
    assert diff < 1e-5, f"全 0 Mask 状态被改变 (max_diff={diff:.3e})"


@pytest.mark.torch
def test_rwkv7_inference_mask_all_one_equivalent(
    rwkv7_inference_op, rwkv7_inputs, device
):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = torch.ones((B, T), dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    with torch.no_grad():
        y_no_mask, s_no_mask = _call_op(
            rwkv7_inference_op, tgt, output_final_state=True
        )
        y_all_one, s_all_one = _call_op(
            rwkv7_inference_op, tgt, output_final_state=True, mask=mask
        )

    pred_diff = (y_all_one - y_no_mask).abs().max().item()
    state_diff = (s_all_one - s_no_mask).abs().max().item()
    assert pred_diff < 1e-5, f"全 1 Mask 输出不一致 (max_diff={pred_diff:.3e})"
    assert state_diff < 1e-5, f"全 1 Mask 状态不一致 (max_diff={state_diff:.3e})"


@pytest.mark.torch
def test_rwkv7_inference_last_frame_only(rwkv7_inference_op, rwkv7_inputs, device):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.zeros((B, T), dtype=np.float32)
    mask_np[:, -1] = 1.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    with torch.no_grad():
        _, state_last = _call_op(
            rwkv7_inference_op, tgt, output_final_state=True, mask=mask
        )

    state_change = (state_last - tgt["h0"]).abs().mean().item()
    assert state_change > 0, "最后一帧 Mask 状态未变化"

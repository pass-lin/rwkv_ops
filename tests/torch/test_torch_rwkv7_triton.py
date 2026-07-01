"""
RWKV-7 Torch Triton kernel 数值测试。
"""

import numpy as np
import pytest
import torch

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.fixture(scope="module")
def triton_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op


@pytest.fixture(scope="module")
def native_op():
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


def _make_inputs(rwkv7_inputs, device, dtype="bfloat16", grad=False):
    tensors = {
        name: _to_torch(rwkv7_inputs[name], dtype, device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    tensors["h0"] = _to_torch(rwkv7_inputs["h0"], "float32", device)
    if grad:
        for t in tensors.values():
            t.requires_grad_(True)
    return tensors


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
@pytest.mark.slow
def test_rwkv7_triton_forward_state(triton_op, native_op, rwkv7_inputs, device):
    ref = _make_inputs(rwkv7_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(native_op, ref, output_final_state=True)
    y_tgt, s_tgt = _call_op(triton_op, tgt, output_final_state=True)

    assert_allclose_with_stats(y_ref, y_tgt, "y", atol=1e-5, rtol=1e-2)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state", atol=1e-5, rtol=1e-3)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_triton_backward(triton_op, native_op, rwkv7_inputs, device):
    def grads(op, tensors):
        t = {k: v.clone().requires_grad_(True) for k, v in tensors.items()}
        y, s = _call_op(op, t, output_final_state=True)
        loss = (y.float() ** 2).mean() - (s.float() ** 2).mean()
        loss = loss.abs()
        loss.backward()
        return {k: t[k].grad for k in t}

    ref = _make_inputs(rwkv7_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    g_ref = grads(native_op, ref)
    g_tgt = grads(triton_op, tgt)

    for name in ["r", "k", "v", "a", "b", "w", "h0"]:
        assert_allclose_with_stats(
            g_ref[name], g_tgt[name], f"grad_{name}", atol=7e-3, rtol=7e-3
        )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_triton_forward_state_masked(
    triton_op, native_op, rwkv7_inputs, device, rng
):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.ones((B, T), dtype=np.float32)
    freeze = rng.random((B, T)) < 0.3
    mask_np[freeze] = 0.0
    mask_np[:, -5:] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    ref = _make_inputs(rwkv7_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(native_op, ref, output_final_state=True, mask=mask)
    y_tgt, s_tgt = _call_op(triton_op, tgt, output_final_state=True, mask=mask)

    assert_allclose_with_stats(y_ref, y_tgt, "y_mask", atol=1e-5, rtol=1e-2)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state_mask", atol=1e-5, rtol=1e-3)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_triton_backward_masked(triton_op, native_op, rwkv7_inputs, device, rng):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask_np = np.ones((B, T), dtype=np.float32)
    freeze = rng.random((B, T)) < 0.3
    mask_np[freeze] = 0.0
    mask_np[:, -5:] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    def grads(op, tensors, mask):
        t = {k: v.clone().requires_grad_(True) for k, v in tensors.items()}
        y, s = _call_op(op, t, output_final_state=True, mask=mask)
        loss = (y.float() ** 2).mean() - (s.float() ** 2).mean()
        loss = loss.abs()
        loss.backward()
        return {k: t[k].grad for k in t}

    ref = _make_inputs(rwkv7_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    g_ref = grads(native_op, ref, mask)
    g_tgt = grads(triton_op, tgt, mask)

    for name in ["r", "k", "v", "a", "b", "w", "h0"]:
        assert_allclose_with_stats(
            g_ref[name], g_tgt[name], f"grad_{name}_mask", atol=7e-3, rtol=7e-3
        )


@pytest.mark.torch
def test_rwkv7_triton_mask_all_zero_frozen(triton_op, rwkv7_inputs, device):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = torch.zeros((B, T), dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")
    h0_frozen = tgt["h0"].clone()

    with torch.no_grad():
        _, state_frozen = _call_op(
            triton_op,
            {**tgt, "h0": h0_frozen},
            output_final_state=True,
            mask=mask,
        )

    diff = (state_frozen - h0_frozen).abs().max().item()
    assert diff < 1e-5, f"全 0 Mask 状态被改变 (max_diff={diff:.3e})"


@pytest.mark.torch
def test_rwkv7_triton_mask_all_one_equivalent(triton_op, rwkv7_inputs, device):
    B, T = rwkv7_inputs["r"].shape[:2]
    mask = torch.ones((B, T), dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_inputs, device, "bfloat16")

    with torch.no_grad():
        y_no_mask, s_no_mask = _call_op(triton_op, tgt, output_final_state=True)
        y_all_one, s_all_one = _call_op(
            triton_op, tgt, output_final_state=True, mask=mask
        )

    pred_diff = (y_all_one - y_no_mask).abs().max().item()
    state_diff = (s_all_one - s_no_mask).abs().max().item()
    assert pred_diff < 1e-5, f"全 1 Mask 输出不一致 (max_diff={pred_diff:.3e})"
    assert state_diff < 1e-5, f"全 1 Mask 状态不一致 (max_diff={state_diff:.3e})"

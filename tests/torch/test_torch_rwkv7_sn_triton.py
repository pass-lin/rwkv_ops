"""
RWKV-7 State Neutralization Torch Triton kernel 数值测试。

运行方式：
    KERAS_BACKEND=torch pytest tests/torch/test_torch_rwkv7_sn_triton.py -v
"""

import warnings

import numpy as np
import pytest
import torch

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


def _make_inputs(rwkv7_sn_inputs, device, dtype="bfloat16", grad=False):
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    tensors = {
        name: _to_torch(rwkv7_sn_inputs[name], dtype, device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    tensors["tau"] = _to_torch(rwkv7_sn_inputs["tau"], "float32", device)
    tensors["mask"] = torch.ones(B, T // 16, dtype=torch.float32, device=device)
    tensors["h0"] = _to_torch(rwkv7_sn_inputs["h0"], "float32", device)
    if grad:
        for name in ["r", "k", "v", "a", "b", "w", "tau", "h0"]:
            tensors[name].requires_grad_(True)
    return tensors


_UNSET = object()


def _call_op(op, tensors, output_final_state=True, mask=_UNSET):
    if mask is _UNSET:
        mask = tensors["mask"]
    return op(
        r=tensors["r"],
        k=tensors["k"],
        v=tensors["v"],
        a=tensors["a"],
        b=tensors["b"],
        w=tensors["w"],
        tau=tensors["tau"],
        mask=mask,
        initial_state=tensors["h0"],
        output_final_state=output_final_state,
    )


def _test_is_close(name, ref, tgt, atol, rtol):
    ref_f = ref.detach().float().cpu().numpy()
    tgt_f = tgt.detach().float().cpu().numpy()
    diff = np.abs(ref_f - tgt_f)
    exact_rate = np.sum(diff < 1e-7) / ref_f.size * 100
    avg_err = diff.mean()
    max_diff = diff.max()
    print("-" * 80)
    print(
        f"[{name}] exact={exact_rate:.2f}%, avg_err={avg_err:.6e}, max_diff={max_diff:.6e}"
    )
    assert_allclose_with_stats(ref, tgt, name, atol=atol, rtol=rtol)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_forward_state(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device
):
    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_sn_native_op, ref, output_final_state=True)
    y_tgt, s_tgt = _call_op(rwkv7_sn_triton_op, tgt, output_final_state=True)

    _test_is_close("y", y_ref, y_tgt, atol=1e-4, rtol=1e-2)
    _test_is_close("final_state", s_ref, s_tgt, atol=1e-5, rtol=1e-3)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_backward(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device
):
    def grads(op, tensors):
        t = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in tensors.items()
            if k != "mask"
        }
        t["mask"] = tensors["mask"]
        y, s = _call_op(op, t, output_final_state=True)
        loss = (y.float() ** 2).mean() + (s.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in ["r", "k", "v", "a", "b", "w", "tau", "h0"]}

    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sn_native_op, ref)
    g_tgt = grads(rwkv7_sn_triton_op, tgt)

    thresholds = {
        "r": (1e-4, 1e-2),
        "k": (7e-3, 1e-2),
        "v": (7e-3, 1e-2),
        "a": (7e-3, 1e-2),
        "b": (7e-3, 1e-2),
        "w": (7e-3, 1e-2),
        "tau": (7e-3, 1e-2),
        "h0": (1e-5, 1e-3),
    }
    for name in thresholds:
        atol, rtol = thresholds[name]
        _test_is_close(f"grad_{name}", g_ref[name], g_tgt[name], atol, rtol)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_forward_state_masked(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device, rng
):
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    n_chunks = T // 16
    mask_np = np.ones((B, n_chunks), dtype=np.float32)
    freeze = rng.random((B, n_chunks)) < 0.3
    mask_np[freeze] = 0.0
    mask_np[:, -1] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_sn_native_op, ref, output_final_state=True, mask=mask)
    y_tgt, s_tgt = _call_op(rwkv7_sn_triton_op, tgt, output_final_state=True, mask=mask)

    _test_is_close("y_mask", y_ref, y_tgt, atol=1e-4, rtol=1e-2)
    _test_is_close("final_state_mask", s_ref, s_tgt, atol=1e-5, rtol=1e-3)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_backward_masked(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device, rng
):
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    n_chunks = T // 16
    mask_np = np.ones((B, n_chunks), dtype=np.float32)
    freeze = rng.random((B, n_chunks)) < 0.3
    mask_np[freeze] = 0.0
    mask_np[:, -1] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    def grads(op, tensors, mask):
        t = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in tensors.items()
            if k != "mask"
        }
        t["mask"] = tensors["mask"]
        y, s = _call_op(op, t, output_final_state=True, mask=mask)
        loss = (y.float() ** 2).mean() + (s.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in ["r", "k", "v", "a", "b", "w", "tau", "h0"]}

    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sn_native_op, ref, mask)
    g_tgt = grads(rwkv7_sn_triton_op, tgt, mask)

    thresholds = {
        "r": (1e-4, 1e-2),
        "k": (7e-3, 1e-2),
        "v": (7e-3, 1e-2),
        "a": (7e-3, 1e-2),
        "b": (1e-2, 1e-2),
        "w": (7e-3, 1e-2),
        "tau": (7e-3, 1e-2),
        "h0": (1e-5, 1e-3),
    }
    for name in thresholds:
        atol, rtol = thresholds[name]
        _test_is_close(f"grad_{name}_mask", g_ref[name], g_tgt[name], atol, rtol)


@pytest.mark.torch
def test_rwkv7_sn_triton_no_mask_y_matches_all_one(
    rwkv7_sn_triton_op, rwkv7_sn_inputs, device
):
    B, T, H, K = rwkv7_sn_inputs["r"].shape
    n_chunks = T // 16
    mask = torch.ones(B, n_chunks, dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            y_no_mask, s_no_mask = _call_op(
                rwkv7_sn_triton_op, tgt, output_final_state=True, mask=None
            )
        assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)

    with torch.no_grad():
        y_all_one, s_all_one = _call_op(
            rwkv7_sn_triton_op, tgt, output_final_state=True, mask=mask
        )

    assert s_no_mask is None
    pred_diff = (y_all_one - y_no_mask).abs().max().item()
    assert pred_diff < 1e-5, (
        f"无 mask 与全 1 mask 输出不一致 (max_diff={pred_diff:.3e})"
    )
    assert s_all_one is not None


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_no_mask_forward_state(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device
):
    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16")

    with torch.no_grad():
        y_ref = _call_op(rwkv7_sn_native_op, ref, output_final_state=False)
        y_tgt = _call_op(rwkv7_sn_triton_op, tgt, output_final_state=False)

    _test_is_close("y_no_mask", y_ref, y_tgt, atol=1e-4, rtol=1e-2)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sn_triton_no_mask_backward(
    rwkv7_sn_triton_op, rwkv7_sn_native_op, rwkv7_sn_inputs, device
):
    def grads(op, tensors):
        t = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in tensors.items()
            if k != "mask"
        }
        t["mask"] = tensors["mask"]
        y = _call_op(op, t, output_final_state=False)
        loss = (y.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in ["r", "k", "v", "a", "b", "w", "tau", "h0"]}

    ref = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sn_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sn_native_op, ref)
    g_tgt = grads(rwkv7_sn_triton_op, tgt)

    thresholds = {
        "r": (1e-4, 1e-2),
        "k": (7e-3, 1e-2),
        "v": (7e-3, 1e-2),
        "a": (7e-3, 1e-2),
        "b": (7e-3, 1e-2),
        "w": (7e-3, 1e-2),
        "tau": (7e-3, 1e-2),
        "h0": (1e-5, 1e-3),
    }
    for name in thresholds:
        atol, rtol = thresholds[name]
        _test_is_close(f"grad_no_mask_{name}", g_ref[name], g_tgt[name], atol, rtol)

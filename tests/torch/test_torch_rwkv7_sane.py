"""
RWKV-7 State Anomaly Neutralization Torch CUDA kernel 数值测试。

运行方式：
    KERAS_BACKEND=torch pytest tests/torch/test_rwkv7_sane.py -v
"""

import warnings

import numpy as np
import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


def _generate_chunk_tau_mask(rng, B, T, H, chunk_size, masked=False):
    """生成指定 chunk_size 的 tau 与 mask。"""
    n_chunks = T // chunk_size
    x = rng.standard_normal((B, n_chunks, H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    if masked:
        mask = rng.integers(0, 2, (B, n_chunks)).astype(np.float32)
        mask[:, -1] = 0.0
    else:
        mask = np.ones((B, n_chunks), dtype=np.float32)
    return tau.astype(np.float32), mask


def _make_inputs(
    rwkv7_sane_inputs, device, dtype="bfloat16", grad=False, chunk_size=16, rng=None
):
    """构造指定 chunk_size 的 Torch 输入张量。"""
    tensors = {
        name: _to_torch(rwkv7_sane_inputs[name], dtype, device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    B, T, H, _ = rwkv7_sane_inputs["r"].shape
    n_chunks = T // chunk_size
    if rng is None and chunk_size == 16:
        tensors["tau"] = _to_torch(rwkv7_sane_inputs["tau"], "float32", device)
    else:
        if rng is None:
            raise ValueError("chunk_size != 16 时必须提供 rng")
        tau_np, _ = _generate_chunk_tau_mask(rng, B, T, H, chunk_size, masked=False)
        tensors["tau"] = _to_torch(tau_np, "float32", device)
    tensors["mask"] = torch.ones(B, n_chunks, dtype=torch.float32, device=device)
    tensors["h0"] = _to_torch(rwkv7_sane_inputs["h0"], "float32", device)
    if grad:
        for name in ["r", "k", "v", "a", "b", "w", "tau", "h0"]:
            tensors[name].requires_grad_(True)
    return tensors


_UNSET = object()


def _call_op(op, tensors, output_final_state=True, mask=_UNSET, chunk_size=None):
    if mask is _UNSET:
        mask = tensors["mask"]
    kwargs = dict(
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
    if chunk_size is not None:
        kwargs["chunk_size"] = chunk_size
    return op(**kwargs)


def _test_is_close(name, ref, tgt, atol, rtol, min_exact_rate=None):
    ref_f = ref.detach().float().cpu().numpy()
    tgt_f = tgt.detach().float().cpu().numpy()
    diff = np.abs(ref_f - tgt_f)
    total = ref_f.size
    exact_rate = np.sum(diff < 1e-7) / total * 100
    avg_err = diff.mean()
    max_diff = diff.max()
    print("-" * 80)
    print(
        f"[{name}] exact={exact_rate:.2f}%, avg_err={avg_err:.6e}, max_diff={max_diff:.6e}"
    )
    # exact match rate 仅作为诊断信息打印，不作为通过/失败条件。
    # 数值正确性由下面的 atol/rtol 保证。
    assert_allclose_with_stats(ref, tgt, name, atol=atol, rtol=rtol)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_forward_state(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device
):
    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(rwkv7_sane_native_op, ref, output_final_state=True)
    y_tgt, s_tgt = _call_op(rwkv7_sane_op, tgt, output_final_state=True)

    _test_is_close("y", y_ref, y_tgt, atol=1e-4, rtol=1e-2, min_exact_rate=99.0)
    _test_is_close(
        "final_state", s_ref, s_tgt, atol=1e-5, rtol=1e-3, min_exact_rate=55.0
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_backward(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device
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

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sane_native_op, ref)
    g_tgt = grads(rwkv7_sane_op, tgt)

    thresholds = {
        "r": (1e-4, 1e-2, 98.0),
        "k": (7e-3, 1e-2, 60.0),
        "v": (7e-3, 1e-2, 35.0),
        "a": (7e-3, 1e-2, 35.0),
        "b": (7e-3, 1e-2, 50.0),
        "w": (7e-3, 1e-2, 50.0),
        "tau": (7e-3, 1e-2, 0.0),
        "h0": (1e-5, 1e-3, 85.0),
    }
    for name in thresholds:
        atol, rtol, exact = thresholds[name]
        _test_is_close(f"grad_{name}", g_ref[name], g_tgt[name], atol, rtol, exact)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_forward_state_masked(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    B, T, H, K = rwkv7_sane_inputs["r"].shape
    n_chunks = T // 16
    mask_np = np.ones((B, n_chunks), dtype=np.float32)
    freeze = rng.random((B, n_chunks)) < 0.3
    mask_np[freeze] = 0.0
    mask_np[:, -1] = 0.0
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")

    y_ref, s_ref = _call_op(
        rwkv7_sane_native_op, ref, output_final_state=True, mask=mask
    )
    y_tgt, s_tgt = _call_op(rwkv7_sane_op, tgt, output_final_state=True, mask=mask)

    _test_is_close("y_mask", y_ref, y_tgt, atol=1e-4, rtol=1e-2, min_exact_rate=99.0)
    _test_is_close(
        "final_state_mask", s_ref, s_tgt, atol=1e-5, rtol=1e-3, min_exact_rate=55.0
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_backward_masked(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    B, T, H, K = rwkv7_sane_inputs["r"].shape
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

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sane_native_op, ref, mask)
    g_tgt = grads(rwkv7_sane_op, tgt, mask)

    thresholds = {
        "r": (1e-4, 1e-2, 98.0),
        "k": (7e-3, 1e-2, 60.0),
        "v": (7e-3, 1e-2, 35.0),
        "a": (7e-3, 1e-2, 35.0),
        "b": (1e-2, 1e-2, 50.0),
        "w": (7e-3, 1e-2, 50.0),
        "tau": (7e-3, 1e-2, 0.0),
        "h0": (1e-5, 1e-3, 85.0),
    }
    for name in thresholds:
        atol, rtol, exact = thresholds[name]
        _test_is_close(f"grad_{name}_mask", g_ref[name], g_tgt[name], atol, rtol, exact)


@pytest.mark.torch
def test_rwkv7_sane_no_mask_y_matches_all_one(rwkv7_sane_op, rwkv7_sane_inputs, device):
    """无 mask 算子与全 1 mask 算子的 y 应一致；无 mask 路径返回 None state。"""
    B, T, H, K = rwkv7_sane_inputs["r"].shape
    n_chunks = T // 16
    mask = torch.ones(B, n_chunks, dtype=torch.float32, device=device)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            y_no_mask, s_no_mask = _call_op(
                rwkv7_sane_op, tgt, output_final_state=True, mask=None
            )
        assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)

    with torch.no_grad():
        y_all_one, s_all_one = _call_op(
            rwkv7_sane_op, tgt, output_final_state=True, mask=mask
        )

    assert s_no_mask is None
    pred_diff = (y_all_one - y_no_mask).abs().max().item()
    assert pred_diff < 1e-5, (
        f"无 mask 与全 1 mask 输出不一致 (max_diff={pred_diff:.3e})"
    )
    assert s_all_one is not None


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_no_mask_forward_state(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device
):
    """output_final_state=False 时走无 mask 算子，y 与 native 一致且不返回 state。"""
    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16")

    with torch.no_grad():
        y_ref = _call_op(rwkv7_sane_native_op, ref, output_final_state=False)
        y_tgt = _call_op(rwkv7_sane_op, tgt, output_final_state=False)

    _test_is_close("y_no_mask", y_ref, y_tgt, atol=1e-4, rtol=1e-2, min_exact_rate=99.0)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_no_mask_backward(
    rwkv7_sane_op, rwkv7_sane_native_op, rwkv7_sane_inputs, device
):
    """无 mask 路径反向梯度与 native 对比（output_final_state=False，仅对 y 求导）。"""

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

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", grad=True)

    g_ref = grads(rwkv7_sane_native_op, ref)
    g_tgt = grads(rwkv7_sane_op, tgt)

    thresholds = {
        "r": (1e-4, 1e-2, 98.0),
        "k": (7e-3, 1e-2, 60.0),
        "v": (7e-3, 1e-2, 35.0),
        "a": (7e-3, 1e-2, 35.0),
        "b": (7e-3, 1e-2, 50.0),
        "w": (7e-3, 1e-2, 50.0),
        "tau": (7e-3, 1e-2, 0.0),
        "h0": (1e-5, 1e-3, 85.0),
    }
    for name in thresholds:
        atol, rtol, exact = thresholds[name]
        _test_is_close(
            f"grad_no_mask_{name}", g_ref[name], g_tgt[name], atol, rtol, exact
        )


@pytest.mark.torch
def test_rwkv7_sane_inference_arbitrary_length(
    rwkv7_sane_op, rwkv7_sane_inference_op, rwkv7_sane_inputs, device
):
    """推理入口支持 T 不被 16 整除，此时 tau 长度只需等于 T // 16。"""
    B, T, H, K = rwkv7_sane_inputs["r"].shape
    actual_len = 34

    tensors = {
        name: _to_torch(rwkv7_sane_inputs[name][:, :actual_len], "bfloat16", device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    tensors["tau"] = _to_torch(
        rwkv7_sane_inputs["tau"][:, : actual_len // 16], "float32", device
    )
    tensors["h0"] = _to_torch(rwkv7_sane_inputs["h0"], "float32", device)

    with torch.no_grad():
        y = rwkv7_sane_inference_op(
            r=tensors["r"],
            w=tensors["w"],
            k=tensors["k"],
            v=tensors["v"],
            a=tensors["a"],
            b=tensors["b"],
            tau=tensors["tau"],
            initial_state=tensors["h0"],
            output_final_state=False,
            head_first=False,
        )
    assert y.shape == (B, actual_len, H, K)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            y2, s = rwkv7_sane_inference_op(
                r=tensors["r"],
                w=tensors["w"],
                k=tensors["k"],
                v=tensors["v"],
                a=tensors["a"],
                b=tensors["b"],
                tau=tensors["tau"],
                initial_state=tensors["h0"],
                output_final_state=True,
                head_first=False,
            )
    assert y2.shape == (B, actual_len, H, K)
    assert s is None
    assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)


@pytest.mark.torch
def test_rwkv7_sane_irregular_padding(
    rwkv7_sane_op,
    rwkv7_sane_native_op,
    rwkv7_sane_rnn_native_op,
    rwkv7_sane_inputs,
    device,
):
    """
    验证不规则长度 padding 场景：实际长度 34，pad 到 48，
    padding chunk mask=0，且 padding 位置 k=v=a=b=0, w=-inf，
    最终 state 应与逐 native 单步跑完 34 个 token 一致。
    """
    B, T, H, K = rwkv7_sane_inputs["r"].shape
    actual_len = 34
    pad_len = ((actual_len + 15) // 16) * 16  # 48
    assert pad_len <= T

    def _pad(name, val, pad_val):
        full = rwkv7_sane_inputs[name][:, :pad_len].copy()
        full[:, actual_len:] = pad_val
        return full

    r = _pad("r", rwkv7_sane_inputs["r"][:, :pad_len], 0.0)
    k = _pad("k", rwkv7_sane_inputs["k"][:, :pad_len], 0.0)
    v = _pad("v", rwkv7_sane_inputs["v"][:, :pad_len], 0.0)
    a = _pad("a", rwkv7_sane_inputs["a"][:, :pad_len], 0.0)
    b = _pad("b", rwkv7_sane_inputs["b"][:, :pad_len], 0.0)
    # w = -inf 对应 decay=1，状态不更新
    w = _pad("w", rwkv7_sane_inputs["w"][:, :pad_len], -1e9)

    tau_full = rwkv7_sane_inputs["tau"][:, : pad_len // 16].copy()
    mask_np = np.ones((B, pad_len // 16), dtype=np.float32)
    mask_np[:, actual_len // 16 :] = 0.0

    h0 = rwkv7_sane_inputs["h0"]

    tensors = {
        "r": _to_torch(r, "bfloat16", device),
        "k": _to_torch(k, "bfloat16", device),
        "v": _to_torch(v, "bfloat16", device),
        "a": _to_torch(a, "bfloat16", device),
        "b": _to_torch(b, "bfloat16", device),
        "w": _to_torch(w, "bfloat16", device),
        "tau": _to_torch(tau_full, "float32", device),
        "mask": _to_torch(mask_np, "float32", device),
        "h0": _to_torch(h0, "float32", device),
    }

    with torch.no_grad():
        _, state_cuda = _call_op(rwkv7_sane_op, tensors, output_final_state=True)

    # 参考：先跑完前 32 个 token 的训练版本（mask [1,1]）
    pre_tensors = {
        "r": _to_torch(r[:, :32], "bfloat16", device),
        "k": _to_torch(k[:, :32], "bfloat16", device),
        "v": _to_torch(v[:, :32], "bfloat16", device),
        "a": _to_torch(a[:, :32], "bfloat16", device),
        "b": _to_torch(b[:, :32], "bfloat16", device),
        "w": _to_torch(w[:, :32], "bfloat16", device),
        "tau": _to_torch(tau_full[:, :2], "float32", device),
        "mask": torch.ones(B, 2, dtype=torch.float32, device=device),
        "h0": _to_torch(h0, "float32", device),
    }
    with torch.no_grad():
        _, state_ref = _call_op(
            rwkv7_sane_native_op, pre_tensors, output_final_state=True
        )

    # 再用 native 单步跑 token 32,33（不触发 SANE）
    state_ref = state_ref.to(device)
    for step in range(32, actual_len):
        rr = _to_torch(rwkv7_sane_inputs["r"][:, step : step + 1], "bfloat16", device)
        kk = _to_torch(rwkv7_sane_inputs["k"][:, step : step + 1], "bfloat16", device)
        vv = _to_torch(rwkv7_sane_inputs["v"][:, step : step + 1], "bfloat16", device)
        aa = _to_torch(rwkv7_sane_inputs["a"][:, step : step + 1], "bfloat16", device)
        bb = _to_torch(rwkv7_sane_inputs["b"][:, step : step + 1], "bfloat16", device)
        ww = _to_torch(rwkv7_sane_inputs["w"][:, step : step + 1], "bfloat16", device)
        tau_s = _to_torch(rwkv7_sane_inputs["tau"][:, 2], "float32", device)
        _, state_ref = rwkv7_sane_rnn_native_op(
            r=rr,
            w=ww,
            k=kk,
            v=vv,
            a=aa,
            b=bb,
            tau=tau_s,
            do_sane=False,
            initial_state=state_ref,
            output_final_state=True,
            head_first=False,
        )

    _test_is_close(
        "irregular_padding_state",
        state_ref,
        state_cuda,
        atol=1e-4,
        rtol=1e-3,
        min_exact_rate=5.0,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_forward_state(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE CUDA 训练算子在 chunk_size=8 时与 native 前向对齐。"""
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_sane_inputs["r"].shape
    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)

    y_ref, s_ref = _call_op(
        rwkv7_sane_native_op, ref, output_final_state=True, chunk_size=8
    )
    y_tgt, s_tgt = _call_op(op, tgt, output_final_state=True, chunk_size=8)

    _test_is_close("y_chunk8", y_ref, y_tgt, atol=1e-4, rtol=1e-2, min_exact_rate=99.0)
    _test_is_close(
        "final_state_chunk8",
        s_ref,
        s_tgt,
        atol=1e-5,
        rtol=1e-3,
        min_exact_rate=55.0,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_backward(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE CUDA 训练算子 chunk_size=8 反向梯度与 native 对齐。"""

    def grads(operator, tensors):
        t = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in tensors.items()
            if k != "mask"
        }
        t["mask"] = tensors["mask"]
        y, s = _call_op(operator, t, output_final_state=True, chunk_size=8)
        loss = (y.float() ** 2).mean() + (s.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in ["r", "k", "v", "a", "b", "w", "tau", "h0"]}

    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_sane_inputs["r"].shape
    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    ref = _make_inputs(
        rwkv7_sane_inputs, device, "bfloat16", grad=True, chunk_size=8, rng=rng
    )
    tgt = _make_inputs(
        rwkv7_sane_inputs, device, "bfloat16", grad=True, chunk_size=8, rng=rng
    )

    g_ref = grads(rwkv7_sane_native_op, ref)
    g_tgt = grads(op, tgt)

    thresholds = {
        "r": (1e-4, 1e-2, 98.0),
        "k": (7e-3, 1e-2, 60.0),
        "v": (7e-3, 1e-2, 35.0),
        "a": (7e-3, 1e-2, 35.0),
        "b": (7e-3, 1e-2, 50.0),
        "w": (7e-3, 1e-2, 50.0),
        "tau": (7e-3, 1e-2, 0.0),
        "h0": (1e-5, 1e-3, 85.0),
    }
    for name in thresholds:
        atol, rtol, exact = thresholds[name]
        _test_is_close(
            f"grad_chunk8_{name}", g_ref[name], g_tgt[name], atol, rtol, exact
        )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_forward_state_masked(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE CUDA 训练算子 chunk_size=8 随机 mask 前向与 native 对齐。"""
    from rwkv_ops import get_generalized_delta_rule_sane

    B, T, H, K = rwkv7_sane_inputs["r"].shape
    _, mask_np = _generate_chunk_tau_mask(rng, B, T, H, 8, masked=True)
    mask = _to_torch(mask_np, "float32", device)

    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)

    y_ref, s_ref = _call_op(
        rwkv7_sane_native_op, ref, output_final_state=True, mask=mask, chunk_size=8
    )
    y_tgt, s_tgt = _call_op(op, tgt, output_final_state=True, mask=mask, chunk_size=8)

    _test_is_close(
        "y_chunk8_mask", y_ref, y_tgt, atol=1e-4, rtol=1e-2, min_exact_rate=99.0
    )
    _test_is_close(
        "final_state_chunk8_mask",
        s_ref,
        s_tgt,
        atol=1e-5,
        rtol=1e-3,
        min_exact_rate=55.0,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_backward_masked(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE CUDA 训练算子 chunk_size=8 随机 mask 反向梯度与 native 对齐。"""

    def grads(operator, tensors, mask):
        t = {
            k: v.clone().detach().requires_grad_(True)
            for k, v in tensors.items()
            if k != "mask"
        }
        t["mask"] = tensors["mask"]
        y, s = _call_op(operator, t, output_final_state=True, mask=mask, chunk_size=8)
        loss = (y.float() ** 2).mean() + (s.float() ** 2).mean()
        loss.backward()
        return {k: t[k].grad for k in ["r", "k", "v", "a", "b", "w", "tau", "h0"]}

    from rwkv_ops import get_generalized_delta_rule_sane

    B, T, H, K = rwkv7_sane_inputs["r"].shape
    _, mask_np = _generate_chunk_tau_mask(rng, B, T, H, 8, masked=True)
    mask = _to_torch(mask_np, "float32", device)

    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    ref = _make_inputs(
        rwkv7_sane_inputs, device, "bfloat16", grad=True, chunk_size=8, rng=rng
    )
    tgt = _make_inputs(
        rwkv7_sane_inputs, device, "bfloat16", grad=True, chunk_size=8, rng=rng
    )

    g_ref = grads(rwkv7_sane_native_op, ref, mask)
    g_tgt = grads(op, tgt, mask)

    thresholds = {
        "r": (1e-4, 1e-2, 98.0),
        "k": (7e-3, 1e-2, 60.0),
        "v": (7e-3, 1e-2, 35.0),
        "a": (7e-3, 1e-2, 35.0),
        "b": (1e-2, 1e-2, 50.0),
        "w": (7e-3, 1e-2, 50.0),
        "tau": (7e-3, 1e-2, 0.0),
        "h0": (1e-5, 1e-3, 85.0),
    }
    for name in thresholds:
        atol, rtol, exact = thresholds[name]
        _test_is_close(
            f"grad_chunk8_mask_{name}",
            g_ref[name],
            g_tgt[name],
            atol,
            rtol,
            exact,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_triton_forward_state(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE Triton 训练算子 chunk_size=8 前向与 native 对齐。"""
    pytest.importorskip("triton")
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_sane_inputs["r"].shape
    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="triton", chunk_size=8
    )

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)

    y_ref, s_ref = _call_op(
        rwkv7_sane_native_op, ref, output_final_state=True, chunk_size=8
    )
    y_tgt, s_tgt = _call_op(op, tgt, output_final_state=True, chunk_size=8)

    _test_is_close(
        "y_chunk8_triton",
        y_ref,
        y_tgt,
        atol=1e-4,
        rtol=1e-2,
        min_exact_rate=99.0,
    )
    _test_is_close(
        "final_state_chunk8_triton",
        s_ref,
        s_tgt,
        atol=1e-5,
        rtol=1e-3,
        min_exact_rate=55.0,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_inference(rwkv7_sane_inputs, device, rng):
    """RWKV-7-SANE CUDA 推理算子 chunk_size=8 支持任意长度（T 不被 8 整除）。"""
    from rwkv_ops import get_generalized_delta_rule_sane

    B, _, H, K = rwkv7_sane_inputs["r"].shape
    actual_len = 34
    _, op = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    tensors = {
        name: _to_torch(rwkv7_sane_inputs[name][:, :actual_len], "bfloat16", device)
        for name in ["r", "k", "v", "a", "b", "w"]
    }
    tau_np, _ = _generate_chunk_tau_mask(rng, B, actual_len, H, 8, masked=False)
    tensors["tau"] = _to_torch(tau_np, "float32", device)
    tensors["h0"] = _to_torch(rwkv7_sane_inputs["h0"], "float32", device)

    with torch.no_grad():
        y = op(
            r=tensors["r"],
            w=tensors["w"],
            k=tensors["k"],
            v=tensors["v"],
            a=tensors["a"],
            b=tensors["b"],
            tau=tensors["tau"],
            initial_state=tensors["h0"],
            output_final_state=False,
            head_first=False,
            chunk_size=8,
        )
    assert y.shape == (B, actual_len, H, K)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        with torch.no_grad():
            y2, s = op(
                r=tensors["r"],
                w=tensors["w"],
                k=tensors["k"],
                v=tensors["v"],
                a=tensors["a"],
                b=tensors["b"],
                tau=tensors["tau"],
                initial_state=tensors["h0"],
                output_final_state=True,
                head_first=False,
                chunk_size=8,
            )
    assert y2.shape == (B, actual_len, H, K)
    assert s is None
    assert len(rec) == 1 and issubclass(rec[-1].category, UserWarning)


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv7_sane_chunk8_no_mask_forward(
    rwkv7_sane_native_op, rwkv7_sane_inputs, device, rng
):
    """RWKV-7-SANE CUDA 训练算子 chunk_size=8 无 mask 路径前向与 native 对齐。"""
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_sane_inputs["r"].shape
    op, _ = get_generalized_delta_rule_sane(
        HEAD_SIZE=K, KERNEL_TYPE="cuda", chunk_size=8
    )

    ref = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)
    tgt = _make_inputs(rwkv7_sane_inputs, device, "bfloat16", chunk_size=8, rng=rng)

    with torch.no_grad():
        y_ref = _call_op(
            rwkv7_sane_native_op,
            ref,
            output_final_state=False,
            mask=None,
            chunk_size=8,
        )
        y_tgt = _call_op(op, tgt, output_final_state=False, mask=None, chunk_size=8)

    _test_is_close(
        "y_chunk8_no_mask",
        y_ref,
        y_tgt,
        atol=1e-4,
        rtol=1e-2,
        min_exact_rate=99.0,
    )

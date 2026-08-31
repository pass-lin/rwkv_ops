"""Gated DeltaNet chunkwise SANE Triton 数值测试。"""

import warnings

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")

import torch

from rwkv_ops import get_gated_delta_net_chunk_sane
from rwkv_ops.gdn_chunk.native_keras_op import gated_delta_net_chunk as gdn_native_chunk
from rwkv_ops.gdn_chunk_sane.native_keras_op import (
    gated_delta_net_chunk_sane as native_chunk_sane,
)
from rwkv_ops.gdn_chunk_sane.torch_triton_kernel import (
    gated_delta_net_chunk_sane as triton_chunk_sane,
)
from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, device, dtype=torch.bfloat16):
    """numpy -> torch tensor，放在指定 device 上。"""
    t = torch.from_numpy(arr).to(device)
    if dtype is not None:
        t = t.to(dtype)
    return t


@pytest.mark.torch
def test_chunk_sane_triton_fwd_vs_native(gdn_sane_inputs, device):
    """Triton SANE chunkwise 前向/最终 state 与 native 参考对拍（bf16 I/O）。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau, mask = gdn_sane_inputs["tau"], gdn_sane_inputs["mask"]

    q_t = _to_torch(q, device)
    k_t = _to_torch(k, device)
    v_t = _to_torch(v, device)
    g_t = _to_torch(g, device, dtype=torch.float32)
    beta_t = _to_torch(beta, device, dtype=torch.float32)
    tau_t = _to_torch(tau, device, dtype=torch.float32)
    mask_t = _to_torch(mask, device, dtype=torch.float32)
    h0_t = _to_torch(h0, device, dtype=torch.float32)

    out_ref, state_ref = native_chunk_sane(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=mask_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )
    out_triton, state_triton = triton_chunk_sane(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=mask_t,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )

    assert_allclose_with_stats(
        out_ref, out_triton, "chunk sane triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref,
        state_triton,
        "chunk sane triton vs native state",
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_sane_triton_bwd_vs_native(gdn_sane_inputs, device):
    """Triton SANE chunkwise 反向（含 dtau）与 native Keras autograd 对拍。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau, mask = gdn_sane_inputs["tau"], gdn_sane_inputs["mask"]

    def _run_and_grad(op):
        q_t = _to_torch(q, device).requires_grad_(True)
        k_t = _to_torch(k, device).requires_grad_(True)
        v_t = _to_torch(v, device).requires_grad_(True)
        g_t = _to_torch(g, device, dtype=torch.float32).requires_grad_(True)
        beta_t = _to_torch(beta, device, dtype=torch.float32).requires_grad_(True)
        tau_t = _to_torch(tau, device, dtype=torch.float32).requires_grad_(True)
        mask_t = _to_torch(mask, device, dtype=torch.float32)
        h0_t = _to_torch(h0, device, dtype=torch.float32).requires_grad_(True)

        out, state = op(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            tau_t,
            mask=mask_t,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=16,
        )
        loss = out.to(torch.float32).pow(2).mean() + state.pow(2).mean()
        loss.backward()
        return (
            q_t.grad,
            k_t.grad,
            v_t.grad,
            g_t.grad,
            beta_t.grad,
            tau_t.grad,
            h0_t.grad,
        )

    grads_native = _run_and_grad(native_chunk_sane)
    grads_triton = _run_and_grad(triton_chunk_sane)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_native, grads_triton):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=2e-1,
            rtol=2e-1,
        )


@pytest.mark.torch
def test_chunk_sane_triton_no_mask_warning_and_none_state(gdn_sane_inputs, device):
    """mask=None 且 output_final_state=True 时发出 UserWarning 并返回 None state。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau = gdn_sane_inputs["tau"]

    q_t = _to_torch(q, device)
    k_t = _to_torch(k, device)
    v_t = _to_torch(v, device)
    g_t = _to_torch(g, device, dtype=torch.float32)
    beta_t = _to_torch(beta, device, dtype=torch.float32)
    tau_t = _to_torch(tau, device, dtype=torch.float32)
    h0_t = _to_torch(h0, device, dtype=torch.float32)

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        out, final_state = triton_chunk_sane(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            tau_t,
            mask=None,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=16,
        )
        user_warnings = [w for w in rec if issubclass(w.category, UserWarning)]
        assert len(user_warnings) == 1
        assert "mask is None" in str(user_warnings[0].message)

    assert final_state is None
    assert out.shape == q.shape[:-1] + (v.shape[-1],)


@pytest.mark.torch
def test_chunk_sane_triton_all_one_mask_equals_no_mask(gdn_sane_inputs, device):
    """全 1 mask 与 mask=None 的输出一致，且 final_state 保留。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    C = T // 16
    all_one_mask = torch.ones((B, C), dtype=torch.float32, device=device)

    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau = gdn_sane_inputs["tau"]

    q_t = _to_torch(q, device)
    k_t = _to_torch(k, device)
    v_t = _to_torch(v, device)
    g_t = _to_torch(g, device, dtype=torch.float32)
    beta_t = _to_torch(beta, device, dtype=torch.float32)
    tau_t = _to_torch(tau, device, dtype=torch.float32)
    h0_t = _to_torch(h0, device, dtype=torch.float32)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        y_no_mask, s_no_mask = triton_chunk_sane(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            tau_t,
            mask=None,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=16,
        )

    y_all_one, s_all_one = triton_chunk_sane(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=all_one_mask,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )

    assert s_no_mask is None
    assert s_all_one is not None
    assert_allclose_with_stats(
        y_no_mask, y_all_one, "no_mask vs all_one_mask output", atol=2e-5, rtol=1e-5
    )


@pytest.mark.torch
def test_chunk_sane_triton_all_zero_mask_matches_non_sane(gdn_sane_inputs, device):
    """全 0 mask 等价于不使用 SANE 的 GDN chunk。"""
    B, T, _, _ = gdn_sane_inputs["q"].shape
    C = T // 16
    zero_mask = torch.zeros((B, C), dtype=torch.float32, device=device)

    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau = gdn_sane_inputs["tau"]

    q_t = _to_torch(q, device)
    k_t = _to_torch(k, device)
    v_t = _to_torch(v, device)
    g_t = _to_torch(g, device, dtype=torch.float32)
    beta_t = _to_torch(beta, device, dtype=torch.float32)
    tau_t = _to_torch(tau, device, dtype=torch.float32)
    h0_t = _to_torch(h0, device, dtype=torch.float32)

    out_triton, state_triton = triton_chunk_sane(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=zero_mask,
        initial_state=h0_t,
        output_final_state=True,
        chunk_size=16,
    )
    out_ref, state_ref = gdn_native_chunk(
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
        out_ref, out_triton, "all_zero_mask vs non-sane output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_triton, "all_zero_mask vs non-sane state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
def test_chunk_sane_triton_irregular_T_raises(gdn_sane_inputs, device):
    """T 不被 chunk_size 整除时 Triton 后端显式抛 ValueError。"""
    T = 34
    q = _to_torch(gdn_sane_inputs["q"][:, :T], device)
    k = _to_torch(gdn_sane_inputs["k"][:, :T], device)
    v = _to_torch(gdn_sane_inputs["v"][:, :T], device)
    g = _to_torch(gdn_sane_inputs["g"][:, :T], device, dtype=torch.float32)
    beta = _to_torch(gdn_sane_inputs["beta"][:, :T], device, dtype=torch.float32)
    tau = _to_torch(
        gdn_sane_inputs["tau"][:, : (T // 16), :], device, dtype=torch.float32
    )

    with pytest.raises(ValueError, match="divisible"):
        triton_chunk_sane(
            q,
            k,
            v,
            g,
            beta,
            tau,
            mask=None,
            output_final_state=False,
            chunk_size=16,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_sane_triton_different_chunk_size(gdn_sane_inputs, device):
    """不同 chunk_size 下 Triton SANE 与 native 均等价。"""
    q, k, v = gdn_sane_inputs["q"], gdn_sane_inputs["k"], gdn_sane_inputs["v"]
    g, beta, h0 = gdn_sane_inputs["g"], gdn_sane_inputs["beta"], gdn_sane_inputs["h0"]
    tau = gdn_sane_inputs["tau"]

    q_t = _to_torch(q, device)
    k_t = _to_torch(k, device)
    v_t = _to_torch(v, device)
    g_t = _to_torch(g, device, dtype=torch.float32)
    beta_t = _to_torch(beta, device, dtype=torch.float32)
    tau_t = _to_torch(tau, device, dtype=torch.float32)
    h0_t = _to_torch(h0, device, dtype=torch.float32)

    for chunk_size in (32, 64):
        C = q.shape[1] // chunk_size
        tau_chunk = tau_t[:, :C, :]
        out_ref, state_ref = native_chunk_sane(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            tau_chunk,
            mask=None,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=chunk_size,
        )
        out_triton, state_triton = triton_chunk_sane(
            q_t,
            k_t,
            v_t,
            g_t,
            beta_t,
            tau_chunk,
            mask=None,
            initial_state=h0_t,
            output_final_state=True,
            chunk_size=chunk_size,
        )

        assert_allclose_with_stats(
            out_ref,
            out_triton,
            f"chunk_size={chunk_size} output",
            atol=1e-2,
            rtol=1e-2,
        )
        assert_allclose_with_stats(
            state_ref,
            state_triton,
            f"chunk_size={chunk_size} state",
            atol=1e-2,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_chunk_sane_factory_dispatch(gdn_sane_inputs, device):
    """get_gated_delta_net_chunk_sane 工厂正确分发 Triton / native 后端。"""
    q_t = _to_torch(gdn_sane_inputs["q"], device)
    k_t = _to_torch(gdn_sane_inputs["k"], device)
    v_t = _to_torch(gdn_sane_inputs["v"], device)
    g_t = _to_torch(gdn_sane_inputs["g"], device, dtype=torch.float32)
    beta_t = _to_torch(gdn_sane_inputs["beta"], device, dtype=torch.float32)
    tau_t = _to_torch(gdn_sane_inputs["tau"], device, dtype=torch.float32)
    h0_t = _to_torch(gdn_sane_inputs["h0"], device, dtype=torch.float32)

    native_op = get_gated_delta_net_chunk_sane(KERNEL_TYPE="native", chunk_size=16)
    triton_op = get_gated_delta_net_chunk_sane(KERNEL_TYPE="triton", chunk_size=16)

    out_native, _ = native_op(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=None,
        initial_state=h0_t,
        output_final_state=False,
    )
    out_triton, _ = triton_op(
        q_t,
        k_t,
        v_t,
        g_t,
        beta_t,
        tau_t,
        mask=None,
        initial_state=h0_t,
        output_final_state=False,
    )

    assert_allclose_with_stats(
        out_native, out_triton, "factory dispatch output", atol=1e-2, rtol=1e-2
    )

"""Gated DeltaNet recurrent SANE Triton 前向/反向数值测试。"""

import numpy as np
import pytest
import torch

pytest.importorskip("triton")

from rwkv_ops import (
    get_gated_delta_net_recurrent_sane,
    get_gated_delta_net_recurrent_sane_inference,
)
from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent as gdn_native_recurrent_no_sane,
    gated_delta_net_recurrent_single_step as gdn_native_single_step_no_sane,
)
from rwkv_ops.gdn_recurrent_sane.native_keras_op import (
    gated_delta_net_recurrent_sane as gdn_native_recurrent,
    gated_delta_net_recurrent_sane_inference as gdn_native_inference,
    gated_delta_net_recurrent_sane_single_step as gdn_native_single_step,
)
from rwkv_ops.gdn_recurrent_sane.torch_triton_kernel import (
    gated_delta_net_recurrent_sane as gdn_triton_recurrent,
    gated_delta_net_recurrent_sane_inference as gdn_triton_inference,
    gated_delta_net_recurrent_sane_single_step as gdn_triton_single_step,
)
from tests.conftest import assert_allclose_with_stats


@pytest.fixture(scope="session")
def gdn_sane_cuda_device(device):
    """Triton recurrent SANE kernel 需要 CUDA，否则跳过整个文件。"""
    if device == "cpu" or not torch.cuda.is_available():
        pytest.skip("Gated DeltaNet recurrent SANE Triton kernel requires CUDA.")
    return torch.device("cuda:0")


def _to_cuda_tensor(arr, device, dtype=torch.float32):
    """把 numpy 数组转成指定 dtype 的 CUDA torch 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _gdn_sane_grads(fn, q, k, v, g, beta, tau, mask, h0):
    """计算 GDN recurrent SANE 算子各输入梯度。"""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    g = g.clone().detach().requires_grad_(True)
    beta = beta.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    h0 = h0.clone().detach().requires_grad_(True)
    out, state = fn(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    loss = (out.float() ** 2).mean() + (state.float() ** 2).mean()
    loss.backward()
    return q.grad, k.grad, v.grad, g.grad, beta.grad, tau.grad, h0.grad


def _make_chunk_size_8_tau_mask(B, H, T, rng):
    """为 chunk_size=8 生成与 tests/conftest.py 同分布的 tau 与 mask。"""
    chunk_num = T // 8
    x = rng.standard_normal((B, chunk_num, H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    mask = rng.integers(0, 2, (B, chunk_num)).astype(np.float32)
    return tau.astype(np.float32), mask


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_recurrent_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """Triton recurrent SANE 训练算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_bwd_matches_native(gdn_sane_inputs, gdn_sane_cuda_device):
    """Triton recurrent SANE 反向梯度与 native Keras autograd 对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)

    grads_tri = _gdn_sane_grads(gdn_triton_recurrent, q, k, v, g, beta, tau, mask, h0)
    grads_ref = _gdn_sane_grads(gdn_native_recurrent, q, k, v, g, beta, tau, mask, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_ref, grads_tri):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.torch
def test_gdn_sane_triton_inference_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """Triton recurrent SANE 推理算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_inference(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_sane_triton_single_step_do_sane_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """Triton recurrent SANE 单步 RNN（do_sane=1）与 native SANE 单步对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"][:, 0], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"][:, 0], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"][:, 0], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"][:, 0], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"][:, 0], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"][:, 0], gdn_sane_cuda_device)
    do_sane = torch.ones(q.shape[0], dtype=torch.float32, device=gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "single step do_sane=1 vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "single step do_sane=1 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_gdn_sane_triton_single_step_do_skip_matches_original_gdn(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """Triton recurrent SANE 单步 RNN（do_sane=0）应跳过 SANE，与原 GDN 单步一致。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"][:, 0], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"][:, 0], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"][:, 0], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"][:, 0], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"][:, 0], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"][:, 0], gdn_sane_cuda_device)
    do_sane = torch.zeros(q.shape[0], dtype=torch.float32, device=gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_single_step(
        q, k, v, g, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step_no_sane(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "single step do_sane=0 vs original GDN output",
        atol=1e-5,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "single step do_sane=0 vs original GDN state",
        atol=1e-5,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_gdn_sane_triton_no_final_state(gdn_sane_inputs, gdn_sane_cuda_device):
    """output_final_state=False 时不返回最终 state。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, tau, mask=mask, output_final_state=False
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, tau, mask=mask, output_final_state=False
    )

    assert state_tri is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref, out_tri, "no-state triton vs native output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_sane_triton_mask_all_ones_matches_no_mask(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """mask=None 与 mask 全 1 输出一致，但 mask=None 时 final_state 为 None 并报警告。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask_ones = torch.ones_like(
        _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)
    )

    out_masked, state_masked = gdn_triton_recurrent(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask_ones,
        initial_state=h0,
        output_final_state=True,
    )
    with pytest.warns(UserWarning, match="mask is None"):
        out_uncond, state_uncond = gdn_triton_recurrent(
            q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )

    assert state_uncond is None
    assert_allclose_with_stats(
        out_uncond, out_masked, "masked(ones) vs uncond output", atol=1e-4, rtol=1e-3
    )
    # final_state 被污染风险：显式 mask 全 1 时返回有效 state，mask=None 时返回 None。
    assert state_masked is not None


@pytest.mark.torch
def test_gdn_sane_native_mask_none_warns_and_returns_none_state(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """native 实现：mask=None 且 output_final_state=True 时报警告并返回 None state。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)

    with pytest.warns(UserWarning, match="mask is None"):
        out, state = gdn_native_recurrent(
            q, k, v, g, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )
    assert state is None
    assert out is not None


@pytest.mark.torch
def test_gdn_sane_triton_mask_all_zeros_matches_original_gdn(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """mask 全 0 时应跳过 SANE，输出与原 GDN recurrent 一致。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask_zeros = torch.zeros_like(
        _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)
    )

    out_sane, state_sane = gdn_triton_recurrent(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask=mask_zeros,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = gdn_native_recurrent_no_sane(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_sane,
        "mask=0 vs original GDN output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_sane,
        "mask=0 vs original GDN state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_gdn_sane_triton_rejects_arbitrary_length(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """recurrent SANE Triton 训练核只支持 T 被 chunk_size 整除。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"][:, :37], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"][:, :37], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"][:, :37], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"][:, :37], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"][:, :37], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"][:, :2], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"][:, :2], gdn_sane_cuda_device)

    with pytest.raises(ValueError, match="必须被 chunk_size"):
        gdn_triton_recurrent(
            q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
        )


@pytest.mark.torch
def test_gdn_sane_triton_bfloat16(gdn_sane_inputs, gdn_sane_cuda_device):
    """bfloat16 I/O 下 Triton 结果仍与 native float32 参考一致。"""
    pytest.importorskip("torch").bfloat16  # noqa: B015

    q = _to_cuda_tensor(
        gdn_sane_inputs["q"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    k = _to_cuda_tensor(
        gdn_sane_inputs["k"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    v = _to_cuda_tensor(
        gdn_sane_inputs["v"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)
    tau = _to_cuda_tensor(gdn_sane_inputs["tau"], gdn_sane_cuda_device)
    mask = _to_cuda_tensor(gdn_sane_inputs["mask"], gdn_sane_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "bf16 triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "bf16 triton vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_recurrent_chunk_size_8_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 Triton recurrent SANE 训练算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, mask8 = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8 = _to_cuda_tensor(mask8, gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    out_tri, state_tri = op_tri(
        q, k, v, g, beta, tau8, mask=mask8, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q,
        k,
        v,
        g,
        beta,
        tau8,
        mask=mask8,
        initial_state=h0,
        output_final_state=True,
        chunk_size=8,
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "chunk_size=8 triton vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "chunk_size=8 triton vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_bwd_chunk_size_8_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 Triton recurrent SANE 反向梯度与 native Keras autograd 对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, mask8 = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8 = _to_cuda_tensor(mask8, gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    def op_ref(*args, **kwargs):
        return gdn_native_recurrent(*args, **kwargs, chunk_size=8)

    grads_tri = _gdn_sane_grads(op_tri, q, k, v, g, beta, tau8, mask8, h0)
    grads_ref = _gdn_sane_grads(op_ref, q, k, v, g, beta, tau8, mask8, h0)

    names = ["q", "k", "v", "g", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_ref, grads_tri):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"triton {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"chunk_size=8 bwd {name}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_inference_chunk_size_8_matches_native(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 Triton recurrent SANE 推理算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, mask8 = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8 = _to_cuda_tensor(mask8, gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane_inference(
        KERNEL_TYPE="triton", chunk_size=8
    )

    out_tri, state_tri = op_tri(
        q, k, v, g, beta, tau8, mask=mask8, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_inference(
        q,
        k,
        v,
        g,
        beta,
        tau8,
        mask=mask8,
        initial_state=h0,
        output_final_state=True,
        chunk_size=8,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "chunk_size=8 triton inference vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "chunk_size=8 triton inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_no_final_state_chunk_size_8(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 output_final_state=False 不返回最终 state。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, mask8 = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8 = _to_cuda_tensor(mask8, gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    out_tri, state_tri = op_tri(
        q, k, v, g, beta, tau8, mask=mask8, output_final_state=False
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, tau8, mask=mask8, output_final_state=False, chunk_size=8
    )

    assert state_tri is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "chunk_size=8 no-state triton vs native output",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_mask_all_ones_matches_no_mask_chunk_size_8(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 mask=None 与 mask 全 1 输出一致，但 final_state 为 None 并报警告。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, _ = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8_ones = torch.ones(B, T // 8, dtype=torch.float32, device=gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    out_masked, state_masked = op_tri(
        q,
        k,
        v,
        g,
        beta,
        tau8,
        mask=mask8_ones,
        initial_state=h0,
        output_final_state=True,
    )
    with pytest.warns(UserWarning, match="mask is None"):
        out_uncond, state_uncond = op_tri(
            q, k, v, g, beta, tau8, mask=None, initial_state=h0, output_final_state=True
        )

    assert state_uncond is None
    assert_allclose_with_stats(
        out_uncond,
        out_masked,
        "chunk_size=8 masked(ones) vs uncond output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert state_masked is not None


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_mask_all_zeros_matches_original_gdn_chunk_size_8(
    gdn_sane_inputs, gdn_sane_cuda_device
):
    """chunk_size=8 时 mask 全 0 应跳过 SANE，输出与原 GDN recurrent 一致。"""
    q = _to_cuda_tensor(gdn_sane_inputs["q"], gdn_sane_cuda_device)
    k = _to_cuda_tensor(gdn_sane_inputs["k"], gdn_sane_cuda_device)
    v = _to_cuda_tensor(gdn_sane_inputs["v"], gdn_sane_cuda_device)
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, _ = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8_zeros = torch.zeros(
        B, T // 8, dtype=torch.float32, device=gdn_sane_cuda_device
    )

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    out_sane, state_sane = op_tri(
        q,
        k,
        v,
        g,
        beta,
        tau8,
        mask=mask8_zeros,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = gdn_native_recurrent_no_sane(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_sane,
        "chunk_size=8 mask=0 vs original GDN output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_sane,
        "chunk_size=8 mask=0 vs original GDN state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_sane_triton_bfloat16_chunk_size_8(gdn_sane_inputs, gdn_sane_cuda_device):
    """chunk_size=8 时 bfloat16 I/O 下 Triton 结果仍与 native float32 参考一致。"""
    pytest.importorskip("torch").bfloat16  # noqa: B015

    q = _to_cuda_tensor(
        gdn_sane_inputs["q"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    k = _to_cuda_tensor(
        gdn_sane_inputs["k"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    v = _to_cuda_tensor(
        gdn_sane_inputs["v"], gdn_sane_cuda_device, dtype=torch.bfloat16
    )
    g = _to_cuda_tensor(gdn_sane_inputs["g"], gdn_sane_cuda_device)
    beta = _to_cuda_tensor(gdn_sane_inputs["beta"], gdn_sane_cuda_device)
    h0 = _to_cuda_tensor(gdn_sane_inputs["h0"], gdn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau8, mask8 = _make_chunk_size_8_tau_mask(B, H, T, np.random.default_rng(42))
    tau8 = _to_cuda_tensor(tau8, gdn_sane_cuda_device)
    mask8 = _to_cuda_tensor(mask8, gdn_sane_cuda_device)

    op_tri = get_gated_delta_net_recurrent_sane(KERNEL_TYPE="triton", chunk_size=8)

    out_tri, state_tri = op_tri(
        q, k, v, g, beta, tau8, mask=mask8, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q,
        k,
        v,
        g,
        beta,
        tau8,
        mask=mask8,
        initial_state=h0,
        output_final_state=True,
        chunk_size=8,
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "chunk_size=8 bf16 triton vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "chunk_size=8 bf16 triton vs native state",
        atol=1e-2,
        rtol=1e-2,
    )

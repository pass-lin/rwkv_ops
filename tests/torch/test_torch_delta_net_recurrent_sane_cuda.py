"""Gated DeltaNet recurrent SANE CUDA 前向/反向数值测试。"""

import numpy as np
import pytest
import torch

from rwkv_ops import (
    get_delta_net_recurrent_sane,
    get_delta_net_recurrent_sane_inference,
)
from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent_no_sane,
    delta_net_recurrent_single_step as dn_native_single_step_no_sane,
)
from rwkv_ops.delta_net_recurrent_sane.native_keras_op import (
    delta_net_recurrent_sane as dn_native_recurrent,
    delta_net_recurrent_sane_inference as dn_native_inference,
    delta_net_recurrent_sane_single_step as dn_native_single_step,
)
from rwkv_ops.delta_net_recurrent_sane.torch_cuda_kernel.delta_net_recurrent_sane_torch import (
    delta_net_recurrent_sane as dn_cuda_recurrent,
    delta_net_recurrent_sane_inference as dn_cuda_inference,
    delta_net_recurrent_sane_single_step as dn_cuda_single_step,
)
from tests.conftest import assert_allclose_with_stats


@pytest.fixture(scope="session")
def dn_sane_cuda_device(device):
    """CUDA recurrent SANE kernel 需要 CUDA，否则跳过整个文件。"""
    if device == "cpu" or not torch.cuda.is_available():
        pytest.skip("Gated DeltaNet recurrent SANE CUDA kernel requires CUDA.")
    return torch.device("cuda:0")


def _to_cuda_tensor(arr, device, dtype=torch.float32):
    """把 numpy 数组转成指定 dtype 的 CUDA torch 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _dn_sane_grads(fn, q, k, v, beta, tau, mask, h0, head_first=None):
    """计算 GDN recurrent SANE 算子各输入梯度。head_first=None 时不传该参数。"""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    beta = beta.clone().detach().requires_grad_(True)
    tau = tau.clone().detach().requires_grad_(True)
    h0 = h0.clone().detach().requires_grad_(True)
    kwargs = dict(mask=mask, initial_state=h0, output_final_state=True)
    if head_first is not None:
        kwargs["head_first"] = head_first
    out, state = fn(q, k, v, beta, tau, **kwargs)
    loss = (out.float() ** 2).mean() + (state.float() ** 2).mean()
    loss.backward()
    return q.grad, k.grad, v.grad, beta.grad, tau.grad, h0.grad


def _make_chunk_size_32_tau_mask(B, H, T, rng):
    """为 chunk_size=32 生成与 tests/conftest.py 同分布的 tau 与 mask。"""
    chunk_num = T // 32
    x = rng.standard_normal((B, chunk_num, H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    mask = rng.integers(0, 2, (B, chunk_num)).astype(np.float32)
    return tau.astype(np.float32), mask


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_recurrent_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """CUDA recurrent SANE 训练算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_bwd_matches_native(delta_net_sane_inputs, dn_sane_cuda_device):
    """CUDA recurrent SANE 反向梯度与 native Keras autograd 对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    grads_cuda = _dn_sane_grads(dn_cuda_recurrent, q, k, v, beta, tau, mask, h0)
    grads_ref = _dn_sane_grads(dn_native_recurrent, q, k, v, beta, tau, mask, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_ref, grads_cuda):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"cuda {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"bwd {name}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.torch
def test_dn_sane_cuda_inference_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """CUDA recurrent SANE 推理算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_dn_sane_cuda_inference_arbitrary_length_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """CUDA recurrent SANE 推理算子支持 T 不被 chunk_size 整除的任意长度。"""
    T = 100
    num_chunks = T // 16
    q = _to_cuda_tensor(delta_net_sane_inputs["q"][:, :T], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"][:, :T], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"][:, :T], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"][:, :T], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(
        delta_net_sane_inputs["tau"][:, :num_chunks], dn_sane_cuda_device
    )
    mask = _to_cuda_tensor(
        delta_net_sane_inputs["mask"][:, :num_chunks], dn_sane_cuda_device
    )

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "arbitrary-length cuda inference vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "arbitrary-length cuda inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_dn_sane_cuda_single_step_do_sane_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """CUDA recurrent SANE 单步 RNN（do_sane=1）与 native SANE 单步对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"][:, 0], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"][:, 0], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"][:, 0], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"][:, 0], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"][:, 0], dn_sane_cuda_device)
    do_sane = torch.ones(q.shape[0], dtype=torch.float32, device=dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_single_step(
        q, k, v, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step(
        q, k, v, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "single step do_sane=1 vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "single step do_sane=1 vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_dn_sane_cuda_single_step_do_skip_matches_original_gdn(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """CUDA recurrent SANE 单步 RNN（do_sane=0）应跳过 SANE，与原 GDN 单步一致。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"][:, 0], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"][:, 0], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"][:, 0], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"][:, 0], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"][:, 0], dn_sane_cuda_device)
    do_sane = torch.zeros(q.shape[0], dtype=torch.float32, device=dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_single_step(
        q, k, v, beta, tau, do_sane, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step_no_sane(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "single step do_sane=0 vs original GDN output",
        atol=1e-5,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "single step do_sane=0 vs original GDN state",
        atol=1e-5,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_dn_sane_cuda_no_final_state(delta_net_sane_inputs, dn_sane_cuda_device):
    """output_final_state=False 时不返回最终 state。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, tau, mask=mask, output_final_state=False
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, tau, mask=mask, output_final_state=False
    )

    assert state_cuda is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref, out_cuda, "no-state cuda vs native output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_dn_sane_cuda_mask_all_ones_matches_no_mask(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """mask=None 与 mask 全 1 输出一致，但 mask=None 时 final_state 为 None 并报警告。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask_ones = torch.ones_like(
        _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)
    )

    out_masked, state_masked = dn_cuda_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask_ones,
        initial_state=h0,
        output_final_state=True,
    )
    with pytest.warns(UserWarning, match="mask is None"):
        out_uncond, state_uncond = dn_cuda_recurrent(
            q, k, v, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )

    assert state_uncond is None
    assert_allclose_with_stats(
        out_uncond, out_masked, "masked(ones) vs uncond output", atol=1e-4, rtol=1e-3
    )
    # final_state 被污染风险：显式 mask 全 1 时返回有效 state，mask=None 时返回 None。
    assert state_masked is not None


@pytest.mark.torch
def test_dn_sane_native_mask_none_warns_and_returns_none_state(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """native 实现：mask=None 且 output_final_state=True 时报警告并返回 None state。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)

    with pytest.warns(UserWarning, match="mask is None"):
        out, state = dn_native_recurrent(
            q, k, v, beta, tau, mask=None, initial_state=h0, output_final_state=True
        )
    assert state is None
    assert out is not None


@pytest.mark.torch
def test_dn_sane_cuda_mask_all_zeros_matches_original_gdn(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """mask 全 0 时应跳过 SANE，输出与原 GDN recurrent 一致。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask_zeros = torch.zeros_like(
        _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)
    )

    out_sane, state_sane = dn_cuda_recurrent(
        q,
        k,
        v,
        beta,
        tau,
        mask=mask_zeros,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = dn_native_recurrent_no_sane(
        q, k, v, beta, initial_state=h0, output_final_state=True
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
def test_dn_sane_cuda_rejects_arbitrary_length(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """recurrent SANE CUDA 训练核只支持 T 被 chunk_size 整除。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"][:, :37], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"][:, :37], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"][:, :37], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"][:, :37], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"][:, :2], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"][:, :2], dn_sane_cuda_device)

    with pytest.raises(ValueError, match="必须被 chunk_size"):
        dn_cuda_recurrent(
            q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
        )


@pytest.mark.torch
def test_dn_sane_cuda_bfloat16(delta_net_sane_inputs, dn_sane_cuda_device):
    """bfloat16 I/O 下 CUDA 结果仍与 native float32 参考一致。"""
    q = _to_cuda_tensor(
        delta_net_sane_inputs["q"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    k = _to_cuda_tensor(
        delta_net_sane_inputs["k"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    v = _to_cuda_tensor(
        delta_net_sane_inputs["v"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "bf16 cuda vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "bf16 cuda vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_recurrent_head_first(delta_net_sane_inputs, dn_sane_cuda_device):
    """head_first=True layout 下前向与反向均与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)
    tau = _to_cuda_tensor(delta_net_sane_inputs["tau"], dn_sane_cuda_device)
    mask = _to_cuda_tensor(delta_net_sane_inputs["mask"], dn_sane_cuda_device)

    q_hf = q.transpose(1, 2).contiguous()
    k_hf = k.transpose(1, 2).contiguous()
    v_hf = v.transpose(1, 2).contiguous()
    beta_hf = beta.transpose(1, 2).contiguous()

    out_cuda, state_cuda = dn_cuda_recurrent(
        q_hf,
        k_hf,
        v_hf,
        beta_hf,
        tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, tau, mask=mask, initial_state=h0, output_final_state=True
    )
    out_ref = out_ref.transpose(1, 2)

    assert_allclose_with_stats(
        out_ref, out_cuda, "head_first cuda vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "head_first cuda vs native state", atol=1e-4, rtol=1e-3
    )

    grads_ref = _dn_sane_grads(dn_native_recurrent, q, k, v, beta, tau, mask, h0)
    grads_cuda = _dn_sane_grads(
        dn_cuda_recurrent,
        q_hf,
        k_hf,
        v_hf,
        beta_hf,
        tau,
        mask,
        h0,
        head_first=True,
    )
    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_ref, grads_cuda):
        if name not in ("tau", "h0"):
            tgt = tgt.transpose(1, 2)
        assert_allclose_with_stats(
            ref,
            tgt,
            f"head_first bwd {name}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_recurrent_chunk_size_32_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 CUDA recurrent SANE 训练算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, mask32 = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32 = _to_cuda_tensor(mask32, dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    out_cuda, state_cuda = op_cuda(
        q, k, v, beta, tau32, mask=mask32, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q,
        k,
        v,
        beta,
        tau32,
        mask=mask32,
        initial_state=h0,
        output_final_state=True,
        chunk_size=32,
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "chunk_size=32 cuda vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "chunk_size=32 cuda vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_bwd_chunk_size_32_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 CUDA recurrent SANE 反向梯度与 native Keras autograd 对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, mask32 = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32 = _to_cuda_tensor(mask32, dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    def op_ref(*args, **kwargs):
        return dn_native_recurrent(*args, **kwargs, chunk_size=32)

    grads_cuda = _dn_sane_grads(op_cuda, q, k, v, beta, tau32, mask32, h0)
    grads_ref = _dn_sane_grads(op_ref, q, k, v, beta, tau32, mask32, h0)

    names = ["q", "k", "v", "beta", "tau", "h0"]
    for name, ref, tgt in zip(names, grads_ref, grads_cuda):
        assert ref is not None, f"native {name} grad is None"
        assert tgt is not None, f"cuda {name} grad is None"
        assert_allclose_with_stats(
            ref,
            tgt,
            f"chunk_size=32 bwd {name}",
            atol=7e-3,
            rtol=1e-3,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_inference_chunk_size_32_matches_native(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 CUDA recurrent SANE 推理算子前向与 native 参考对齐。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, mask32 = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32 = _to_cuda_tensor(mask32, dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane_inference(KERNEL_TYPE="cuda", chunk_size=32)

    out_cuda, state_cuda = op_cuda(
        q, k, v, beta, tau32, mask=mask32, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q,
        k,
        v,
        beta,
        tau32,
        mask=mask32,
        initial_state=h0,
        output_final_state=True,
        chunk_size=32,
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "chunk_size=32 cuda inference vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "chunk_size=32 cuda inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_no_final_state_chunk_size_32(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 output_final_state=False 不返回最终 state。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, mask32 = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32 = _to_cuda_tensor(mask32, dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    out_cuda, state_cuda = op_cuda(
        q, k, v, beta, tau32, mask=mask32, output_final_state=False
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, tau32, mask=mask32, output_final_state=False, chunk_size=32
    )

    assert state_cuda is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "chunk_size=32 no-state cuda vs native output",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_mask_all_ones_matches_no_mask_chunk_size_32(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 mask=None 与 mask 全 1 输出一致，但 final_state 为 None 并报警告。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, _ = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32_ones = torch.ones(B, T // 8, dtype=torch.float32, device=dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    out_masked, state_masked = op_cuda(
        q,
        k,
        v,
        beta,
        tau32,
        mask=mask32_ones,
        initial_state=h0,
        output_final_state=True,
    )
    with pytest.warns(UserWarning, match="mask is None"):
        out_uncond, state_uncond = op_cuda(
            q, k, v, beta, tau32, mask=None, initial_state=h0, output_final_state=True
        )

    assert state_uncond is None
    assert_allclose_with_stats(
        out_uncond,
        out_masked,
        "chunk_size=32 masked(ones) vs uncond output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert state_masked is not None


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_mask_all_zeros_matches_original_gdn_chunk_size_32(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 mask 全 0 应跳过 SANE，输出与原 GDN recurrent 一致。"""
    q = _to_cuda_tensor(delta_net_sane_inputs["q"], dn_sane_cuda_device)
    k = _to_cuda_tensor(delta_net_sane_inputs["k"], dn_sane_cuda_device)
    v = _to_cuda_tensor(delta_net_sane_inputs["v"], dn_sane_cuda_device)
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, _ = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32_zeros = torch.zeros(
        B, T // 8, dtype=torch.float32, device=dn_sane_cuda_device
    )

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    out_sane, state_sane = op_cuda(
        q,
        k,
        v,
        beta,
        tau32,
        mask=mask32_zeros,
        initial_state=h0,
        output_final_state=True,
    )
    out_ref, state_ref = dn_native_recurrent_no_sane(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_sane,
        "chunk_size=32 mask=0 vs original GDN output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_sane,
        "chunk_size=32 mask=0 vs original GDN state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_dn_sane_cuda_bfloat16_chunk_size_32(
    delta_net_sane_inputs, dn_sane_cuda_device
):
    """chunk_size=32 时 bfloat16 I/O 下 CUDA 结果仍与 native float32 参考一致。"""
    q = _to_cuda_tensor(
        delta_net_sane_inputs["q"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    k = _to_cuda_tensor(
        delta_net_sane_inputs["k"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    v = _to_cuda_tensor(
        delta_net_sane_inputs["v"], dn_sane_cuda_device, dtype=torch.bfloat16
    )
    beta = _to_cuda_tensor(delta_net_sane_inputs["beta"], dn_sane_cuda_device)
    h0 = _to_cuda_tensor(delta_net_sane_inputs["h0"], dn_sane_cuda_device)

    B, T, H, _ = q.shape
    tau32, mask32 = _make_chunk_size_32_tau_mask(B, H, T, np.random.default_rng(42))
    tau32 = _to_cuda_tensor(tau32, dn_sane_cuda_device)
    mask32 = _to_cuda_tensor(mask32, dn_sane_cuda_device)

    op_cuda = get_delta_net_recurrent_sane(KERNEL_TYPE="cuda", chunk_size=32)

    out_cuda, state_cuda = op_cuda(
        q, k, v, beta, tau32, mask=mask32, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q,
        k,
        v,
        beta,
        tau32,
        mask=mask32,
        initial_state=h0,
        output_final_state=True,
        chunk_size=32,
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "chunk_size=32 bf16 cuda vs native output",
        atol=1e-2,
        rtol=1e-2,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "chunk_size=32 bf16 cuda vs native state",
        atol=1e-2,
        rtol=1e-2,
    )

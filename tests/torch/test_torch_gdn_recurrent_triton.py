"""Gated DeltaNet recurrent Triton 前向 kernel 的 Torch 后端测试。"""

import pytest
import torch

pytest.importorskip("triton")

from rwkv_ops.gdn_recurrent.native_keras_op import (
    gated_delta_net_recurrent as gdn_native_recurrent,
    gated_delta_net_recurrent_inference as gdn_native_inference,
    gated_delta_net_recurrent_single_step as gdn_native_single_step,
)
from rwkv_ops.gdn_recurrent.torch_triton_kernel import (
    gated_delta_net_recurrent as gdn_triton_recurrent,
    gated_delta_net_recurrent_inference as gdn_triton_inference,
    gated_delta_net_recurrent_single_step as gdn_triton_single_step,
)
from tests.conftest import assert_allclose_with_stats


@pytest.fixture(scope="session")
def gdn_cuda_device(device):
    """Triton recurrent kernel 需要 CUDA，否则跳过整个文件。"""
    if device == "cpu" or not torch.cuda.is_available():
        pytest.skip("Gated DeltaNet recurrent Triton kernel requires CUDA.")
    return torch.device("cuda:0")


def _to_cuda_tensor(arr, device, dtype=torch.float32):
    """把 numpy 数组转成指定 dtype 的 CUDA torch 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _gdn_grads(fn, q, k, v, g, beta, h0):
    """计算 GDN recurrent 算子各输入梯度。"""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    g = g.clone().detach().requires_grad_(True)
    beta = beta.clone().detach().requires_grad_(True)
    h0 = h0.clone().detach().requires_grad_(True)
    out, state = fn(q, k, v, g, beta, initial_state=h0, output_final_state=True)
    loss = (out.float() ** 2).mean() + (state.float() ** 2).mean()
    loss.backward()
    return q.grad, k.grad, v.grad, g.grad, beta.grad, h0.grad


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_triton_recurrent_matches_native(gdn_inputs, gdn_cuda_device):
    """Triton recurrent 训练算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_triton_inference_matches_native(gdn_inputs, gdn_cuda_device):
    """Triton recurrent 推理算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_inference(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_triton_single_step_matches_native(gdn_inputs, gdn_cuda_device):
    """Triton recurrent 单步 RNN 算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(gdn_inputs["q"][:, 0], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"][:, 0], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"][:, 0], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"][:, 0], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"][:, 0], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_single_step(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_triton_no_final_state(gdn_inputs, gdn_cuda_device):
    """output_final_state=False 时不返回最终 state。"""
    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, output_final_state=False
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, output_final_state=False
    )

    assert state_tri is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref, out_tri, "no-state triton vs native output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_gdn_triton_initial_state_broadcast(gdn_inputs, gdn_cuda_device):
    """initial_state [1, H, K, V] 应能广播到 batch。"""
    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"][:1], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "broadcast state triton vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "broadcast state triton vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_gdn_triton_arbitrary_length(gdn_inputs, gdn_cuda_device):
    """recurrent kernel 支持任意长度（不被 chunk 整除）。"""
    q = _to_cuda_tensor(gdn_inputs["q"][:, :37], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"][:, :37], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"][:, :37], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"][:, :37], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"][:, :37], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = gdn_native_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "arbitrary length triton vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "arbitrary length triton vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_gdn_triton_bfloat16(gdn_inputs, gdn_cuda_device):
    """bfloat16 I/O 下 Triton 结果仍与 native float32 参考一致。"""
    pytest.importorskip("torch").bfloat16  # noqa: B015

    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device, dtype=torch.bfloat16)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device, dtype=torch.bfloat16)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device, dtype=torch.bfloat16)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device, dtype=torch.bfloat16)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device, dtype=torch.bfloat16)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device, dtype=torch.float32)

    out_tri, state_tri = gdn_triton_recurrent(
        q, k, v, g, beta, initial_state=h0, output_final_state=True
    )

    q_ref = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device, dtype=torch.float32)
    k_ref = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device, dtype=torch.float32)
    v_ref = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device, dtype=torch.float32)
    g_ref = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device, dtype=torch.float32)
    beta_ref = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device, dtype=torch.float32)

    out_ref, state_ref = gdn_native_recurrent(
        q_ref, k_ref, v_ref, g_ref, beta_ref, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "bf16 triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "bf16 triton vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_gdn_triton_recurrent_backward(gdn_inputs, gdn_cuda_device):
    """Triton recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(gdn_inputs["q"], gdn_cuda_device)
    k = _to_cuda_tensor(gdn_inputs["k"], gdn_cuda_device)
    v = _to_cuda_tensor(gdn_inputs["v"], gdn_cuda_device)
    g = _to_cuda_tensor(gdn_inputs["g"], gdn_cuda_device)
    beta = _to_cuda_tensor(gdn_inputs["beta"], gdn_cuda_device)
    h0 = _to_cuda_tensor(gdn_inputs["h0"], gdn_cuda_device)

    g_ref = _gdn_grads(gdn_native_recurrent, q, k, v, g, beta, h0)
    g_tri = _gdn_grads(gdn_triton_recurrent, q, k, v, g, beta, h0)
    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, gr, gt in zip(names, g_ref, g_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"grad_{name} triton vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("V", [64, 256, 512])
def test_gdn_triton_recurrent_backward_various_v(V, gdn_cuda_device):
    """反向在不同 V 维度（含 NV>1 的 atomic_add 路径）下与 native 对齐。"""
    torch.manual_seed(42)
    B, T, H, K = 2, 32, 4, 64
    q = torch.randn(B, T, H, K, device=gdn_cuda_device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=gdn_cuda_device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=gdn_cuda_device, requires_grad=True)
    g = torch.randn(B, T, H, device=gdn_cuda_device, requires_grad=True)
    beta = torch.sigmoid(
        torch.randn(B, T, H, device=gdn_cuda_device, requires_grad=True)
    )
    h0 = torch.randn(B, H, K, V, device=gdn_cuda_device, requires_grad=True)

    g_ref = _gdn_grads(gdn_native_recurrent, q, k, v, g, beta, h0)
    g_tri = _gdn_grads(gdn_triton_recurrent, q, k, v, g, beta, h0)
    names = ["q", "k", "v", "g", "beta", "h0"]
    for name, gr, gt in zip(names, g_ref, g_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"V={V} grad_{name} triton vs native",
            atol=7e-3,
            rtol=1e-2,
        )

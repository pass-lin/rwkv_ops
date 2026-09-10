"""DeltaNet recurrent Triton kernel 的 Torch 后端测试。"""

import pytest
import torch

pytest.importorskip("triton")

from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent,
    delta_net_recurrent_inference as dn_native_inference,
    delta_net_recurrent_single_step as dn_native_single_step,
)
from rwkv_ops.delta_net_recurrent.torch_triton_kernel import (
    delta_net_recurrent as dn_triton_recurrent,
    delta_net_recurrent_inference as dn_triton_inference,
    delta_net_recurrent_single_step as dn_triton_single_step,
)
from tests.conftest import assert_allclose_with_stats


@pytest.fixture(scope="session")
def dn_cuda_device(device):
    """Triton recurrent kernel 需要 CUDA，否则跳过整个文件。"""
    if device == "cpu" or not torch.cuda.is_available():
        pytest.skip("DeltaNet recurrent Triton kernel requires CUDA.")
    return torch.device("cuda:0")


def _to_cuda_tensor(arr, device, dtype=torch.float32):
    """把 numpy 数组转成指定 dtype 的 CUDA torch 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _dn_grads(fn, q, k, v, beta, h0, chunk_size=16):
    """计算 DeltaNet recurrent 算子各输入梯度。"""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    beta = beta.clone().detach().requires_grad_(True)
    h0 = h0.clone().detach().requires_grad_(True)
    out, state = fn(
        q,
        k,
        v,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=chunk_size,
    )
    loss = (out.float() ** 2).mean() + (state.float() ** 2).mean()
    loss.backward()
    return q.grad, k.grad, v.grad, beta.grad, h0.grad


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_triton_recurrent_matches_native(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 训练算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_triton_inference_matches_native(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 推理算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_triton_single_step_matches_native(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 单步 RNN 算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"][:, 0], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"][:, 0], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"][:, 0], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"][:, 0], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "triton single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "triton single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_triton_single_step_loop_matches_recurrent(
    delta_net_inputs, dn_cuda_device
):
    """Triton 单步逐步跑完整序列应与整序列 recurrent 对齐。"""
    steps = 16
    q = _to_cuda_tensor(delta_net_inputs["q"][:, :steps], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"][:, :steps], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"][:, :steps], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"][:, :steps], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    state = h0
    outs = []
    for t in range(steps):
        out_t, state = dn_triton_single_step(
            q[:, t],
            k[:, t],
            v[:, t],
            beta[:, t],
            initial_state=state,
            output_final_state=True,
        )
        outs.append(out_t)
    out_step = torch.stack(outs, dim=1)

    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_step,
        "triton single step loop vs native recurrent output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state,
        "triton single step loop vs native recurrent state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_triton_no_final_state(delta_net_inputs, dn_cuda_device):
    """output_final_state=False 时不返回最终 state。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)

    out_tri, state_tri = dn_triton_recurrent(q, k, v, beta, output_final_state=False)
    out_ref, state_ref = dn_native_recurrent(q, k, v, beta, output_final_state=False)

    assert state_tri is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref, out_tri, "no-state triton vs native output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_triton_initial_state_broadcast(delta_net_inputs, dn_cuda_device):
    """initial_state [1, H, K, V] 应能广播到 batch。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"][:1], dn_cuda_device)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
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
def test_delta_net_triton_head_first(delta_net_inputs, dn_cuda_device):
    """head_first=True 时 Triton 结果应与默认 layout 的 native 参考对齐。"""
    q_hf = _to_cuda_tensor(delta_net_inputs["q"].transpose(0, 2, 1, 3), dn_cuda_device)
    k_hf = _to_cuda_tensor(delta_net_inputs["k"].transpose(0, 2, 1, 3), dn_cuda_device)
    v_hf = _to_cuda_tensor(delta_net_inputs["v"].transpose(0, 2, 1, 3), dn_cuda_device)
    beta_hf = _to_cuda_tensor(
        delta_net_inputs["beta"].transpose(0, 2, 1), dn_cuda_device
    )
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_recurrent(
        q_hf,
        k_hf,
        v_hf,
        beta_hf,
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )

    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri.transpose(1, 2),
        "head_first triton vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "head_first triton vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_triton_rejects_arbitrary_length(delta_net_inputs, dn_cuda_device):
    """recurrent Triton 训练核只支持 T 被 chunk_size 整除。"""
    q = _to_cuda_tensor(delta_net_inputs["q"][:, :37], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"][:, :37], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"][:, :37], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"][:, :37], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    with pytest.raises(ValueError, match="必须被 chunk_size"):
        dn_triton_recurrent(q, k, v, beta, initial_state=h0, output_final_state=True)


@pytest.mark.torch
def test_delta_net_triton_bfloat16(delta_net_inputs, dn_cuda_device):
    """bfloat16 I/O 下 Triton 结果仍与 native float32 参考一致。"""
    pytest.importorskip("torch").bfloat16  # noqa: B015

    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device, dtype=torch.bfloat16)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device, dtype=torch.bfloat16)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device, dtype=torch.bfloat16)
    beta = _to_cuda_tensor(
        delta_net_inputs["beta"], dn_cuda_device, dtype=torch.bfloat16
    )
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device, dtype=torch.float32)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    q_ref = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device, dtype=torch.float32)
    k_ref = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device, dtype=torch.float32)
    v_ref = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device, dtype=torch.float32)
    beta_ref = _to_cuda_tensor(
        delta_net_inputs["beta"], dn_cuda_device, dtype=torch.float32
    )

    out_ref, state_ref = dn_native_recurrent(
        q_ref, k_ref, v_ref, beta_ref, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_tri, "bf16 triton vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_tri, "bf16 triton vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_triton_recurrent_backward(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0)
    g_tri = _dn_grads(dn_triton_recurrent, q, k, v, beta, h0)
    names = ["q", "k", "v", "beta", "h0"]
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
def test_delta_net_triton_recurrent_backward_various_v(V, dn_cuda_device):
    """反向在不同 V 维度（含 NV>1 的 atomic_add 路径）下与 native 对齐。"""
    torch.manual_seed(42)
    B, T, H, K = 2, 32, 4, 64
    q = torch.randn(B, T, H, K, device=dn_cuda_device, requires_grad=True)
    k = torch.randn(B, T, H, K, device=dn_cuda_device, requires_grad=True)
    v = torch.randn(B, T, H, V, device=dn_cuda_device, requires_grad=True)
    beta = torch.sigmoid(
        torch.randn(B, T, H, device=dn_cuda_device, requires_grad=True)
    )
    h0 = torch.randn(B, H, K, V, device=dn_cuda_device, requires_grad=True)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0)
    g_tri = _dn_grads(dn_triton_recurrent, q, k, v, beta, h0)
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gt in zip(names, g_ref, g_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"V={V} grad_{name} triton vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_triton_recurrent_chunk_size_8(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 训练算子在 chunk_size=8 时前向与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )

    assert_allclose_with_stats(
        out_ref,
        out_tri,
        "chunk_size=8 triton recurrent vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_tri,
        "chunk_size=8 triton recurrent vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_triton_inference_chunk_size_8(delta_net_inputs, dn_cuda_device):
    """Triton recurrent 推理算子在 chunk_size=8 时前向与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_tri, state_tri = dn_triton_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
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
def test_delta_net_triton_recurrent_backward_chunk_size_8(
    delta_net_inputs, dn_cuda_device
):
    """Triton recurrent 训练算子在 chunk_size=8 时反向梯度与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0, chunk_size=8)
    g_tri = _dn_grads(dn_triton_recurrent, q, k, v, beta, h0, chunk_size=8)
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gt in zip(names, g_ref, g_tri):
        assert_allclose_with_stats(
            gr,
            gt,
            f"chunk_size=8 grad_{name} triton vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
def test_delta_net_triton_bfloat16_chunk_size_8(delta_net_inputs, dn_cuda_device):
    """bfloat16 I/O 下 Triton 在 chunk_size=8 时仍与 native float32 参考一致。"""
    pytest.importorskip("torch").bfloat16  # noqa: B015

    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device, dtype=torch.bfloat16)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device, dtype=torch.bfloat16)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device, dtype=torch.bfloat16)
    beta = _to_cuda_tensor(
        delta_net_inputs["beta"], dn_cuda_device, dtype=torch.bfloat16
    )
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device, dtype=torch.float32)

    out_tri, state_tri = dn_triton_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )

    q_ref = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device, dtype=torch.float32)
    k_ref = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device, dtype=torch.float32)
    v_ref = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device, dtype=torch.float32)
    beta_ref = _to_cuda_tensor(
        delta_net_inputs["beta"], dn_cuda_device, dtype=torch.float32
    )

    out_ref, state_ref = dn_native_recurrent(
        q_ref,
        k_ref,
        v_ref,
        beta_ref,
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

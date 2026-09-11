"""DeltaNet recurrent CUDA kernel 的 Torch 后端测试。"""

import numpy as np
import pytest
import torch

from rwkv_ops.delta_net_recurrent.native_keras_op import (
    delta_net_recurrent as dn_native_recurrent,
    delta_net_recurrent_inference as dn_native_inference,
    delta_net_recurrent_single_step as dn_native_single_step,
)
from rwkv_ops.delta_net_recurrent.torch_cuda_kernel.delta_net_recurrent_torch import (
    delta_net_recurrent as dn_cuda_recurrent,
    delta_net_recurrent_inference as dn_cuda_inference,
    delta_net_recurrent_single_step as dn_cuda_single_step,
)
from tests.conftest import assert_allclose_with_stats


@pytest.fixture(scope="session")
def dn_cuda_device(device):
    """CUDA recurrent kernel 需要 CUDA，否则跳过整个文件。"""
    if device == "cpu" or not torch.cuda.is_available():
        pytest.skip("DeltaNet recurrent CUDA kernel requires CUDA.")
    return torch.device("cuda:0")


def _to_cuda_tensor(arr, device, dtype=torch.float32):
    """把 numpy 数组转成指定 dtype 的 CUDA torch 张量。"""
    return torch.from_numpy(arr).to(device=device, dtype=dtype)


def _dn_grads(fn, q, k, v, beta, h0, chunk_size=16, head_first=None):
    """计算 DeltaNet recurrent 算子各输入梯度。head_first=None 时不传该参数。"""
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    v = v.clone().detach().requires_grad_(True)
    beta = beta.clone().detach().requires_grad_(True)
    h0 = h0.clone().detach().requires_grad_(True)
    kwargs = dict(initial_state=h0, output_final_state=True, chunk_size=chunk_size)
    if head_first is not None:
        kwargs["head_first"] = head_first
    out, state = fn(q, k, v, beta, **kwargs)
    loss = (out.float() ** 2).mean() + (state.float() ** 2).mean()
    loss.backward()
    return q.grad, k.grad, v.grad, beta.grad, h0.grad


def _stable_inputs(B, T, H, K, V, device, seed=42):
    """生成稳定分布的 DeltaNet 输入（与根 conftest 的 delta_net_inputs 同款分布）。"""
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, V), dtype=np.float32)
    beta_raw = rng.standard_normal((B, T, H), dtype=np.float32)
    beta = (1.0 / (1.0 + np.exp(-beta_raw))).astype(np.float32)
    h0 = rng.standard_normal((B, H, K, V), dtype=np.float32) * 0.1
    return [torch.from_numpy(x).to(device) for x in (q, k, v, beta, h0)]


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_cuda_recurrent_matches_native(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 训练算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda recurrent vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda recurrent vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_cuda_inference_matches_native(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 推理算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda inference vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda inference vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_cuda_single_step_matches_native(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 单步 RNN 算子前向结果应与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"][:, 0], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"][:, 0], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"][:, 0], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"][:, 0], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_single_step(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_cuda, "cuda single step vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "cuda single step vs native state", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_cuda_no_final_state(delta_net_inputs, dn_cuda_device):
    """output_final_state=False 时不返回最终 state。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(q, k, v, beta, output_final_state=False)
    out_ref, state_ref = dn_native_recurrent(q, k, v, beta, output_final_state=False)

    assert state_cuda is None
    assert state_ref is None
    assert_allclose_with_stats(
        out_ref, out_cuda, "no-state cuda vs native output", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
def test_delta_net_cuda_no_initial_state(delta_net_inputs, dn_cuda_device):
    """initial_state=None 时 CUDA 结果与 native 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=None, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=None, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "no initial_state cuda vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "no initial_state cuda vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_cuda_initial_state_broadcast(delta_net_inputs, dn_cuda_device):
    """initial_state [1, H, K, V] 应能广播到 batch。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"][:1], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "broadcast state cuda vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "broadcast state cuda vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_cuda_rejects_arbitrary_length(delta_net_inputs, dn_cuda_device):
    """recurrent CUDA 训练核只支持 T 被 chunk_size 整除。"""
    q = _to_cuda_tensor(delta_net_inputs["q"][:, :37], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"][:, :37], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"][:, :37], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"][:, :37], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    with pytest.raises(ValueError, match="必须被 chunk_size"):
        dn_cuda_recurrent(q, k, v, beta, initial_state=h0, output_final_state=True)


@pytest.mark.torch
def test_delta_net_cuda_bfloat16(delta_net_inputs, dn_cuda_device):
    """bfloat16 I/O 下 CUDA 结果仍与 native float32 参考一致。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device, dtype=torch.bfloat16)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device, dtype=torch.bfloat16)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device, dtype=torch.bfloat16)
    beta = _to_cuda_tensor(
        delta_net_inputs["beta"], dn_cuda_device, dtype=torch.bfloat16
    )
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device, dtype=torch.float32)

    out_cuda, state_cuda = dn_cuda_recurrent(
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
        out_ref, out_cuda, "bf16 cuda vs native output", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "bf16 cuda vs native state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_cuda_recurrent_backward(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 训练算子反向梯度与 native Keras 参考对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0)
    g_cuda = _dn_grads(dn_cuda_recurrent, q, k, v, beta, h0)
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, g_ref, g_cuda):
        assert_allclose_with_stats(
            gr,
            gc,
            f"grad_{name} cuda vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_cuda_recurrent_head_first(delta_net_inputs, dn_cuda_device):
    """head_first=True layout 下前向与反向均与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    q_hf = q.transpose(1, 2).contiguous()
    k_hf = k.transpose(1, 2).contiguous()
    v_hf = v.transpose(1, 2).contiguous()
    beta_hf = beta.transpose(1, 2).contiguous()

    out_cuda, state_cuda = dn_cuda_recurrent(
        q_hf,
        k_hf,
        v_hf,
        beta_hf,
        initial_state=h0,
        output_final_state=True,
        head_first=True,
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref = out_ref.transpose(1, 2)

    assert_allclose_with_stats(
        out_ref, out_cuda, "head_first cuda vs native output", atol=1e-4, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_cuda, "head_first cuda vs native state", atol=1e-4, rtol=1e-3
    )

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0)
    g_cuda = _dn_grads(
        dn_cuda_recurrent, q_hf, k_hf, v_hf, beta_hf, h0, head_first=True
    )
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, g_ref, g_cuda):
        if name != "h0":
            gc = gc.transpose(1, 2)
        assert_allclose_with_stats(
            gr,
            gc,
            f"head_first grad_{name} cuda vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("V", [64, 256, 512])
def test_delta_net_cuda_recurrent_backward_various_v(V, dn_cuda_device):
    """反向在不同 V 维度下与 native 对齐。"""
    q, k, v, beta, h0 = _stable_inputs(2, 32, 4, 64, V, dn_cuda_device)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0)
    g_cuda = _dn_grads(dn_cuda_recurrent, q, k, v, beta, h0)
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, g_ref, g_cuda):
        assert_allclose_with_stats(
            gr,
            gc,
            f"V={V} grad_{name} cuda vs native",
            atol=7e-3,
            rtol=1e-2,
        )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_cuda_recurrent_chunk_size_8(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 训练算子在 chunk_size=8 时前向与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )
    out_ref, state_ref = dn_native_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "chunk_size=8 cuda recurrent vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "chunk_size=8 cuda recurrent vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
def test_delta_net_cuda_inference_chunk_size_8(delta_net_inputs, dn_cuda_device):
    """CUDA recurrent 推理算子在 chunk_size=8 时前向与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    out_cuda, state_cuda = dn_cuda_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )
    out_ref, state_ref = dn_native_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=8
    )

    assert_allclose_with_stats(
        out_ref,
        out_cuda,
        "chunk_size=8 cuda inference vs native output",
        atol=1e-4,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_ref,
        state_cuda,
        "chunk_size=8 cuda inference vs native state",
        atol=1e-4,
        rtol=1e-3,
    )


@pytest.mark.torch
@pytest.mark.slow
def test_delta_net_cuda_recurrent_backward_chunk_size_8(
    delta_net_inputs, dn_cuda_device
):
    """CUDA recurrent 训练算子在 chunk_size=8 时反向梯度与 native 对齐。"""
    q = _to_cuda_tensor(delta_net_inputs["q"], dn_cuda_device)
    k = _to_cuda_tensor(delta_net_inputs["k"], dn_cuda_device)
    v = _to_cuda_tensor(delta_net_inputs["v"], dn_cuda_device)
    beta = _to_cuda_tensor(delta_net_inputs["beta"], dn_cuda_device)
    h0 = _to_cuda_tensor(delta_net_inputs["h0"], dn_cuda_device)

    g_ref = _dn_grads(dn_native_recurrent, q, k, v, beta, h0, chunk_size=8)
    g_cuda = _dn_grads(dn_cuda_recurrent, q, k, v, beta, h0, chunk_size=8)
    names = ["q", "k", "v", "beta", "h0"]
    for name, gr, gc in zip(names, g_ref, g_cuda):
        assert_allclose_with_stats(
            gr,
            gc,
            f"chunk_size=8 grad_{name} cuda vs native",
            atol=7e-3,
            rtol=1e-2,
        )

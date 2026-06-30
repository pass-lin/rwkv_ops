"""
mHC Pre-Op Torch Triton 正确性测试。
"""

import pytest
import torch

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.fixture(scope="module")
def ops(device):
    from rwkv_ops.mhc_kernel.native_op import (
        sinkhorn_knopp as native_sinkhorn,
        stream_aggregate as native_aggregate,
    )
    from rwkv_ops.mhc_kernel.torch_triton_op.mhc_pre_op import mhc_pre_op_fused

    return native_sinkhorn, native_aggregate, mhc_pre_op_fused


def _make_tensors(inputs, device, grad=False):
    x = _to_torch(inputs["x"], "bfloat16", device).requires_grad_(grad)
    h_res = _to_torch(inputs["h_res"], "float32", device).requires_grad_(grad)
    h_pre = _to_torch(inputs["h_pre"], "float32", device).requires_grad_(grad)
    return x, h_res, h_pre


@pytest.mark.torch
@pytest.mark.slow
def test_mhc_pre_op_forward(ops, mhc_pre_inputs, device):
    native_sinkhorn, native_aggregate, triton_op = ops
    x_n, hr_n, hp_n = _make_tensors(mhc_pre_inputs, device, grad=False)
    x_t, hr_t, hp_t = _make_tensors(mhc_pre_inputs, device, grad=False)
    with torch.no_grad():
        x_t.copy_(x_n)
        hr_t.copy_(hr_n)
        hp_t.copy_(hp_n)

    x_in_n = native_aggregate(x_n, hp_n)
    h_res_n = native_sinkhorn(hr_n, num_iters=20)

    x_in_t, h_res_t = triton_op(x_t, hr_t, hp_t, num_iters=20)

    assert_allclose_with_stats(x_in_n, x_in_t, "x_layer_in", atol=1e-2, rtol=1e-2)
    assert_allclose_with_stats(h_res_n, h_res_t, "H_res", atol=1e-2, rtol=1e-2)


@pytest.mark.torch
@pytest.mark.slow
def test_mhc_pre_op_backward(ops, mhc_pre_inputs, device):
    native_sinkhorn, native_aggregate, triton_op = ops
    x_n, hr_n, hp_n = _make_tensors(mhc_pre_inputs, device, grad=True)
    x_t, hr_t, hp_t = _make_tensors(mhc_pre_inputs, device, grad=True)
    with torch.no_grad():
        x_t.copy_(x_n)
        hr_t.copy_(hr_n)
        hp_t.copy_(hp_n)

    x_in_n = native_aggregate(x_n, hp_n)
    h_res_n = native_sinkhorn(hr_n, num_iters=20)
    loss_n = (x_in_n.float() ** 2).mean() + (h_res_n**2).mean()
    loss_n.backward()

    x_in_t, h_res_t = triton_op(x_t, hr_t, hp_t, num_iters=20)
    loss_t = (x_in_t.float() ** 2).mean() + (h_res_t**2).mean()
    loss_t.backward()

    assert_allclose_with_stats(x_n.grad, x_t.grad, "grad_x", atol=1e-2, rtol=1e-2)
    assert_allclose_with_stats(hr_n.grad, hr_t.grad, "grad_h_res", atol=1e-2, rtol=1e-2)
    assert_allclose_with_stats(hp_n.grad, hp_t.grad, "grad_h_pre", atol=1e-2, rtol=1e-2)

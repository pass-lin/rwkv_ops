"""
mHC Post-Op Torch Triton 正确性测试。
"""

import pytest
import torch

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("triton")


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.fixture(scope="module")
def ops():
    from rwkv_ops.mhc_kernel.native_op import mhc_post_op as native_mhc_op
    from rwkv_ops.mhc_kernel.torch_triton_op.mhc_post_op import (
        mhc_post_op as triton_mhc_op,
    )

    return native_mhc_op, triton_mhc_op


def _make_tensors(inputs, device, grad=False):
    layer_out = _to_torch(inputs["layer_out"], "bfloat16", device).requires_grad_(grad)
    x_expanded = _to_torch(inputs["x_expanded"], "bfloat16", device).requires_grad_(
        grad
    )
    h_post = _to_torch(inputs["h_post"], "float32", device).requires_grad_(grad)
    H_res = _to_torch(inputs["H_res"], "float32", device).requires_grad_(grad)
    return layer_out, x_expanded, h_post, H_res


@pytest.mark.torch
@pytest.mark.slow
def test_mhc_post_op_forward(ops, mhc_post_inputs, device):
    native_mhc_op, triton_mhc_op = ops
    l_n, x_n, h_n, H_n = _make_tensors(mhc_post_inputs, device, grad=False)
    l_t, x_t, h_t, H_t = _make_tensors(mhc_post_inputs, device, grad=False)
    with torch.no_grad():
        l_t.copy_(l_n)
        x_t.copy_(x_n)
        h_t.copy_(h_n)
        H_t.copy_(H_n)

    out_n = native_mhc_op(layer_out=l_n, x_expanded=x_n, h_post_raw=h_n, H_res=H_n)
    out_t = triton_mhc_op(layer_out=l_t, x_expanded=x_t, h_post_raw=h_t, H_res=H_t)

    assert_allclose_with_stats(out_n, out_t, "output", atol=1e-2, rtol=1e-2)


@pytest.mark.torch
@pytest.mark.slow
def test_mhc_post_op_backward(ops, mhc_post_inputs, device):
    native_mhc_op, triton_mhc_op = ops
    l_n, x_n, h_n, H_n = _make_tensors(mhc_post_inputs, device, grad=True)
    l_t, x_t, h_t, H_t = _make_tensors(mhc_post_inputs, device, grad=True)
    with torch.no_grad():
        l_t.copy_(l_n)
        x_t.copy_(x_n)
        h_t.copy_(h_n)
        H_t.copy_(H_n)

    out_n = native_mhc_op(layer_out=l_n, x_expanded=x_n, h_post_raw=h_n, H_res=H_n)
    loss_n = (out_n.float() ** 2).mean()
    loss_n.backward()

    out_t = triton_mhc_op(layer_out=l_t, x_expanded=x_t, h_post_raw=h_t, H_res=H_t)
    loss_t = (out_t.float() ** 2).mean()
    loss_t.backward()

    names = ["layer_out", "x_expanded", "h_post_raw", "H_res"]
    for ref, tgt, name in zip(
        [l_n.grad, x_n.grad, h_n.grad, H_n.grad],
        [l_t.grad, x_t.grad, h_t.grad, H_t.grad],
        names,
    ):
        assert_allclose_with_stats(ref, tgt, f"grad_{name}", atol=1e-2, rtol=1e-2)

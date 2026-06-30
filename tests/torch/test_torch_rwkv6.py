"""
RWKV-6 Torch CUDA kernel 数值测试。

运行方式：
    KERAS_BACKEND=torch pytest tests/torch/test_rwkv6.py -v
"""

import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.mark.torch
@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_rwkv6_forward_state(
    torch_op, native_op, sample_inputs, sample_shape, device, dtype
):
    B, T, H, N = sample_shape
    r, k, v, w, u, init = sample_inputs

    # ground truth：native float32
    r_ref = _to_torch(r, "float32", device)
    k_ref = _to_torch(k, "float32", device)
    v_ref = _to_torch(v, "float32", device)
    w_ref = _to_torch(w, "float32", device)
    u_ref = _to_torch(u, "float32", device)
    init_ref = _to_torch(init, "float32", device)

    y_ref, s_ref = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_ref,
        output_final_state=True,
    )

    # cuda kernel：目标 dtype
    r_c = _to_torch(r, dtype, device)
    k_c = _to_torch(k, dtype, device)
    v_c = _to_torch(v, dtype, device)
    w_c = _to_torch(w, dtype, device)
    u_c = _to_torch(u, dtype, device)
    init_c = _to_torch(init, dtype, device)

    y_c, s_c = torch_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_c,
        initial_state=init_c,
        output_final_state=True,
    )

    if dtype == "float32":
        assert_allclose_with_stats(y_ref, y_c, "y", atol=1e-4, rtol=1e-4)
        assert_allclose_with_stats(s_ref, s_c, "final_state", atol=1e-4, rtol=1e-3)
    else:
        # bfloat16 以 fp32 ground truth 为参考，允许稍大误差
        assert_allclose_with_stats(y_ref, y_c, "y", atol=1.0, rtol=1e-1)
        assert_allclose_with_stats(s_ref, s_c, "final_state", atol=1.0, rtol=1e-1)


@pytest.mark.torch
@pytest.mark.parametrize("dtype", ["float32"])
def test_rwkv6_state_map(
    torch_op, native_op, sample_inputs, sample_shape, device, dtype
):
    B, T, H, N = sample_shape
    r, k, v, w, u, init = sample_inputs

    r_ref = _to_torch(r, "float32", device)
    k_ref = _to_torch(k, "float32", device)
    v_ref = _to_torch(v, "float32", device)
    w_ref = _to_torch(w, "float32", device)
    u_ref = _to_torch(u, "float32", device)
    init_map_ref = _to_torch(init[:1], "float32", device)

    y_ref, s_ref = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_map_ref,
        output_final_state=True,
        state_map=[0] * B,
    )

    r_c = _to_torch(r, dtype, device)
    k_c = _to_torch(k, dtype, device)
    v_c = _to_torch(v, dtype, device)
    w_c = _to_torch(w, dtype, device)
    u_c = _to_torch(u, dtype, device)
    init_map_c = _to_torch(init[:1], dtype, device)

    y_c, s_c = torch_op(
        r_c,
        k_c,
        v_c,
        w_c,
        u_c,
        initial_state=init_map_c,
        output_final_state=True,
        state_map=[0] * B,
    )

    assert_allclose_with_stats(y_ref, y_c, "y_state_map", atol=1e-4, rtol=1e-4)
    assert_allclose_with_stats(
        s_ref, s_c, "final_state_state_map", atol=1e-4, rtol=1e-3
    )


@pytest.mark.torch
@pytest.mark.slow
@pytest.mark.parametrize("dtype", ["float32"])
def test_rwkv6_backward(
    torch_op, native_op, sample_inputs, sample_shape, device, dtype
):
    B, T, H, N = sample_shape
    r, k, v, w, u, _ = sample_inputs

    def grads(op, r, k, v, w, u):
        r_t = r.clone().requires_grad_(True)
        k_t = k.clone().requires_grad_(True)
        v_t = v.clone().requires_grad_(True)
        w_t = w.clone().requires_grad_(True)
        u_t = u.clone().requires_grad_(True)
        y = op(r_t, k_t, v_t, w_t, u_t)
        loss = (y.float() ** 2).sum()
        loss.backward()
        return r_t.grad, k_t.grad, v_t.grad, w_t.grad, u_t.grad

    r_ref = _to_torch(r, "float32", device)
    k_ref = _to_torch(k, "float32", device)
    v_ref = _to_torch(v, "float32", device)
    w_ref = _to_torch(w, "float32", device)
    u_ref = _to_torch(u, "float32", device)

    g_ref = grads(native_op, r_ref, k_ref, v_ref, w_ref, u_ref)
    g_c = grads(torch_op, r_ref, k_ref, v_ref, w_ref, u_ref)

    for name, gr, gc in zip(["gr", "gk", "gv", "gw", "gu"], g_ref, g_c):
        assert_allclose_with_stats(gr, gc, name, atol=1e-2, rtol=1e-2)

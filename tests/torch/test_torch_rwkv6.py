"""RWKV-6 Torch CUDA kernel 数值测试。"""

import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.mark.torch
def test_rwkv6_forward_state(torch_op, native_op, sample_inputs, sample_shape, device):
    B, T, H, N = sample_shape
    r, k, v, w, u, init = sample_inputs

    r_ref = _to_torch(r, "bfloat16", device)
    k_ref = _to_torch(k, "bfloat16", device)
    v_ref = _to_torch(v, "bfloat16", device)
    w_ref = _to_torch(w, "bfloat16", device)
    u_ref = _to_torch(u, "bfloat16", device)
    init_ref = _to_torch(init, "bfloat16", device)

    y_ref, s_ref = native_op(
        r_ref,
        k_ref,
        v_ref,
        w_ref,
        u_ref,
        initial_state=init_ref,
        output_final_state=True,
    )

    y_c, s_c = torch_op(
        r_ref.clone(),
        k_ref.clone(),
        v_ref.clone(),
        w_ref.clone(),
        u_ref.clone(),
        initial_state=init_ref.clone(),
        output_final_state=True,
    )

    # native 参考实现内部按 fp32 计算，CUDA 为 bf16，统一 cast 到 bf16 后对比。
    assert_allclose_with_stats(y_ref.bfloat16(), y_c, "y", atol=1e-2, rtol=1e-2)
    assert_allclose_with_stats(
        s_ref.bfloat16(), s_c, "final_state", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
def test_rwkv6_state_map(torch_op, native_op, sample_inputs, sample_shape, device):
    B, T, H, N = sample_shape
    r, k, v, w, u, init = sample_inputs

    r_ref = _to_torch(r, "bfloat16", device)
    k_ref = _to_torch(k, "bfloat16", device)
    v_ref = _to_torch(v, "bfloat16", device)
    w_ref = _to_torch(w, "bfloat16", device)
    u_ref = _to_torch(u, "bfloat16", device)
    init_map_ref = _to_torch(init[:1], "bfloat16", device)

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

    y_c, s_c = torch_op(
        r_ref.clone(),
        k_ref.clone(),
        v_ref.clone(),
        w_ref.clone(),
        u_ref.clone(),
        initial_state=init_map_ref.clone(),
        output_final_state=True,
        state_map=[0] * B,
    )

    assert_allclose_with_stats(
        y_ref.bfloat16(), y_c, "y_state_map", atol=1e-2, rtol=1e-2
    )
    assert_allclose_with_stats(
        s_ref.bfloat16(), s_c, "final_state_state_map", atol=1e-2, rtol=1e-2
    )


@pytest.mark.torch
@pytest.mark.slow
def test_rwkv6_backward(torch_op, native_op, sample_inputs, sample_shape, device):
    B, T, H, N = sample_shape
    r, k, v, w, u, _ = sample_inputs

    def grads(op, r, k, v, w, u):
        r_t = r.clone().requires_grad_(True)
        k_t = k.clone().requires_grad_(True)
        v_t = v.clone().requires_grad_(True)
        w_t = w.clone().requires_grad_(True)
        u_t = u.clone().requires_grad_(True)
        y = op(r_t, k_t, v_t, w_t, u_t)
        loss = (y.float() ** 2).mean()
        loss.backward()
        return r_t.grad, k_t.grad, v_t.grad, w_t.grad, u_t.grad

    r_ref = _to_torch(r, "bfloat16", device)
    k_ref = _to_torch(k, "bfloat16", device)
    v_ref = _to_torch(v, "bfloat16", device)
    w_ref = _to_torch(w, "bfloat16", device)
    u_ref = _to_torch(u, "bfloat16", device)

    g_ref = grads(native_op, r_ref, k_ref, v_ref, w_ref, u_ref)
    g_c = grads(
        torch_op,
        r_ref.clone(),
        k_ref.clone(),
        v_ref.clone(),
        w_ref.clone(),
        u_ref.clone(),
    )

    for name, gr, gc in zip(["gr", "gk", "gv", "gw", "gu"], g_ref, g_c):
        assert_allclose_with_stats(gr, gc, name, atol=1e-2, rtol=1e-2)

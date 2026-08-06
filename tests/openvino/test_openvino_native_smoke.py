"""OpenVINO 后端 smoke 测试。"""

import numpy as np
import pytest
from keras import ops

pytestmark = pytest.mark.openvino

B, T, H, K = 2, 32, 3, 64


def _inputs():
    rng = np.random.default_rng(42)
    r = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, K), dtype=np.float32)
    z = rng.standard_normal((B, T, H, K), dtype=np.float32)
    norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
    a = -(z / norm).astype(np.float32)
    b = (z / norm).astype(np.float32)
    w = (-np.log1p(np.exp(rng.standard_normal((B, T, H, K)))) - 0.5).astype(np.float32)
    h0 = (rng.standard_normal((B, H, K, K)) * 0.1).astype(np.float32)
    return r, w, k, v, a, b, h0


def _to_numpy(x):
    return np.asarray(ops.convert_to_numpy(x), dtype=np.float32)


def _check(name, y, s, y_shape, s_shape):
    y = _to_numpy(y)
    assert y.shape == y_shape, f"{name} y shape {y.shape} != {y_shape}"
    assert not np.isnan(y).any(), f"{name} y contains NaN"
    if s is not None:
        s = _to_numpy(s)
        assert s.shape == s_shape, f"{name} state shape {s.shape} != {s_shape}"
        assert not np.isnan(s).any(), f"{name} state contains NaN"


@pytest.mark.openvino
def test_rwkv7_native_openvino():
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    r, w, k, v, a, b, h0 = _inputs()
    y, s = generalized_delta_rule(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        initial_state=h0,
        output_final_state=True,
    )
    _check("rwkv7", y, s, (B, T, H, K), (B, H, K, K))


@pytest.mark.openvino
def test_rwkv7_sane_native_openvino():
    from rwkv_ops.rwkv7_sane_kernel.native_keras_op import generalized_delta_rule_sane

    r, w, k, v, a, b, h0 = _inputs()
    rng = np.random.default_rng(1)
    tau = (np.log1p(np.exp(rng.standard_normal((B, T // 16, H)) + 7.0)) + 1.0).astype(
        np.float32
    )
    mask = np.ones((B, T // 16), dtype=np.float32)
    y, s = generalized_delta_rule_sane(
        r=r,
        w=w,
        k=k,
        v=v,
        a=a,
        b=b,
        tau=tau,
        mask=mask,
        initial_state=h0,
        output_final_state=True,
    )
    _check("rwkv7_sane", y, s, (B, T, H, K), (B, H, K, K))


@pytest.mark.openvino
def test_rwkv6_native_openvino():
    from rwkv_ops.rwkv6_kernel.native_keras_op import rwkv6

    rng = np.random.default_rng(2)
    C = H * K
    r = rng.standard_normal((B, T, C), dtype=np.float32)
    k = rng.standard_normal((B, T, C), dtype=np.float32)
    v = rng.standard_normal((B, T, C), dtype=np.float32)
    w = rng.standard_normal((B, T, C), dtype=np.float32) * 0.5
    u = (rng.standard_normal((H, K)) * 0.1).astype(np.float32)
    h0 = (rng.standard_normal((B, H, K, K)) * 0.1).astype(np.float32)
    y, s = rwkv6(r, k, v, w, u, initial_state=h0, output_final_state=True)
    _check("rwkv6", y, s, (B, T, C), (B, H, K, K))


@pytest.mark.openvino
def test_mhc_native_openvino():
    from rwkv_ops.mhc_kernel.native_op import mhc_post_op, mhc_pre_op_fused

    rng = np.random.default_rng(3)
    n, C = 4, 128
    x = rng.standard_normal((B, T, n, C), dtype=np.float32)
    h_res = rng.standard_normal((B, T, n, n), dtype=np.float32)
    h_pre = rng.standard_normal((B, T, n), dtype=np.float32)

    x_layer_in, H_res = mhc_pre_op_fused(x, h_res, h_pre, num_iters=20)
    x_layer_in = _to_numpy(x_layer_in)
    H_res = _to_numpy(H_res)
    assert x_layer_in.shape == (B, T, C)
    assert H_res.shape == (B, T, n, n)
    assert not np.isnan(x_layer_in).any()
    assert not np.isnan(H_res).any()

    layer_out = rng.standard_normal((B, T, C), dtype=np.float32)
    h_post = np.abs(rng.standard_normal((B, T, n), dtype=np.float32))
    x_next = _to_numpy(mhc_post_op(layer_out, x, h_post, H_res))
    assert x_next.shape == (B, T, n, C)
    assert not np.isnan(x_next).any()

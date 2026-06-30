"""
numpy 后端 native kernel 烟雾测试。
"""

import numpy as np
import pytest


@pytest.mark.numpy
def test_rwkv6_native_smoke(native_op, sample_inputs, sample_shape):
    r, k, v, w, u, init = sample_inputs
    B, T, H, N = sample_shape

    y, s = native_op(
        r,
        k,
        v,
        w,
        u,
        initial_state=init,
        output_final_state=True,
    )

    assert y.shape == (B, T, H * N)
    assert s.shape == (B, H, N, N)
    assert not np.isnan(y).any()
    assert not np.isnan(s).any()


@pytest.mark.numpy
def test_rwkv7_native_smoke(rwkv7_native_op, rwkv7_inputs, rwkv7_shape):
    B, T, H, K = rwkv7_shape
    y, s = rwkv7_native_op(
        r=rwkv7_inputs["r"],
        k=rwkv7_inputs["k"],
        v=rwkv7_inputs["v"],
        a=rwkv7_inputs["a"],
        b=rwkv7_inputs["b"],
        w=rwkv7_inputs["w"],
        initial_state=rwkv7_inputs["h0"],
        output_final_state=True,
    )

    assert y.shape == (B, T, H, K)
    assert s.shape == (B, H, K, K)
    assert not np.isnan(y).any()
    assert not np.isnan(s).any()

"""
RWKV-7 Torch CUDA 单步 RNN 接口数值测试。
"""

import numpy as np
import pytest
import torch

from tests.conftest import assert_allclose_with_stats


def _to_torch(arr, dtype, device):
    return torch.tensor(arr, dtype=getattr(torch, dtype), device=device)


@pytest.fixture
def single_step_inputs(rwkv7_inputs, rng):
    B, _, H, K = rwkv7_inputs["r"].shape
    return {
        name: rng.standard_normal((B, 1, H, K), dtype=np.float32)
        for name in ["r", "k", "v", "a", "b", "w"]
    } | {"h0": rng.standard_normal((B, H, K, K), dtype=np.float32)}


@pytest.mark.torch
def test_rwkv7_single_step_forward_state(
    rwkv7_rnn_op, rwkv7_native_op, single_step_inputs, device
):
    def make(tensors, dtype):
        return {
            name: _to_torch(tensors[name], dtype, device)
            for name in ["r", "k", "v", "a", "b", "w"]
        } | {"h0": _to_torch(tensors["h0"], "float32", device)}

    ref = make(single_step_inputs, "float32")
    tgt = make(single_step_inputs, "bfloat16")

    def call(op, t):
        return op(
            r=t["r"],
            k=t["k"],
            v=t["v"],
            a=t["a"],
            b=t["b"],
            w=t["w"],
            initial_state=t["h0"],
            output_final_state=True,
        )

    y_ref, s_ref = call(rwkv7_native_op, ref)
    y_tgt, s_tgt = call(rwkv7_rnn_op, tgt)

    assert_allclose_with_stats(y_ref, y_tgt, "y", atol=1.0, rtol=1e-1)
    assert_allclose_with_stats(s_ref, s_tgt, "final_state", atol=1.0, rtol=1e-1)

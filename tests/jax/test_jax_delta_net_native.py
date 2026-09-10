"""DeltaNet 原生 Keras 实现的数值等价性测试（JAX 后端）。

通过 importlib 直接加载模块文件，避免触发 rwkv_ops/__init__.py 中的
JAX CUDA 编译（本阶段只验证 native Keras 实现）。
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from keras import ops

from tests.conftest import assert_allclose_with_stats

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")

_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()


def _load_native_module(name, relpath):
    """直接加载模块文件，不执行 rwkv_ops/__init__.py。"""
    path = _PROJECT_ROOT / relpath
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_delta_net_chunk_mod = _load_native_module(
    "delta_net_chunk_native", "rwkv_ops/delta_net_chunk/native_keras_op.py"
)
_delta_net_recurrent_mod = _load_native_module(
    "delta_net_recurrent_native", "rwkv_ops/delta_net_recurrent/native_keras_op.py"
)
_gdn_chunk_mod = _load_native_module(
    "gdn_chunk_native", "rwkv_ops/gdn_chunk/native_keras_op.py"
)
_gdn_recurrent_mod = _load_native_module(
    "gdn_recurrent_native", "rwkv_ops/gdn_recurrent/native_keras_op.py"
)

delta_net_chunk = _delta_net_chunk_mod.delta_net_chunk
delta_net_recurrent = _delta_net_recurrent_mod.delta_net_recurrent
delta_net_reference = _delta_net_recurrent_mod.delta_net_reference
delta_net_recurrent_inference = _delta_net_recurrent_mod.delta_net_recurrent_inference
delta_net_recurrent_single_step = (
    _delta_net_recurrent_mod.delta_net_recurrent_single_step
)
gated_delta_net_chunk = _gdn_chunk_mod.gated_delta_net_chunk
gated_delta_net_recurrent = _gdn_recurrent_mod.gated_delta_net_recurrent


@pytest.mark.jax
def test_delta_net_chunk_matches_reference(delta_net_inputs):
    """chunkwise 实现应与最简 recurrent reference 数值一致。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_ref, state_ref = delta_net_reference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_chunk, "chunk vs reference output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_chunk, "chunk vs reference state", atol=1e-2, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_recurrent_matches_reference(delta_net_inputs):
    """fori_loop recurrent 实现应与最简 reference 一致。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = delta_net_reference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_rec, "recurrent vs reference output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_rec, "recurrent vs reference state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_chunk_matches_recurrent(delta_net_inputs):
    """chunkwise 与 recurrent 两种实现应等价。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_chunk, "chunk vs recurrent output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_chunk, "chunk vs recurrent state", atol=1e-2, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_no_final_state(delta_net_inputs):
    """output_final_state=False 时不应返回 state。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta = delta_net_inputs["beta"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, output_final_state=False, chunk_size=64
    )
    out_rec, state_rec = delta_net_recurrent(q, k, v, beta, output_final_state=False)

    assert state_chunk is None
    assert state_rec is None
    assert_allclose_with_stats(
        out_rec, out_chunk, "no-state chunk vs recurrent output", atol=1e-2, rtol=1e-3
    )


@pytest.mark.jax
@pytest.mark.slow
def test_delta_net_different_chunk_size(delta_net_inputs):
    """chunk_size 改变时 chunkwise 结果应保持不变。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_c64, state_c64 = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )
    out_c32, state_c32 = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=32
    )

    assert_allclose_with_stats(
        out_c64, out_c32, "chunk_size=64 vs 32 output", atol=1e-2, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_c64, state_c32, "chunk_size=64 vs 32 state", atol=1e-2, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_arbitrary_length(delta_net_inputs):
    """recurrent / reference 支持任意长度（非 chunk 整除）。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    q, k, v = q[:, :37], k[:, :37], v[:, :37]
    beta = beta[:, :37]

    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_ref, state_ref = delta_net_reference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_ref, out_rec, "arbitrary length output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_ref, state_rec, "arbitrary length state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_single_step_matches_recurrent(delta_net_inputs):
    """单步 RNN 逐步跑完整序列应与 recurrent 对齐。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    steps = 16
    state = h0
    outs = []
    for t in range(steps):
        out_t, state = delta_net_recurrent_single_step(
            q[:, t],
            k[:, t],
            v[:, t],
            beta[:, t],
            initial_state=state,
            output_final_state=True,
        )
        outs.append(out_t)
    out_step = ops.stack(outs, axis=1)

    out_rec, state_rec = delta_net_recurrent(
        q[:, :steps],
        k[:, :steps],
        v[:, :steps],
        beta[:, :steps],
        initial_state=h0,
        output_final_state=True,
    )

    assert_allclose_with_stats(
        out_rec, out_step, "single step vs recurrent output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state, "single step vs recurrent state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_inference_matches_recurrent(delta_net_inputs):
    """推理封装应与 recurrent 训练实现输出一致。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]

    out_inf, state_inf = delta_net_recurrent_inference(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_rec, out_inf, "inference vs recurrent output", atol=1e-5, rtol=1e-3
    )
    assert_allclose_with_stats(
        state_rec, state_inf, "inference vs recurrent state", atol=1e-5, rtol=1e-3
    )


@pytest.mark.jax
def test_delta_net_zero_g_recurrent_matches_gdn(delta_net_inputs):
    """g 取全 0 时 Gated DeltaNet recurrent 应退化为 DeltaNet recurrent。"""
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]
    zeros_g = np.zeros(q.shape[:-1], dtype=np.float32)

    out_gdn, state_gdn = gated_delta_net_recurrent(
        q, k, v, zeros_g, beta, initial_state=h0, output_final_state=True
    )
    out_rec, state_rec = delta_net_recurrent(
        q, k, v, beta, initial_state=h0, output_final_state=True
    )

    assert_allclose_with_stats(
        out_gdn,
        out_rec,
        "zero-g gdn recurrent vs delta net output",
        atol=1e-5,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_gdn,
        state_rec,
        "zero-g gdn recurrent vs delta net state",
        atol=1e-5,
        rtol=1e-3,
    )


@pytest.mark.jax
def test_delta_net_zero_g_chunk_matches_gdn(delta_net_inputs):
    """g 取全 0 时 Gated DeltaNet chunkwise 应退化为 DeltaNet chunkwise。

    输入 cast bf16，模拟训练实际精度。
    """
    q, k, v = delta_net_inputs["q"], delta_net_inputs["k"], delta_net_inputs["v"]
    beta, h0 = delta_net_inputs["beta"], delta_net_inputs["h0"]
    zeros_g = np.zeros(q.shape[:-1], dtype=np.float32)
    q = ops.cast(q, "bfloat16")
    k = ops.cast(k, "bfloat16")
    v = ops.cast(v, "bfloat16")

    out_gdn, state_gdn = gated_delta_net_chunk(
        q,
        k,
        v,
        zeros_g,
        beta,
        initial_state=h0,
        output_final_state=True,
        chunk_size=64,
    )
    out_chunk, state_chunk = delta_net_chunk(
        q, k, v, beta, initial_state=h0, output_final_state=True, chunk_size=64
    )

    assert_allclose_with_stats(
        out_gdn,
        out_chunk,
        "zero-g gdn chunk vs delta net output",
        atol=1e-2,
        rtol=1e-3,
    )
    assert_allclose_with_stats(
        state_gdn,
        state_chunk,
        "zero-g gdn chunk vs delta net state",
        atol=1e-2,
        rtol=1e-3,
    )

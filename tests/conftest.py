"""
公共 pytest 配置与工具函数。

注意：
- 本文件不 import torch/jax，也不 import rwkv_ops/keras，
  避免在收集阶段锁定 Keras 后端。
- torch/jax 相关的 import 请放到 tests/torch/ 或 tests/jax/ 各自的
  conftest.py / 测试函数里。
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 让测试能导入项目根目录的 clean_build_artifacts
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    config.addinivalue_line("markers", "torch: require PyTorch")
    config.addinivalue_line("markers", "jax: require JAX")
    config.addinivalue_line("markers", "numpy: require Keras numpy backend")
    config.addinivalue_line("markers", "tensorflow: require TensorFlow")
    config.addinivalue_line("markers", "openvino: require OpenVINO")
    config.addinivalue_line("markers", "slow: slow tests (e.g. compilation)")


def pytest_sessionfinish(session, exitstatus):
    """整个 pytest 会话结束后自动清理编译产物。"""
    # 避免在 pytest 收集阶段就 import rwkv_ops/keras
    from clean_build_artifacts import clean_all

    print("\n[pytest] 测试结束，自动清理编译产物...")
    clean_all()


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_shape():
    """默认 (B, T, H, N)。"""
    return 2, 16, 6, 64


@pytest.fixture(scope="session")
def sample_inputs(rng, sample_shape):
    B, T, H, N = sample_shape
    C = H * N
    r = rng.standard_normal((B, T, C), dtype=np.float32)
    k = rng.standard_normal((B, T, C), dtype=np.float32)
    v = rng.standard_normal((B, T, C), dtype=np.float32)
    w = rng.standard_normal((B, T, C), dtype=np.float32) * 0.5
    u = rng.standard_normal((H, N), dtype=np.float32) * 0.1
    init_state = rng.standard_normal((B, H, N, N), dtype=np.float32) * 0.1
    return r, k, v, w, u, init_state


@pytest.fixture(scope="session")
def rwkv7_shape():
    """RWKV-7 默认 (B, T, H, K)。"""
    return 5, 128, 6, 64


@pytest.fixture(scope="session")
def rwkv7_inputs(rng, rwkv7_shape):
    B, T, H, K = rwkv7_shape
    r = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, K), dtype=np.float32)

    z = rng.standard_normal((B, T, H, K), dtype=np.float32)
    norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
    a = -(z / norm).astype(np.float32)
    b = (z / norm).astype(np.float32)

    w_raw = rng.standard_normal((B, T, H, K), dtype=np.float32)
    # softplus(w) = log(1 + exp(w))
    w = -np.log1p(np.exp(w_raw)) - 0.5
    w = w.astype(np.float32)

    h0 = rng.standard_normal((B, H, K, K), dtype=np.float32)
    return {"r": r, "k": k, "v": v, "a": a, "b": b, "w": w, "h0": h0}


@pytest.fixture(scope="session")
def rwkv7_sn_inputs(rng, rwkv7_inputs):
    """
    RWKV-7-SN 测试输入：在 rwkv7_inputs 基础上补充 tau。
    tau = softplus(x) + 1，x 随机，均值约 4 以让 tau 接近 100（近似恒等映射）。
    """
    B, T, H, _ = rwkv7_inputs["r"].shape
    x = rng.standard_normal((B, T // 16, H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    return {**rwkv7_inputs, "tau": tau.astype(np.float32)}


@pytest.fixture(scope="session")
def mhc_shape():
    """mHC 默认 (B, T, n, C)。"""
    return 64, 64, 4, 512


@pytest.fixture(scope="session")
def mhc_pre_inputs(rng, mhc_shape):
    B, T, n, C = mhc_shape
    x = rng.standard_normal((B, T, n, C), dtype=np.float32)
    h_res = rng.standard_normal((B, T, n, n), dtype=np.float32)
    h_pre = rng.standard_normal((B, T, n), dtype=np.float32)
    return {"x": x, "h_res": h_res, "h_pre": h_pre}


@pytest.fixture(scope="session")
def mhc_post_inputs(rng, mhc_shape):
    B, T, n, C = mhc_shape
    layer_out = rng.standard_normal((B, T, C), dtype=np.float32)
    x_expanded = rng.standard_normal((B, T, n, C), dtype=np.float32)
    h_post = rng.standard_normal((B, T, n), dtype=np.float32)
    H_res = rng.standard_normal((B, T, n, n), dtype=np.float32)
    return {
        "layer_out": layer_out,
        "x_expanded": x_expanded,
        "h_post": h_post,
        "H_res": H_res,
    }


def to_numpy(x):
    """统一把后端张量转成 numpy。"""
    if hasattr(x, "detach"):
        x = x.detach().cpu()
        import torch

        if x.dtype == torch.bfloat16:
            x = x.float()
        return x.numpy()
    if hasattr(x, "numpy"):
        return np.array(x)
    return np.array(x)


def assert_allclose_with_stats(
    ref,
    tgt,
    name,
    atol=1e-3,
    rtol=1e-3,
):
    """
    数值比较工具：打印最大/平均差异、完全一致数、近似一致数，
    然后调用 np.testing.assert_allclose。

    参考 rwkv7 测试代码中的“完全一致”语义：
    - exact_match：转换到 float32 后逐元素 == 的数量
    - close_match：|ref - tgt| <= atol 的元素数量
    """
    ref_arr = to_numpy(ref).astype(np.float32)
    tgt_arr = to_numpy(tgt).astype(np.float32)

    diff = np.abs(ref_arr - tgt_arr)
    total = ref_arr.size
    exact_match = int(np.sum(ref_arr == tgt_arr))
    close_match = int(np.sum(diff <= atol))
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))

    print(
        f"[{name}] total={total}, exact_match={exact_match} "
        f"({exact_match / total:.2%}), close_match={close_match} "
        f"({close_match / total:.2%}), max_diff={max_diff:.3e}, mean_diff={mean_diff:.3e}"
    )

    np.testing.assert_allclose(
        ref_arr,
        tgt_arr,
        atol=atol,
        rtol=rtol,
        err_msg=f"{name} 数值不一致",
    )

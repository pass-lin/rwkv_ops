"""NumPy 后端测试的 session 级配置。"""

import os

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
os.environ.setdefault("KERAS_BACKEND", "numpy")

import pytest

pytest.importorskip("keras")


def _check_numpy_backend():
    """跳过非 numpy 后端环境。"""
    import keras

    backend = keras.config.backend()
    if backend != "numpy":
        pytest.skip(f"Keras numpy backend unavailable (current={backend})")


@pytest.fixture(scope="session")
def native_op(sample_shape):
    """RWKV-6 NumPy native 算子。

    Args:
        sample_shape: tuple, (B, T, H, N)。

    Returns:
        Callable: HEAD_SIZE=N 的 RWKV-6 native kernel。
    """
    _check_numpy_backend()
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_native_op(rwkv7_shape):
    """RWKV-7 NumPy native 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 native kernel。
    """
    _check_numpy_backend()
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="native")
    return op

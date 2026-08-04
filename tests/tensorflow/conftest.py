"""TensorFlow 后端测试的 session 级配置。"""

import os

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
os.environ.setdefault("KERAS_BACKEND", "tensorflow")

import pytest

pytest.importorskip("tensorflow")


def _ensure_keras_tf():
    """跳过非 tensorflow 后端或 Keras 不可用的环境。

    Returns:
        keras: 已验证为 tensorflow 后端的 keras 模块。
    """
    try:
        import keras
    except Exception as e:
        pytest.skip(f"Keras tensorflow backend 不可用: {e}")
    if keras.config.backend() != "tensorflow":
        pytest.skip(
            f"当前 Keras 后端不是 tensorflow (current={keras.config.backend()})"
        )
    return keras


@pytest.fixture(scope="session")
def native_op(sample_shape):
    """RWKV-6 TensorFlow native 算子。

    Args:
        sample_shape: tuple, (B, T, H, N)。

    Returns:
        Callable: HEAD_SIZE=N 的 RWKV-6 native kernel。
    """
    _ensure_keras_tf()
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_native_op(rwkv7_shape):
    """RWKV-7 TensorFlow native 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 native kernel。
    """
    _ensure_keras_tf()
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="native")
    return op

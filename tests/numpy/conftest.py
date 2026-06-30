"""
numpy 后端测试的 session 级配置。
"""

import os

os.environ.setdefault("KERAS_BACKEND", "numpy")

import pytest

pytest.importorskip("keras")


def _check_numpy_backend():
    import keras

    backend = keras.config.backend()
    if backend != "numpy":
        pytest.skip(f"Keras numpy backend unavailable (current={backend})")


@pytest.fixture(scope="session")
def native_op(sample_shape):
    _check_numpy_backend()
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_native_op(rwkv7_shape):
    _check_numpy_backend()
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="native")
    return op

"""Torch 后端测试的 session 级配置。"""

import os
import shutil

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest

pytest.importorskip("torch")


def pytest_sessionstart(session):
    """session 开始时清除 Triton cache，避免跨 session 的 kernel 复用导致数值错误。"""
    triton_cache = os.path.expanduser("~/.triton/cache")
    if os.path.isdir(triton_cache):
        shutil.rmtree(triton_cache)


@pytest.fixture(scope="session")
def device():
    """返回可用的 Torch 设备。

    Returns:
        str: "cuda:0" 或 "cpu"。
    """
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def torch_op(sample_shape):
    """RWKV-6 Torch CUDA 算子。

    Args:
        sample_shape: tuple, (B, T, H, N)。

    Returns:
        Callable: HEAD_SIZE=N 的 RWKV-6 CUDA kernel。
    """
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def native_op(sample_shape):
    """RWKV-6 native Keras 算子。

    Args:
        sample_shape: tuple, (B, T, H, N)。

    Returns:
        Callable: HEAD_SIZE=N 的 RWKV-6 native kernel。
    """
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_op(rwkv7_shape):
    """RWKV-7 Torch CUDA 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 CUDA 训练 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_inference_op(rwkv7_shape):
    """RWKV-7 Torch CUDA 推理算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 CUDA 推理 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    _, op = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_native_op(rwkv7_shape):
    """RWKV-7 native Keras 参考算子。

    Returns:
        Callable: RWKV-7 native_keras_op.generalized_delta_rule。
    """
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


@pytest.fixture(scope="session")
def rwkv7_rnn_op(rwkv7_shape):
    """RWKV-7 Torch CUDA 单步 RNN 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 单步 CUDA kernel。
    """
    from rwkv_ops import get_rnn_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sane_op(rwkv7_shape):
    """RWKV-7-SANE Torch CUDA 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE CUDA 训练 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sane_inference_op(rwkv7_shape):
    """RWKV-7-SANE Torch CUDA 推理算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE CUDA 推理 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    _, op = get_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sane_native_op(rwkv7_shape):
    """RWKV-7-SANE native Keras 参考算子。

    Returns:
        Callable: RWKV-7-SANE native_keras_op.generalized_delta_rule_sane。
    """
    from rwkv_ops.rwkv7_sane_kernel.native_keras_op import generalized_delta_rule_sane

    return generalized_delta_rule_sane


@pytest.fixture(scope="session")
def rwkv7_sane_rnn_native_op(rwkv7_shape):
    """RWKV-7-SANE native Keras 单步 RNN 参考算子。

    Returns:
        Callable: RWKV-7-SANE native_keras_op.generalized_delta_rule_sane_single_step。
    """
    from rwkv_ops.rwkv7_sane_kernel.native_keras_op import (
        generalized_delta_rule_sane_single_step,
    )

    return generalized_delta_rule_sane_single_step


@pytest.fixture(scope="session")
def rwkv7_sane_rnn_op(rwkv7_shape):
    """RWKV-7-SANE Torch CUDA 单步 RNN 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE 单步 CUDA kernel。
    """
    from rwkv_ops import get_rnn_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sane_triton_op(rwkv7_shape):
    """RWKV-7-SANE Torch Triton 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE Triton 训练 kernel。
    """
    pytest.importorskip("triton")
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op

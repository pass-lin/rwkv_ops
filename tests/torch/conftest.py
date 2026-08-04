"""Torch 后端测试的 session 级配置。"""

import os

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest

pytest.importorskip("torch")


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
def rwkv7_sn_op(rwkv7_shape):
    """RWKV-7-SN Torch CUDA 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SN CUDA 训练 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sn_inference_op(rwkv7_shape):
    """RWKV-7-SN Torch CUDA 推理算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SN CUDA 推理 kernel。
    """
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    _, op = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sn_native_op(rwkv7_shape):
    """RWKV-7-SN native Keras 参考算子。

    Returns:
        Callable: RWKV-7-SN native_keras_op.generalized_delta_rule_sn。
    """
    from rwkv_ops.rwkv7_sn_kernel.native_keras_op import generalized_delta_rule_sn

    return generalized_delta_rule_sn


@pytest.fixture(scope="session")
def rwkv7_sn_rnn_native_op(rwkv7_shape):
    """RWKV-7-SN native Keras 单步 RNN 参考算子。

    Returns:
        Callable: RWKV-7-SN native_keras_op.generalized_delta_rule_sn_single_step。
    """
    from rwkv_ops.rwkv7_sn_kernel.native_keras_op import (
        generalized_delta_rule_sn_single_step,
    )

    return generalized_delta_rule_sn_single_step


@pytest.fixture(scope="session")
def rwkv7_sn_rnn_op(rwkv7_shape):
    """RWKV-7-SN Torch CUDA 单步 RNN 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SN 单步 CUDA kernel。
    """
    from rwkv_ops import get_rnn_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sn_triton_op(rwkv7_shape):
    """RWKV-7-SN Torch Triton 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SN Triton 训练 kernel。
    """
    pytest.importorskip("triton")
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op

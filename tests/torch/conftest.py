"""
torch 后端测试的 session 级配置。

注意：必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
"""

import os

os.environ.setdefault("KERAS_BACKEND", "torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest

pytest.importorskip("torch")


@pytest.fixture(scope="session")
def device():
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="session")
def torch_op(sample_shape):
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def native_op(sample_shape):
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_inference_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    _, op = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_native_op(rwkv7_shape):
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


@pytest.fixture(scope="session")
def rwkv7_rnn_op(rwkv7_shape):
    from rwkv_ops import get_rnn_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sn_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sn_inference_op(rwkv7_shape):
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    _, op = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")
    return op


@pytest.fixture(scope="session")
def rwkv7_sn_native_op(rwkv7_shape):
    from rwkv_ops.rwkv7_sn_kernel.native_keras_op import generalized_delta_rule_sn

    return generalized_delta_rule_sn


@pytest.fixture(scope="session")
def rwkv7_sn_rnn_native_op(rwkv7_shape):
    from rwkv_ops.rwkv7_sn_kernel.native_keras_op import (
        generalized_delta_rule_sn_single_step,
    )

    return generalized_delta_rule_sn_single_step


@pytest.fixture(scope="session")
def rwkv7_sn_rnn_op(rwkv7_shape):
    from rwkv_ops import get_rnn_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sn_triton_op(rwkv7_shape):
    pytest.importorskip("triton")
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op

"""
JAX 后端测试的 session 级配置。

注意：必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。
"""

import os
import shutil
import subprocess
import sys


def _set_cuda_host_compiler():
    """如果系统默认 GCC 太新，自动选一个 CUDA 兼容的 host compiler。"""

    def _major_version(name):
        path = shutil.which(name)
        if not path:
            return None
        try:
            out = subprocess.run(
                [path, "-dumpversion"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return int(out.stdout.strip().split(".")[0])
        except Exception:
            return None

    # 用户已显式指定编译器时，尊重用户选择
    if os.environ.get("CC") and os.environ.get("CXX") and os.environ.get("CUDAHOSTCXX"):
        return

    # 默认 gcc 已存在且版本 <= 13，一般可直接使用
    default_major = _major_version("gcc")
    if default_major is not None and default_major <= 13:
        return

    candidates = [
        ("x86_64-conda-linux-gnu-gcc", "x86_64-conda-linux-gnu-g++"),
        ("gcc-13", "g++-13"),
        ("gcc-12", "g++-12"),
        ("gcc-11", "g++-11"),
    ]
    for gcc_name, gpp_name in candidates:
        gcc_path = shutil.which(gcc_name)
        gpp_path = shutil.which(gpp_name)
        if gcc_path and gpp_path:
            os.environ["CC"] = gcc_path
            os.environ["CXX"] = gpp_path
            os.environ["CUDAHOSTCXX"] = gpp_path
            print(
                f"[tests/jax] 自动选择 CUDA host compiler: CC={gcc_path}, CXX={gpp_path}",
                file=sys.stderr,
            )
            return

    print(
        "[tests/jax] Warning: 系统默认 GCC 版本过高（>13）且未找到兼容的 GCC，"
        "JAX CUDA 内核编译可能会失败。请手动安装 gcc-13/g++-13 并设置 "
        "CC/CXX/CUDAHOSTCXX。",
        file=sys.stderr,
    )


_set_cuda_host_compiler()

os.environ.setdefault("KERAS_BACKEND", "jax")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest  # noqa: E402

pytest.importorskip("jax")
pytest.importorskip("jax.numpy")


@pytest.fixture(scope="session")
def jax_op(sample_shape):
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def native_op(sample_shape):
    from rwkv_ops import get_rwkv6_kernel

    _, _, _, N = sample_shape
    return get_rwkv6_kernel(HEAD_SIZE=N, KERNEL_TYPE="native")


@pytest.fixture(scope="session")
def rwkv7_jax_op(rwkv7_shape):
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
def rwkv7_native_op():
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


@pytest.fixture(scope="session")
def rwkv7_rnn_op(rwkv7_shape):
    from rwkv_ops import get_rnn_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sn_jax_op(rwkv7_shape):
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
def rwkv7_sn_native_op():
    from rwkv_ops.rwkv7_sn_kernel.native_keras_op import generalized_delta_rule_sn

    return generalized_delta_rule_sn


@pytest.fixture(scope="session")
def rwkv7_sn_rnn_native_op():
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
def rwkv7_sn_jax_triton_op(rwkv7_shape):
    pytest.importorskip("triton")
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="triton")
    return op


@pytest.fixture(scope="session")
def rwkv7_jax_pallas_op(rwkv7_shape):
    pytest.importorskip("jax.experimental.pallas")
    import jax

    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("pallas 后端仅用于 GPU/TPU")
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="pallas")
    return op


@pytest.fixture(scope="session")
def rwkv7_sn_jax_pallas_op(rwkv7_shape):
    pytest.importorskip("jax.experimental.pallas")
    import jax

    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("pallas 后端仅用于 GPU/TPU")
    from rwkv_ops import get_generalized_delta_rule_sn

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sn(HEAD_SIZE=K, KERNEL_TYPE="pallas")
    return op

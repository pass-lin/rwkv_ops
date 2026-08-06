"""JAX 后端测试的 session 级配置。"""

import os
import shutil
import subprocess
import sys

# 必须在 import keras / rwkv_ops 之前设定 KERAS_BACKEND。


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

    # 系统默认 gcc 版本 <= 13 时视为与 CUDA 兼容，直接沿用。
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
    """RWKV-6 JAX CUDA 算子。

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
def rwkv7_jax_op(rwkv7_shape):
    """RWKV-7 JAX CUDA 训练算子。

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
    """RWKV-7 JAX CUDA 推理算子。

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
def rwkv7_native_op():
    """RWKV-7 native Keras 参考算子。

    Returns:
        Callable: RWKV-7 native_keras_op.generalized_delta_rule。
    """
    from rwkv_ops.rwkv7_kernel.native_keras_op import generalized_delta_rule

    return generalized_delta_rule


@pytest.fixture(scope="session")
def rwkv7_rnn_op(rwkv7_shape):
    """RWKV-7 JAX CUDA 单步 RNN 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 单步 CUDA kernel。
    """
    from rwkv_ops import get_rnn_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sane_jax_op(rwkv7_shape):
    """RWKV-7-SANE JAX CUDA 训练算子。

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
    """RWKV-7-SANE JAX CUDA 推理算子。

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
def rwkv7_sane_native_op():
    """RWKV-7-SANE native Keras 参考算子。

    Returns:
        Callable: RWKV-7-SANE native_keras_op.generalized_delta_rule_sane。
    """
    from rwkv_ops.rwkv7_sane_kernel.native_keras_op import generalized_delta_rule_sane

    return generalized_delta_rule_sane


@pytest.fixture(scope="session")
def rwkv7_sane_rnn_native_op():
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
    """RWKV-7-SANE JAX CUDA 单步 RNN 算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE 单步 CUDA kernel。
    """
    from rwkv_ops import get_rnn_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    return get_rnn_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="cuda")


@pytest.fixture(scope="session")
def rwkv7_sane_jax_triton_op(rwkv7_shape):
    """RWKV-7-SANE JAX Triton 训练算子。

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


@pytest.fixture(scope="session")
def rwkv7_jax_pallas_op(rwkv7_shape):
    """RWKV-7 JAX Pallas 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7 Pallas 训练 kernel。
            非 GPU/TPU 环境会 pytest.skip。
    """
    pytest.importorskip("jax.experimental.pallas")
    import jax

    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("pallas 后端仅用于 GPU/TPU")
    from rwkv_ops import get_generalized_delta_rule

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule(HEAD_SIZE=K, KERNEL_TYPE="native")
    return op


@pytest.fixture(scope="session")
def rwkv7_sane_jax_pallas_op(rwkv7_shape):
    """RWKV-7-SANE JAX Pallas 训练算子。

    Args:
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        Callable: HEAD_SIZE=K 的 RWKV-7-SANE Pallas 训练 kernel。
            非 GPU/TPU 环境会 pytest.skip。
    """
    pytest.importorskip("jax.experimental.pallas")
    import jax

    if jax.devices()[0].platform not in ("gpu", "tpu"):
        pytest.skip("pallas 后端仅用于 GPU/TPU")
    from rwkv_ops import get_generalized_delta_rule_sane

    _, _, _, K = rwkv7_shape
    op, _ = get_generalized_delta_rule_sane(HEAD_SIZE=K, KERNEL_TYPE="native")
    return op

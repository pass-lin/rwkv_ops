"""公共 pytest 配置与共享工具函数。"""

import sys
from pathlib import Path

import numpy as np
import pytest

# 根 conftest 不在收集阶段导入 torch/jax/keras/rwkv_ops，避免锁定 Keras 后端。

# 让测试能导入项目根目录的 clean_build_artifacts。
_PROJECT_ROOT = Path(__file__).parent.parent.resolve()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def pytest_configure(config):
    """注册 pytest 自定义 markers。"""
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
    """返回固定种子 42 的 NumPy 随机数生成器。

    Returns:
        np.random.Generator: 固定种子生成器。
    """
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def sample_shape():
    """RWKV-6 测试默认形状。

    Returns:
        tuple: (B, T, H, N) = (2, 16, 6, 64)。
    """
    return 2, 16, 6, 64


@pytest.fixture(scope="session")
def sample_inputs(rng, sample_shape):
    """RWKV-6 测试输入张量。

    Args:
        rng: np.random.Generator，随机数生成器。
        sample_shape: tuple, (B, T, H, N)。

    Returns:
        tuple: (r, k, v, w, u, init_state)，均为 float32 numpy 数组。
            r/k/v/w: [B, T, H*N]。
            u: [H, N]。
            init_state: [B, H, N, N]。
    """
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
    """RWKV-7 测试默认形状。

    Returns:
        tuple: (B, T, H, K) = (5, 128, 6, 64)。
    """
    return 5, 128, 6, 64


@pytest.fixture(scope="session")
def rwkv7_inputs(rng, rwkv7_shape):
    """RWKV-7 测试输入张量。

    a 与 b 取同一向量的正负单位向量，w 用 -softplus - 0.5 保证稳定衰减。

    Args:
        rng: np.random.Generator，随机数生成器。
        rwkv7_shape: tuple, (B, T, H, K)。

    Returns:
        dict: 包含 r/k/v/a/b/w/h0，均为 float32 numpy 数组。
            r/k/v/a/b/w: [B, T, H, K]。
            h0: [B, H, K, K]。
    """
    B, T, H, K = rwkv7_shape
    r = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, K), dtype=np.float32)

    z = rng.standard_normal((B, T, H, K), dtype=np.float32)
    norm = np.linalg.norm(z, axis=-1, keepdims=True) + 1e-12
    a = -(z / norm).astype(np.float32)
    b = (z / norm).astype(np.float32)

    w_raw = rng.standard_normal((B, T, H, K), dtype=np.float32)
    # softplus(w) = log(1 + exp(w))。
    w = -np.log1p(np.exp(w_raw)) - 0.5
    w = w.astype(np.float32)

    h0 = rng.standard_normal((B, H, K, K), dtype=np.float32)
    return {"r": r, "k": k, "v": v, "a": a, "b": b, "w": w, "h0": h0}


@pytest.fixture(scope="session")
def rwkv7_sane_inputs(rng, rwkv7_inputs):
    """RWKV-7-SANE 测试输入，在 rwkv7_inputs 基础上补充 tau。

    tau = softplus(x) + 1，x 均值约 7 使 tau 接近 1000，近似恒等映射。

    Args:
        rng: np.random.Generator，随机数生成器。
        rwkv7_inputs: dict, RWKV-7 基础输入。

    Returns:
        dict: 包含 rwkv7_inputs 全部字段与 tau。
            tau: [B, T//16, H], float32。
    """
    B, T, H, _ = rwkv7_inputs["r"].shape
    x = rng.standard_normal((B, T // 16, H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    return {**rwkv7_inputs, "tau": tau.astype(np.float32)}


@pytest.fixture(scope="session")
def gdn_shape():
    """GDN 测试默认形状。

    Returns:
        tuple: (B, T, H, K, V) = (2, 128, 4, 64, 128)。
    """
    return 2, 128, 4, 64, 128


@pytest.fixture(scope="session")
def gdn_inputs(rng, gdn_shape):
    """GDN 测试输入张量。

    q/k 不做 L2 norm，由算子内部处理。g 取负 softplus 保证稳定衰减；
    beta 取 sigmoid，使其落在 (0,1)。

    Args:
        rng: np.random.Generator，随机数生成器。
        gdn_shape: tuple, (B, T, H, K, V)。

    Returns:
        dict: 包含 q/k/v/g/beta/h0，均为 float32 numpy 数组。
            q/k: [B, T, H, K]。
            v: [B, T, H, V]。
            g/beta: [B, T, H]。
            h0: [B, H, K, V]。
    """
    B, T, H, K, V = gdn_shape
    q = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, V), dtype=np.float32)

    g_raw = rng.standard_normal((B, T, H), dtype=np.float32)
    g = -np.log1p(np.exp(g_raw)) - 0.5
    g = g.astype(np.float32)

    beta_raw = rng.standard_normal((B, T, H), dtype=np.float32)
    beta = (1.0 / (1.0 + np.exp(-beta_raw))).astype(np.float32)

    h0 = rng.standard_normal((B, H, K, V), dtype=np.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "h0": h0,
    }


@pytest.fixture(scope="session")
def gdn_sane_inputs(rng, gdn_inputs):
    """GDN-SANE 测试输入，在 gdn_inputs 基础上补充 tau 与 mask。

    tau = softplus(x) + 1，x 均值约 7 使 tau 接近 1000，近似恒等映射；
    mask 随机取 0/1，用于验证 padding chunk 行为。

    Args:
        rng: np.random.Generator，随机数生成器。
        gdn_inputs: dict, GDN 基础输入。

    Returns:
        dict: 包含 gdn_inputs 全部字段与 tau/mask。
            tau: [B, T//16, H], float32。
            mask: [B, T//16], float32。
    """
    B, T, H, _ = gdn_inputs["q"].shape
    chunk_num = T // 16
    x = rng.standard_normal((B, max(chunk_num, 1), H), dtype=np.float32) * 0.5 + 7.0
    tau = np.log1p(np.exp(x)) + 1.0
    mask = rng.integers(0, 2, (B, max(chunk_num, 1))).astype(np.float32)
    return {**gdn_inputs, "tau": tau.astype(np.float32), "mask": mask}


@pytest.fixture(scope="session")
def delta_net_shape():
    """DeltaNet 测试默认形状。

    Returns:
        tuple: (B, T, H, K, V) = (2, 128, 4, 64, 128)。
    """
    return 2, 128, 4, 64, 128


@pytest.fixture(scope="session")
def delta_net_inputs(rng, delta_net_shape):
    """DeltaNet 测试输入张量。

    q/k 不做 L2 norm，由算子内部处理。beta 取 sigmoid，使其落在 (0,1)。

    Args:
        rng: np.random.Generator，随机数生成器。
        delta_net_shape: tuple, (B, T, H, K, V)。

    Returns:
        dict: 包含 q/k/v/beta/h0，均为 float32 numpy 数组。
            q/k: [B, T, H, K]。
            v: [B, T, H, V]。
            beta: [B, T, H]。
            h0: [B, H, K, V]。
    """
    B, T, H, K, V = delta_net_shape
    q = rng.standard_normal((B, T, H, K), dtype=np.float32)
    k = rng.standard_normal((B, T, H, K), dtype=np.float32)
    v = rng.standard_normal((B, T, H, V), dtype=np.float32)

    beta_raw = rng.standard_normal((B, T, H), dtype=np.float32)
    beta = (1.0 / (1.0 + np.exp(-beta_raw))).astype(np.float32)

    h0 = rng.standard_normal((B, H, K, V), dtype=np.float32) * 0.1
    return {
        "q": q,
        "k": k,
        "v": v,
        "beta": beta,
        "h0": h0,
    }


@pytest.fixture(scope="session")
def mhc_shape():
    """mHC 测试默认形状。

    Returns:
        tuple: (B, T, n, C) = (64, 64, 4, 512)。
    """
    return 64, 64, 4, 512


@pytest.fixture(scope="session")
def mhc_pre_inputs(rng, mhc_shape):
    """mHC pre-op 测试输入张量。

    Args:
        rng: np.random.Generator，随机数生成器。
        mhc_shape: tuple, (B, T, n, C)。

    Returns:
        dict: 包含 x/h_res/h_pre，均为 float32 numpy 数组。
            x: [B, T, n, C]。
            h_res: [B, T, n, n]。
            h_pre: [B, T, n]。
    """
    B, T, n, C = mhc_shape
    x = rng.standard_normal((B, T, n, C), dtype=np.float32)
    h_res = rng.standard_normal((B, T, n, n), dtype=np.float32)
    h_pre = rng.standard_normal((B, T, n), dtype=np.float32)
    return {"x": x, "h_res": h_res, "h_pre": h_pre}


@pytest.fixture(scope="session")
def mhc_post_inputs(rng, mhc_shape):
    """mHC post-op 测试输入张量。

    Args:
        rng: np.random.Generator，随机数生成器。
        mhc_shape: tuple, (B, T, n, C)。

    Returns:
        dict: 包含 layer_out/x_expanded/h_post/H_res，均为 float32 numpy 数组。
            layer_out: [B, T, C]。
            x_expanded: [B, T, n, C]。
            h_post: [B, T, n]。
            H_res: [B, T, n, n]。
    """
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
    """把后端张量统一转成 numpy 数组。

    Args:
        x: torch.Tensor / jax.Array / tf.Tensor / np.ndarray 等后端张量。

    Returns:
        np.ndarray: float32 numpy 数组。bfloat16 会先做 float() 转换。
    """
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
    """数值比较工具：打印统计信息后调用 np.testing.assert_allclose。

    Args:
        ref: 参考张量，任意后端，会被 to_numpy 转成 float32。
        tgt: 目标张量，任意后端，会被 to_numpy 转成 float32。
        name: str, 当前比较项名称，用于日志与错误信息。
        atol: float, 绝对容差。
        rtol: float, 相对容差。

    Returns:
        None。断言失败时抛出 AssertionError。
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

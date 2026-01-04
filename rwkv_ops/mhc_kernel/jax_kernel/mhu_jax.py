"""
JAX FFI 版 MHC 算子库
- Sinkhorn Knopp: 实现双拟随机矩阵投影
- 接口与 native_keras_op.py 完全一致
"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import numpy as np  # <--- 添加numpy导入
from typing import Tuple
import jax
import jax.numpy as jnp
from jax.ad_checkpoint import checkpoint_policies as cp

# 当前目录
_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


# ---------- 延迟编译机制 ----------
def _ensure_compiled() -> pathlib.Path:
    """首次调用时编译CUDA扩展"""
    _SO_PATH = _CURRENT_DIR / "mhu.so"

    if _SO_PATH.exists():
        return _SO_PATH

    print(f"[mhu_jax] 首次使用 - 正在编译CUDA内核...")

    # 构建目录
    _BUILD_DIR = _CURRENT_DIR / "build"
    build_dir = _BUILD_DIR
    build_dir.mkdir(exist_ok=True)

    # 获取XLA头文件路径
    xla_include_dir = jax.ffi.include_dir()
    if not xla_include_dir:
        raise RuntimeError("jax.ffi.include_dir() 返回空，请检查JAX版本>=0.4.31")

    # CMake配置
    cmake_args = [
        "cmake",
        "-S",
        str(_CURRENT_DIR),
        "-B",
        str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release",
        f"-DXLA_INCLUDE_DIR={xla_include_dir}",
        "-DCMAKE_CUDA_FLAGS=-O3 --use_fast_math -std=c++17",
    ]

    try:
        subprocess.check_call(cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"], cwd=build_dir)
        subprocess.check_call(["cmake", "--install", str(build_dir)], cwd=build_dir)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"CMake编译失败: {e}")

    if not _SO_PATH.exists():
        files = list(_CURRENT_DIR.glob("*"))
        raise RuntimeError(
            f"编译失败 - 无法在 {_SO_PATH} 找到共享库\n"
            f"当前目录内容: {[f.name for f in files]}"
        )

    print(f"[mhu_jax] 编译完成 - 输出: {_SO_PATH}")
    return _SO_PATH


# ---------- FFI目标注册 ----------
_LIB = ctypes.CDLL(_ensure_compiled())
jax.ffi.register_ffi_target(
    "sinkhorn_fwd", jax.ffi.pycapsule(_LIB.SinkhornFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "sinkhorn_bwd", jax.ffi.pycapsule(_LIB.SinkhornBwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "rmsnorm_fwd", jax.ffi.pycapsule(_LIB.RMSNormFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "rmsnorm_bwd", jax.ffi.pycapsule(_LIB.RMSNormBwd), platform="CUDA"
)


def _normalize_shape(x: jnp.ndarray, expected_ndim: int, name: str) -> jnp.ndarray:
    """确保数组维度正确"""
    if x.ndim != expected_ndim:
        raise ValueError(f"{name}期望{expected_ndim}维张量，但输入为{x.ndim}维")
    return x


# ---------- 核心实现 ----------
def _sinkhorn_ffi_fwd(
    inp: jnp.ndarray,
    num_iters: np.int32,  # <--- 使用np.int32
    eps: np.float32,  # <--- 使用np.float32
) -> jnp.ndarray:
    """内部FFI前向调用"""
    inp = inp.astype(jnp.float32)
    out_type = jax.ShapeDtypeStruct(inp.shape, jnp.float32)

    # 直接传递，已经是numpy 32位标量
    out = jax.ffi.ffi_call("sinkhorn_fwd", out_type, vmap_method="broadcast_all")(
        inp, num_iters=num_iters, eps=eps
    )

    return out


def _sinkhorn_ffi_bwd(
    grad: jnp.ndarray,
    out_fwd: jnp.ndarray,
    inp: jnp.ndarray,
    num_iters: np.int32,  # <--- 使用np.int32
    eps: np.float32,  # <--- 使用np.float32
) -> jnp.ndarray:
    """内部FFI反向调用"""
    grad = grad.astype(jnp.float32)
    out_fwd = out_fwd.astype(jnp.float32)
    inp = inp.astype(jnp.float32)

    d_inp_type = jax.ShapeDtypeStruct(inp.shape, jnp.float32)

    d_inp = jax.ffi.ffi_call("sinkhorn_bwd", d_inp_type, vmap_method="broadcast_all")(
        grad, out_fwd, inp, num_iters=num_iters, eps=eps
    )

    return d_inp


# 关键修复：在闭包创建时就将参数转换为numpy 32位类型
def _create_sinkhorn_kernel(num_iters: int, eps: float):
    """创建带有静态参数的sinkhorn kernel"""

    # 在闭包外部转换为numpy 32位标量
    num_iters_static = np.int32(num_iters)  # <--- 确保32位
    eps_static = np.float32(eps)  # <--- 确保32位

    @jax.custom_vjp
    def _kernel(inp: jnp.ndarray) -> jnp.ndarray:
        return _sinkhorn_ffi_fwd(inp, num_iters_static, eps_static)

    def _fwd(inp: jnp.ndarray):
        out = _sinkhorn_ffi_fwd(inp, num_iters_static, eps_static)
        return out, (out, inp)

    def _bwd(saved_vals: Tuple[jnp.ndarray, jnp.ndarray], grad: jnp.ndarray):
        out_fwd, inp = saved_vals
        d_inp = _sinkhorn_ffi_bwd(grad, out_fwd, inp, num_iters_static, eps_static)
        return (d_inp,)

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# ---------- 公共API ----------
def sinkhorn_knopp(
    inp: jnp.ndarray, num_iters: int = 20, eps: float = 1e-8
) -> jnp.ndarray:
    """
    JAX FFI版Sinkhorn Knopp算子

    参数:
        inp: [B, T, N, N] 输入矩阵（任意dtype）
        num_iters: 迭代次数（必须是编译期常量）
        eps: 防止除零的小常数（必须是编译期常量）

    返回:
        [B, T, N, N] 双拟随机矩阵，dtype与输入一致
    """
    # 类型和形状检查
    inp = _normalize_shape(inp, 4, "sinkhorn_knopp")
    original_dtype = inp.dtype

    # 关键修复：在创建kernel前转换为numpy 32位类型
    kernel = _create_sinkhorn_kernel(np.int32(num_iters), np.float32(eps))

    # 使用checkpoint防止重计算
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    # 执行计算
    result = checkpointed_kernel(inp)

    # 转换回原始dtype
    return result.astype(original_dtype)


def _rmsnorm_ffi_fwd(inp: jnp.ndarray, eps: np.float32) -> jnp.ndarray:
    """内部FFI前向调用"""
    # 确保bf16和连续性
    inp = inp.astype(jnp.bfloat16)
    out_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)

    out = jax.ffi.ffi_call("rmsnorm_fwd", out_type, vmap_method="broadcast_all")(
        inp, eps=eps
    )

    return out


def _rmsnorm_ffi_bwd(
    grad: jnp.ndarray, inp: jnp.ndarray, eps: np.float32
) -> jnp.ndarray:
    """内部FFI反向调用"""
    grad = grad.astype(jnp.bfloat16)
    inp = inp.astype(jnp.bfloat16)

    dx_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)

    dx = jax.ffi.ffi_call("rmsnorm_bwd", dx_type, vmap_method="broadcast_all")(
        grad, inp, eps=eps
    )

    return dx


def _create_rmsnorm_kernel(eps: float):
    """创建带有静态eps的rmsnorm kernel"""
    eps_static = np.float32(eps)  # 编译期常量

    @jax.custom_vjp
    def _kernel(inp: jnp.ndarray) -> jnp.ndarray:
        return _rmsnorm_ffi_fwd(inp, eps_static)

    def _fwd(inp: jnp.ndarray):
        out = _rmsnorm_ffi_fwd(inp, eps_static)
        return out, (inp,)  # 保存输入用于反向

    def _bwd(saved_vals: Tuple[jnp.ndarray,], grad: jnp.ndarray):
        (inp,) = saved_vals
        dx = _rmsnorm_ffi_bwd(grad, inp, eps_static)
        return (dx,)

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# ---------- 公共API ----------
def rmsnorm(inp: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    """
    JAX FFI版RMSNorm算子

    参数:
        inp: [..., C] 输入张量（任意dtype）
        eps: 防止除零的小常数

    返回:
        [..., C] 归一化结果，dtype与输入一致
    """
    # 形状检查（至少2维）
    if inp.ndim < 2:
        raise ValueError(f"RMSNorm需要至少2维输入，但得到{inp.ndim}维")

    original_dtype = inp.dtype
    original_shape = inp.shape

    # 展平到2D: [N, C]
    N = inp.shape[0]
    C = inp.shape[-1]
    inp_2d = inp.reshape(-1, C)

    # 创建kernel并执行
    kernel = _create_rmsnorm_kernel(eps)
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    result_2d = checkpointed_kernel(inp_2d)

    # 恢复形状
    return result_2d.astype(original_dtype).reshape(original_shape)

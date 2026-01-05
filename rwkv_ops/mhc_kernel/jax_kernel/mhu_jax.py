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
jax.ffi.register_ffi_target(
    "stream_mix_fwd", jax.ffi.pycapsule(_LIB.StreamMixFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "stream_mix_bwd", jax.ffi.pycapsule(_LIB.StreamMixBwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "stream_aggregate_fwd", jax.ffi.pycapsule(_LIB.StreamAggregateFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "stream_aggregate_bwd", jax.ffi.pycapsule(_LIB.StreamAggregateBwd), platform="CUDA"
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
    inp = jnp.asarray(inp, "float32")
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
    inp = inp.astype(jnp.bfloat16)
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


# ---------- Stream Mix 核心实现 ----------
def _stream_mix_fwd(inp: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """内部FFI前向调用"""
    # 强制类型转换

    out_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)

    out = jax.ffi.ffi_call("stream_mix_fwd", out_type, vmap_method="broadcast_all")(
        inp, M
    )

    return out


def _stream_mix_bwd(
    grad: jnp.ndarray, inp: jnp.ndarray, M: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """内部FFI反向调用"""
    # 关键修复：梯度必须是 fp32，不是 bf16
    grad = grad.astype(jnp.float32)  # 从 jnp.bfloat16 改为 jnp.float32
    inp = inp.astype(jnp.bfloat16)
    M = M.astype(jnp.float32)

    d_inp_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)
    d_M_type = jax.ShapeDtypeStruct(M.shape, jnp.float32)

    d_inp, d_M = jax.ffi.ffi_call(
        "stream_mix_bwd", (d_inp_type, d_M_type), vmap_method="broadcast_all"
    )(grad, inp, M)  # 现在 grad 是 F32，匹配 FFI 签名

    return d_inp, d_M


def _create_stream_mix_kernel():
    """创建Stream Mix kernel（无静态参数）"""

    @jax.custom_vjp
    def _kernel(inp: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
        return _stream_mix_fwd(inp, M)

    def _fwd(inp: jnp.ndarray, M: jnp.ndarray):
        out = _stream_mix_fwd(inp, M)
        # 保存输入用于反向
        return out, (inp, M)

    def _bwd(saved_vals: Tuple[jnp.ndarray, jnp.ndarray], grad: jnp.ndarray):
        inp, M = saved_vals
        d_inp, d_M = _stream_mix_bwd(grad, inp, M)
        # 返回两个梯度，对应forward的两个输入
        return d_inp, d_M

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# ---------- 公共API ----------
def stream_mix(inp: jnp.ndarray, M: jnp.ndarray) -> jnp.ndarray:
    """
    JAX FFI版Stream Mix算子

    参数:
        inp: [B, T, n, C] 输入张量（支持任意dtype，内部转bf16）
        M: [B, T, n, n] 权重矩阵（支持任意dtype，内部转fp32）

    返回:
        [B, T, n, C] 混合结果，dtype与inp一致
    """
    # 形状检查
    if inp.ndim != 4:
        raise ValueError(f"Stream Mix需要4维输入，但得到{inp.ndim}维")
    if M.ndim != 4:
        raise ValueError(f"Stream Mix权重需要4维，但得到{M.ndim}维")
    if inp.shape[:3] != M.shape[:3]:
        raise ValueError(f"Batch/Time/Stream维度不匹配: inp{inp.shape}, M{M.shape}")

    original_dtype = inp.dtype
    inp = inp.astype(jnp.bfloat16)
    M = M.astype(jnp.float32)
    # 创建并执行kernel
    kernel = _create_stream_mix_kernel()
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    result = checkpointed_kernel(inp, M)

    return result.astype(original_dtype)


# ---------- Stream Aggregate 核心实现 ----------
def _stream_aggregate_ffi_fwd(
    inp: jnp.ndarray, H_pre: jnp.ndarray, per_token: bool
) -> jnp.ndarray:
    """内部FFI前向调用"""
    # 强制类型转换: BF16 输入, FP32 权重

    # 输出形状: [B, T, C]
    B, T, n, C = inp.shape
    out_shape = (B, T, C)
    out_type = jax.ShapeDtypeStruct(out_shape, jnp.bfloat16)

    out = jax.ffi.ffi_call(
        "stream_aggregate_fwd", out_type, vmap_method="broadcast_all"
    )(inp, H_pre, per_token=per_token)

    return out


def _stream_aggregate_ffi_bwd(
    grad: jnp.ndarray, inp: jnp.ndarray, H_pre: jnp.ndarray, per_token: bool
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """内部FFI反向调用"""
    # 关键：梯度用 float32 进行高精度规约
    grad_f32 = grad.astype(jnp.float32)
    inp_bf16 = inp.astype(jnp.bfloat16)
    H_f32 = H_pre.astype(jnp.float32)

    B, T, n, C = inp.shape
    # 输出梯度形状
    d_inp_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)
    # d_H_pre 形状需要与 H_pre 保持一致
    d_H_shape = H_pre.shape  # 可能是 [B,T,n] 或 [n]
    d_H_type = jax.ShapeDtypeStruct(d_H_shape, jnp.float32)

    d_inp, d_H = jax.ffi.ffi_call(
        "stream_aggregate_bwd", (d_inp_type, d_H_type), vmap_method="broadcast_all"
    )(grad_f32, inp_bf16, H_f32, per_token=per_token)

    return d_inp, d_H


def _create_stream_aggregate_kernel(per_token: bool):
    """创建Stream Aggregate kernel（支持两种权重模式）"""

    @jax.custom_vjp
    def _kernel(inp: jnp.ndarray, H_pre: jnp.ndarray) -> jnp.ndarray:
        return _stream_aggregate_ffi_fwd(inp, H_pre, per_token)

    def _fwd(inp: jnp.ndarray, H_pre: jnp.ndarray):
        out = _stream_aggregate_ffi_fwd(inp, H_pre, per_token)
        # 保存输入用于反向
        return out, (inp, H_pre)

    def _bwd(saved_vals: Tuple[jnp.ndarray, jnp.ndarray], grad: jnp.ndarray):
        inp, H_pre = saved_vals
        d_inp, d_H_pre = _stream_aggregate_ffi_bwd(grad, inp, H_pre, per_token)
        # 返回两个梯度，对应forward的两个输入
        return d_inp, d_H_pre

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# ---------- 公共API ----------
def stream_aggregate(inp: jnp.ndarray, H_pre: jnp.ndarray) -> jnp.ndarray:
    """
    JAX FFI版Stream Aggregate算子

    功能: Out = sum(inp * H_pre, axis=-2)
    高精度策略: 在float32空间完成乘法和累加，最后转回输入dtype

    参数:
        inp: [B, T, n, C] 输入张量（任意dtype）
        H_pre: [B, T, n] 或 [n] 权重张量（任意dtype）
               - [B, T, n]: per-token权重，每个token有独立权重
               - [n]: per-stream权重，所有token共享权重

    返回:
        [B, T, C] 聚合结果，dtype与inp一致
    """
    # 形状检查
    if inp.ndim != 4:
        raise ValueError(f"Stream Aggregate需要4维输入，但得到{inp.ndim}维")
    if H_pre.ndim not in [1, 3]:
        raise ValueError(f"H_pre必须是1维或3维，但得到{H_pre.ndim}维")

    B, T, n, C = inp.shape
    if H_pre.ndim == 1:
        if H_pre.shape[0] != n:
            raise ValueError(f"全局权重H_pre的形状{n}与输入流数{n}不匹配")
        per_token = False
    else:  # H_pre.ndim == 3
        if H_pre.shape != (B, T, n):
            raise ValueError(
                f"Per-token权重H_pre形状{H_pre.shape}与输入形状{(B, T, n)}不匹配"
            )
        per_token = True

    original_dtype = inp.dtype

    inp = inp.astype(jnp.bfloat16)
    H_pre = H_pre.astype(jnp.float32)
    # 创建并执行kernel
    kernel = _create_stream_aggregate_kernel(per_token)
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    result = checkpointed_kernel(inp, H_pre)

    return result.astype(original_dtype)


# 1. 在 register_ffi_target 部分追加
jax.ffi.register_ffi_target(
    "stream_distribute_fwd",
    jax.ffi.pycapsule(_LIB.StreamDistributeFwd),
    platform="CUDA",
)
jax.ffi.register_ffi_target(
    "stream_distribute_bwd",
    jax.ffi.pycapsule(_LIB.StreamDistributeBwd),
    platform="CUDA",
)


# 2. 实现核心逻辑
def _stream_distribute_ffi_fwd(inp: jnp.ndarray, H_post: jnp.ndarray) -> jnp.ndarray:
    """内部FFI前向调用"""
    B, T, C = inp.shape
    n = H_post.shape[-1]
    out_type = jax.ShapeDtypeStruct((B, T, n, C), jnp.bfloat16)

    # 接口对齐：inp用bf16, H_post用f32
    return jax.ffi.ffi_call(
        "stream_distribute_fwd", out_type, vmap_method="broadcast_all"
    )(inp.astype(jnp.bfloat16), H_post.astype(jnp.float32))


def _stream_distribute_ffi_bwd(
    grad: jnp.ndarray, inp: jnp.ndarray, H_post: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """内部FFI反向调用"""
    # 强制将梯度转为 bf16 匹配 FFI 签名，内部会转 float 计算
    grad_bf16 = grad.astype(jnp.bfloat16)
    inp_bf16 = inp.astype(jnp.bfloat16)
    H_f32 = H_post.astype(jnp.float32)

    d_inp_type = jax.ShapeDtypeStruct(inp.shape, jnp.bfloat16)
    d_H_type = jax.ShapeDtypeStruct(H_post.shape, jnp.float32)

    return jax.ffi.ffi_call(
        "stream_distribute_bwd", (d_inp_type, d_H_type), vmap_method="broadcast_all"
    )(grad_bf16, inp_bf16, H_f32)


def _create_stream_distribute_kernel():
    """创建 Stream Distribute kernel"""

    @jax.custom_vjp
    def _kernel(inp: jnp.ndarray, H_post: jnp.ndarray) -> jnp.ndarray:
        return _stream_distribute_ffi_fwd(inp, H_post)

    def _fwd(inp: jnp.ndarray, H_post: jnp.ndarray):
        out = _stream_distribute_ffi_fwd(inp, H_post)
        return out, (inp, H_post)

    def _bwd(saved_vals: Tuple[jnp.ndarray, jnp.ndarray], grad: jnp.ndarray):
        inp, H_post = saved_vals
        d_inp, d_H_post = _stream_distribute_ffi_bwd(grad, inp, H_post)
        return d_inp, d_H_post

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# 3. 公共 API
def stream_distribute(inp: jnp.ndarray, H_post: jnp.ndarray) -> jnp.ndarray:
    """
    JAX FFI 版 Stream Distribute 算子 (1 -> n)
    功能: Out = inp[:, :, None, :] * H_post[:, :, :, None]

    参数:
        inp: [B, T, C] 输入张量
        H_post: [B, T, n] 权重张量
    返回:
        [B, T, n, C] 分发后的多流张量，dtype 与 inp 一致
    """
    # 形状检查
    if inp.ndim != 3 or H_post.ndim != 3:
        raise ValueError(
            f"stream_distribute 要求输入均为 3 维，得到 {inp.ndim} 和 {H_post.ndim}"
        )

    original_dtype = inp.dtype
    inp = inp.astype(jnp.bfloat16)
    H_post = H_post.astype(jnp.float32)
    # 创建并执行 kernel
    kernel = _create_stream_distribute_kernel()

    # 统一设置永不重计算 (checkpoint policy)
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    result = checkpointed_kernel(inp, H_post)

    # 类型还原，避免梯度计算中出现不必要的类型漂移
    return result.astype(original_dtype)


jax.ffi.register_ffi_target(
    "mhc_post_op_fwd", jax.ffi.pycapsule(_LIB.MhcPostOpFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "mhc_post_op_bwd", jax.ffi.pycapsule(_LIB.MhcPostOpBwd), platform="CUDA"
)


# 2. 内部 FFI 调用封装
def _mhc_post_op_ffi_fwd(layer_out, x_expanded, H_post, H_res):
    out_type = jax.ShapeDtypeStruct(x_expanded.shape, jnp.bfloat16)
    return jax.ffi.ffi_call("mhc_post_op_fwd", out_type, vmap_method="broadcast_all")(
        layer_out.astype(jnp.bfloat16),
        x_expanded.astype(jnp.bfloat16),
        H_post.astype(jnp.float32),
        H_res.astype(jnp.float32),
    )


def _mhc_post_op_ffi_bwd(grad, layer_out, x_expanded, H_post, H_res):
    d_lo_type = jax.ShapeDtypeStruct(layer_out.shape, jnp.bfloat16)
    d_xe_type = jax.ShapeDtypeStruct(x_expanded.shape, jnp.bfloat16)
    d_hp_type = jax.ShapeDtypeStruct(H_post.shape, jnp.float32)
    d_hr_type = jax.ShapeDtypeStruct(H_res.shape, jnp.float32)

    return jax.ffi.ffi_call(
        "mhc_post_op_bwd",
        (d_lo_type, d_xe_type, d_hp_type, d_hr_type),
        vmap_method="broadcast_all",
    )(
        grad.astype(jnp.bfloat16),
        layer_out.astype(jnp.bfloat16),
        x_expanded.astype(jnp.bfloat16),
        H_post.astype(jnp.float32),
        H_res.astype(jnp.float32),
    )


def _create_mhc_post_op_kernel():
    @jax.custom_vjp
    def _kernel(layer_out, x_expanded, H_post, H_res):
        return _mhc_post_op_ffi_fwd(layer_out, x_expanded, H_post, H_res)

    def _fwd(layer_out, x_expanded, H_post, H_res):
        out = _mhc_post_op_ffi_fwd(layer_out, x_expanded, H_post, H_res)
        return out, (layer_out, x_expanded, H_post, H_res)

    def _bwd(saved_vals, grad):
        lo, xe, hp, hr = saved_vals
        return _mhc_post_op_ffi_bwd(grad, lo, xe, hp, hr)

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# 3. 公共 API
def mhc_post_op(
    layer_out: jnp.ndarray,
    x_expanded: jnp.ndarray,
    H_post: jnp.ndarray,
    H_res: jnp.ndarray,
) -> jnp.ndarray:
    """
    mHC 后处理融合算子 (Fused Res-Mix + Post-Distribute)
    实现: x_next = (H_res @ x_expanded) + (layer_out * H_post)

    参数:
        layer_out: [B, T, C] 核心层输出
        x_expanded: [B, T, n, C] 之前的扩展流
        H_post: [B, T, n] 分发权重
        H_res: [B, T, n, n] 混合矩阵
    返回:
        [B, T, n, C] 更新后的流，dtype 与 x_expanded 一致
    """
    original_dtype = x_expanded.dtype
    layer_out = layer_out.astype(jnp.bfloat16)
    x_expanded = x_expanded.astype(jnp.bfloat16)
    H_post = H_post.astype(jnp.float32)
    H_res = H_res.astype(jnp.float32)
    kernel = _create_mhc_post_op_kernel()
    # 强制 checkpoint 以节省显存并避免冗余重计算
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    result = checkpointed_kernel(layer_out, x_expanded, H_post, H_res)
    return result.astype(original_dtype)
# ---------- 在 register_ffi_target 部分追加 ----------
jax.ffi.register_ffi_target(
    "mhc_pre_op_fwd", jax.ffi.pycapsule(_LIB.MhcPreOpFwd), platform="CUDA"
)
jax.ffi.register_ffi_target(
    "mhc_pre_op_bwd", jax.ffi.pycapsule(_LIB.MhcPreOpBwd), platform="CUDA"
)


# ---------- MHC Pre-Op 核心实现 ----------
def _mhc_pre_op_ffi_fwd(x_expanded, h_pre_raw, h_post_raw, h_res_raw, sinkhorn_iters, eps):
    """内部FFI前向调用"""
    # x_expanded: [B, T, n, C]
    # h_pre_raw: [B, T, n]
    # h_post_raw: [B, T, n]
    # h_res_raw: [B, T, n, n] 或 [B, T, n*n]
    
    # 展平 h_res_raw 以匹配 C++ 接口 (期望 [B, T, n*n])
    if h_res_raw.ndim == 4:
        h_res_raw_flat = h_res_raw.reshape(h_res_raw.shape[0], h_res_raw.shape[1], -1)
    else:
        h_res_raw_flat = h_res_raw

    B, T, n, C = x_expanded.shape
    
    # 定义输出形状 (H_res 由 C++ 返回展平格式)
    out_type_x_layer_in = jax.ShapeDtypeStruct((B, T, C), jnp.bfloat16)
    out_type_H = jax.ShapeDtypeStruct((B, T, n), jnp.float32)
    out_type_H_res_flat = jax.ShapeDtypeStruct((B, T, n * n), jnp.float32)

    # 调用 FFI 前向 (返回 4 个张量)
    x_layer_in, H_pre, H_post, H_res_flat = jax.ffi.ffi_call(
        "mhc_pre_op_fwd",
        (out_type_x_layer_in, out_type_H, out_type_H, out_type_H_res_flat),
        vmap_method="broadcast_all",
    )(
        x_expanded.astype(jnp.bfloat16),
        h_pre_raw.astype(jnp.float32),
        h_post_raw.astype(jnp.float32),
        h_res_raw_flat.astype(jnp.float32),
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    
    # 将 H_res 重塑回 4D
    H_res = H_res_flat.reshape(B, T, n, n)
    return x_layer_in, H_pre, H_post, H_res


def _mhc_pre_op_ffi_bwd(grad_layer_in, grad_H_post, grad_H_res, x_expanded, H_pre, H_post, H_res_out, h_res_raw, sinkhorn_iters, eps):
    """内部FFI反向调用"""
    # 展平梯度与残差以匹配 C++ 接口
    grad_H_res_flat = grad_H_res.reshape(grad_H_res.shape[0], grad_H_res.shape[1], -1)
    H_res_out_flat = H_res_out.reshape(H_res_out.shape[0], H_res_out.shape[1], -1)
    
    # h_res_raw 是来自用户的原始输入，可能为 4D
    if h_res_raw.ndim == 4:
        h_res_raw_flat = h_res_raw.reshape(h_res_raw.shape[0], h_res_raw.shape[1], -1)
    else:
        h_res_raw_flat = h_res_raw

    B, T, n, C = x_expanded.shape
    d_x_shape = (B, T, n, C)
    d_h_shape = (B, T, n)
    d_h_res_shape = h_res_raw_flat.shape  # 展平格式

    d_x_type = jax.ShapeDtypeStruct(d_x_shape, jnp.bfloat16)
    d_h_type = jax.ShapeDtypeStruct(d_h_shape, jnp.float32)
    d_h_res_type = jax.ShapeDtypeStruct(d_h_res_shape, jnp.float32)

    # 调用 FFI 反向 (返回 4 个梯度)
    d_x_expanded, d_h_pre_raw, d_h_post_raw, d_h_res_raw_flat = jax.ffi.ffi_call(
        "mhc_pre_op_bwd",
        (d_x_type, d_h_type, d_h_type, d_h_res_type),
        vmap_method="broadcast_all",
    )(
        grad_layer_in.astype(jnp.bfloat16),
        grad_H_post.astype(jnp.float32),
        grad_H_res_flat.astype(jnp.float32),
        x_expanded.astype(jnp.bfloat16),
        H_pre.astype(jnp.float32),
        H_post.astype(jnp.float32),
        H_res_out_flat.astype(jnp.float32),
        h_res_raw_flat.astype(jnp.float32),
        sinkhorn_iters=sinkhorn_iters,
        eps=eps,
    )
    
    # 将 d_h_res_raw 重塑回 4D (若原始输入是 4D)
    if h_res_raw.ndim == 4:
        d_h_res_raw = d_h_res_raw_flat.reshape(B, T, n, n)
    else:
        d_h_res_raw = d_h_res_raw_flat

    return d_x_expanded, d_h_pre_raw, d_h_post_raw, d_h_res_raw


def _create_mhc_pre_op_kernel(sinkhorn_iters: int, eps: float):
    """创建MHC Pre-Op kernel（静态参数固化）"""
    # 在闭包外部固化 Sinkhorn 参数为 NumPy 32 位标量
    sinkhorn_iters_static = np.int32(sinkhorn_iters)
    eps_static = np.float32(eps)

    @jax.custom_vjp
    def _kernel(x_expanded, h_pre_raw, h_post_raw, h_res_raw):
        # 调用 FFI 前向，返回 4 个张量
        x_layer_in, H_pre, H_post, H_res = _mhc_pre_op_ffi_fwd(
            x_expanded, h_pre_raw, h_post_raw, h_res_raw, sinkhorn_iters_static, eps_static
        )
        # 只返回 3 个张量给用户 (PyTorch 版本不返回 H_pre)
        return x_layer_in, H_post, H_res

    def _fwd(x_expanded, h_pre_raw, h_post_raw, h_res_raw):
        # 调用前向并保存残差
        x_layer_in, H_pre, H_post, H_res = _mhc_pre_op_ffi_fwd(
            x_expanded, h_pre_raw, h_post_raw, h_res_raw, sinkhorn_iters_static, eps_static
        )
        # 残差包含反向所需的所有张量 (包括 H_pre, H_post, H_res)
        return (x_layer_in, H_post, H_res), (x_expanded, H_pre, H_post, H_res, h_res_raw)

    def _bwd(residuals, grads):
        # 解包残差
        x_expanded, H_pre, H_post, H_res, h_res_raw = residuals
        # 解包输出梯度 (3 个梯度，对应前向的 3 个输出)
        # grads: (grad_x_layer_in, grad_H_post, grad_H_res)
        grad_layer_in = grads[0]
        grad_H_post = grads[1]
        grad_H_res = grads[2]

        # 调用 FFI 反向
        return _mhc_pre_op_ffi_bwd(
            grad_layer_in,
            grad_H_post,
            grad_H_res,
            x_expanded,
            H_pre,
            H_post,
            H_res,
            h_res_raw,
            sinkhorn_iters_static,
            eps_static,
        )

    _kernel.defvjp(_fwd, _bwd)
    return _kernel


# ---------- 公共 API ----------
def mhc_pre_op(
    x_expanded: jnp.ndarray,
    h_pre_raw: jnp.ndarray,
    h_post_raw: jnp.ndarray,
    h_res_raw: jnp.ndarray,
    num_iters: int = 20,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    mHC 前处理融合算子 (Fused Aggregate + Sigmoid + Sinkhorn)
    
    功能:
        1. H_pre = sigmoid(h_pre_raw)
        2. H_post = 2 * sigmoid(h_post_raw)
        3. H_res = Sinkhorn(h_res_raw)
        4. x_layer_in = sum(H_pre * x_expanded, axis=-2)

    参数:
        x_expanded: [B, T, n, C] 输入张量（任意dtype）
        h_pre_raw: [B, T, n] 预权重原始值（任意dtype）
        h_post_raw: [B, T, n] 后权重原始值（任意dtype）
        h_res_raw: [B, T, n, n] 残差权重原始值（任意dtype）
        num_iters: Sinkhorn 迭代次数（编译期常量）
        eps: 防止除零的小常数（编译期常量）

    返回:
        tuple: (x_layer_in [B, T, C], H_post [B, T, n], H_res [B, T, n, n])
               dtype 与输入一致
    """
    # 形状检查
    if x_expanded.ndim != 4:
        raise ValueError(f"x_expanded 需要 4 维，但得到 {x_expanded.ndim}")
    B, T, n, C = x_expanded.shape
    
    expected_h_shape = (B, T, n)
    if h_pre_raw.shape != expected_h_shape:
        raise ValueError(f"h_pre_raw 形状 {h_pre_raw.shape} 与期望 {expected_h_shape} 不匹配")
    if h_post_raw.shape != expected_h_shape:
        raise ValueError(f"h_post_raw 形状 {h_post_raw.shape} 与期望 {expected_h_shape} 不匹配")
    
    # h_res_raw 可以是 4D [B,T,n,n] 或 3D [B,T,n*n]
    if h_res_raw.ndim == 4:
        if h_res_raw.shape != (B, T, n, n):
            raise ValueError(f"h_res_raw 4D 形状 {h_res_raw.shape} 与期望 {(B, T, n, n)} 不匹配")
    elif h_res_raw.ndim == 3:
        if h_res_raw.shape != (B, T, n * n):
            raise ValueError(f"h_res_raw 3D 形状 {h_res_raw.shape} 与期望 {(B, T, n * n)} 不匹配")
    else:
        raise ValueError(f"h_res_raw 必须是 3 维或 4 维，但得到 {h_res_raw.ndim}")

    original_dtype_x = x_expanded.dtype
    original_dtype_h = h_pre_raw.dtype  # 假设所有 h 张量 dtype 相同

    # 类型转换：激活用 bf16，参数用 fp32
    x_expanded = x_expanded.astype(jnp.bfloat16)
    h_pre_raw = h_pre_raw.astype(jnp.float32)
    h_post_raw = h_post_raw.astype(jnp.float32)
    h_res_raw = h_res_raw.astype(jnp.float32)

    # 创建并执行 kernel
    kernel = _create_mhc_pre_op_kernel(num_iters, eps)
    checkpointed_kernel = jax.checkpoint(
        kernel, policy=cp.save_anything_except_these_names(())
    )

    x_layer_in, H_post, H_res = checkpointed_kernel(x_expanded, h_pre_raw, h_post_raw, h_res_raw)

    # 类型还原
    return (
        x_layer_in.astype(original_dtype_x),
        H_post.astype(original_dtype_h),
        H_res.astype(original_dtype_h),
    )
"""
JAX 版 RWKV7 单步 wkv kernel（仅前向传播）
延迟编译 CUDA 扩展，专为 T=1 场景优化
"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union

# ---------- 延迟编译（改到当前目录） ----------
_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


def get_jax_generalized_delta_rule_single_step(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_single_step_{HEAD_SIZE}"
    _SO_PATH = _CURRENT_DIR / f"build_single_step_{HEAD_SIZE}/wkv7_single_step.so"

    def _ensure_compiled() -> pathlib.Path:
        """首次调用时编译 CUDA 扩展，产出放在当前源码目录"""
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_single_step_jax] First use – compiling CUDA kernel…")
        src_dir = _CURRENT_DIR
        build_dir = _BUILD_DIR
        build_dir.mkdir(exist_ok=True)

        # 获取 XLA 头文件路径
        xla_include_dir = jax.ffi.include_dir()
        if not xla_include_dir:
            raise RuntimeError("jax.ffi.include_dir() 返回空，请检查 JAX >= 0.4.31")

        # CUDA 编译 flags（移除 CHUNK_LEN 定义）
        cuda_flags = [
            "-ftz=true",
            "-prec-div=false",
            "-prec-sqrt=false",
            "--use_fast_math",
            "-O3",
            "-Xptxas=-O3",
            "-res-usage",
            "--extra-device-vectorization",
            f"-D_C_={HEAD_SIZE}",
        ]

        # CMake 配置
        cmake_args = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={_CURRENT_DIR}",
            f"-DXLA_INCLUDE_DIR={xla_include_dir}",
            f"-DCMAKE_CUDA_FLAGS={' '.join(cuda_flags)}",
        ]
        subprocess.check_call(cmake_args)

        # 构建与安装
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])
        subprocess.check_call(["cmake", "--install", str(build_dir)])

        if not _SO_PATH.exists():
            raise RuntimeError("Compilation failed – wkv7_single_step.so not found.")

        print("[rwkv7_single_step_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    # 注册 FFI 符号（仅前向）
    _lib = ctypes.CDLL(_ensure_compiled())
    jax.ffi.register_ffi_target(
        "wkv7_single_step_fwd",
        jax.ffi.pycapsule(_lib.Wkv7SingleStepFwd),
        platform="CUDA",
    )

    # ---------- 工具 ----------
    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        """(B, 1, H, K) <-> (B, H, 1, K)"""
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # ---------- 前向 kernel ----------
    def _wkv7_single_step_kernel(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
    ):
        """
        内部 kernel 接口
        参数: w,q,k,v,a,b,h0 -> y,s
        """
        B, H, K = q.shape
        dtype = q.dtype

        out_type = jax.ShapeDtypeStruct((B, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_single_step_fwd", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, h0)

        return y, s

    def wk7_single_step_kernel(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
    ):
        """前向计算函数"""
        y, s = _wkv7_single_step_kernel(w, q, k, v, a, b, h0)
        final_state = s  # 单步后直接返回状态
        return (y, final_state)

    # ---------- 主接口 ----------
    def generalized_delta_rule_single_step(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        单步广义 delta 规则（仅前向）
        参数:
            r,w,k,v,a,b: 输入张量，形状必须为 (B, 1, H, K) 或 (B, H, 1, K)
            initial_state: 可选 (B, H, K, K) 初始状态，None 则零初始化
            output_final_state: 是否同时返回最后状态
            head_first: 是否将 head 维提前
        返回:
            out: (B, 1, H, K)  与输入 dtype 一致
            last_state: (B, H, K, K) 当 output_final_state=True
        """
        # 统一转 (B, 1, H, K) 并验证 T=1
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)

        B, T, H, K = r.shape
        if T != 1:
            raise ValueError(f"Single-step kernel requires T=1, but got T={T}.")

        # 处理初始状态
        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        # 移除 T 维度后调用 kernel
        r = r[:, 0, :, :]  # (B, H, K)
        w = w[:, 0, :, :]
        k = k[:, 0, :, :]
        v = v[:, 0, :, :]
        a = a[:, 0, :, :]
        b = b[:, 0, :, :]

        # 调用前向 kernel
        out, last_state = wk7_single_step_kernel(w, r, k, v, a, b, h0)

        # 恢复 T 维度
        out = jnp.expand_dims(out, axis=1)  # (B, 1, H, K)
        out = jnp.asarray(out, r.dtype)

        if output_final_state:
            return out, last_state
        return out

    return generalized_delta_rule_single_step

"""
JAX 版 RWKV7-SN 单步 wkv kernel（仅前向）
"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union

_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
_NVCC_WRAPPER = _CURRENT_DIR.parents[1] / "cuda_tools" / "nvcc_wrap"


def get_jax_generalized_delta_rule_sn_single_step(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_single_step_{HEAD_SIZE}"
    _SO_PATH = _BUILD_DIR / "wkv7_sn_single_step.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_sn_single_step_jax] First use – compiling CUDA kernel…")
        src_dir = _CURRENT_DIR
        build_dir = _BUILD_DIR
        build_dir.mkdir(exist_ok=True)

        xla_include_dir = jax.ffi.include_dir()
        if not xla_include_dir:
            raise RuntimeError("jax.ffi.include_dir() 返回空，请检查 JAX >= 0.4.31")

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

        cmake_args = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={_CURRENT_DIR}",
            f"-DXLA_INCLUDE_DIR={xla_include_dir}",
            f"-DCMAKE_CUDA_COMPILER={_NVCC_WRAPPER}",
            f"-DCMAKE_CUDA_FLAGS={' '.join(cuda_flags)}",
        ]
        subprocess.check_call(cmake_args)
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])
        subprocess.check_call(["cmake", "--install", str(build_dir)])

        if not _SO_PATH.exists():
            raise RuntimeError("Compilation failed – wkv7_sn_single_step.so not found.")

        print("[rwkv7_sn_single_step_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    _lib = ctypes.CDLL(_ensure_compiled())
    jax.ffi.register_ffi_target(
        "wkv7_sn_single_step_fwd",
        jax.ffi.pycapsule(_lib.Wkv7SnSingleStepFwd),
        platform="CUDA",
    )

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        return x

    def _wkv7_sn_single_step_kernel(w, q, k, v, a, b, tau, do_sn, h0):
        B, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_sn_single_step_fwd",
            (out_type, s_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, do_sn, h0)
        return y, s

    def generalized_delta_rule_sn_single_step(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        tau: jnp.ndarray,
        do_sn,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)
        tau = jnp.asarray(tau, jnp.float32)

        B, T, H, K = r.shape
        if T != 1:
            raise ValueError(f"Single-step kernel requires T=1, but got T={T}.")

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        r = r[:, 0, :, :]
        w = w[:, 0, :, :]
        k = k[:, 0, :, :]
        v = v[:, 0, :, :]
        a = a[:, 0, :, :]
        b = b[:, 0, :, :]

        # JAX 默认关闭 x64，CUDA 侧也使用 int32（仅 0/1）。
        if isinstance(do_sn, bool):
            do_sn = jnp.full((B,), int(do_sn), dtype=jnp.int32)
        else:
            do_sn = jnp.asarray(do_sn, dtype=jnp.int32)

        out, last_state = _wkv7_sn_single_step_kernel(w, r, k, v, a, b, tau, do_sn, h0)
        out = jnp.expand_dims(out, axis=1)
        out = jnp.asarray(out, r.dtype)

        if output_final_state:
            return out, last_state
        return out

    return generalized_delta_rule_sn_single_step

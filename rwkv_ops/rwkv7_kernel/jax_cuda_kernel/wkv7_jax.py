"""
JAX 版 RWKV7 wkv kernel + generalized_delta_rule
延迟编译 CUDA 扩展，接口与 Torch 版本 1:1 对齐
"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import jax
import jax.numpy as jnp
from typing import Optional, Tuple, Union
from jax.ad_checkpoint import checkpoint_policies as cp

CHUNK_LEN = 16  # 这是一个常数
# ---------- 延迟编译（改到当前目录） ----------
_CURRENT_DIR = pathlib.Path(
    __file__
).parent.absolute()  # rwkv_ops/rwkv7_kernel/jax_cuda_kernel


def get_jax_generalized_delta_rule(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_{HEAD_SIZE}"
    _SO_PATH = _CURRENT_DIR / f"build_{HEAD_SIZE}/wkv7.so"

    def _ensure_compiled() -> pathlib.Path:
        """首次调用时编译 CUDA 扩展，产出放在当前源码目录"""
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_jax] First use – compiling CUDA kernel…")
        src_dir = _CURRENT_DIR
        build_dir = _BUILD_DIR
        build_dir.mkdir(exist_ok=True)

        # ---------- 关键：拿到 JAX 的 XLA 头文件路径 ----------
        xla_include_dir = jax.ffi.include_dir()  # 方案 3 核心 API
        if not xla_include_dir:
            raise RuntimeError("jax.ffi.include_dir() 返回空，请检查 JAX >= 0.4.31")

        # ---------- 关键：把数值稳定性 flag 写死 ----------
        cuda_flags = [
            "-ftz=true",  # flush sub-normal to zero
            "-prec-div=false",  # 更快除法，避免特殊路径
            "-prec-sqrt=false",  # 更快开方
            "--use_fast_math",  # 统一 fast math
            "-O3",
            "-Xptxas=-O3",
            "-res-usage",
            "--extra-device-vectorization",
            "-D_C_=64",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
        ]

        # 1. 配置
        cmake_args = [
            "cmake",
            "-S",
            str(src_dir),
            "-B",
            str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={_CURRENT_DIR}",
            f"-DXLA_INCLUDE_DIR={xla_include_dir}",  # 传给 CMake
            f"-DCMAKE_CUDA_FLAGS={' '.join(cuda_flags)}",
        ]
        subprocess.check_call(cmake_args)

        # 2. 构建
        subprocess.check_call(["cmake", "--build", str(build_dir), "-j"])

        # 3. 安装（把 .so 拷贝到当前目录）
        subprocess.check_call(["cmake", "--install", str(build_dir)])

        if not _SO_PATH.exists():
            raise RuntimeError("Compilation failed – wkv7.so not found.")

        print("[rwkv7_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    # 注册 FFI 符号
    _lib = ctypes.CDLL(_ensure_compiled())
    jax.ffi.register_ffi_target(
        "wkv7_fwd", jax.ffi.pycapsule(_lib.Wkv7Fwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_bwd", jax.ffi.pycapsule(_lib.Wkv7Bwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_inference", jax.ffi.pycapsule(_lib.Wkv7Inference), platform="CUDA"
    )

    # ---------- 工具 ----------
    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        """(B, T, H, K) <-> (B, H, T, K)"""
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # ---------- 前向 + 反向 kernel ----------

    def _wkv7_kernel(
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
        参数顺序与 wkv7_ffi.cc 声明完全一致：
        w,q,k,v,z,a,b  -> y,s,sa
        """
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        y, s, sa = jax.ffi.ffi_call(
            "wkv7_fwd", (out_type, s_type, sa_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, h0)

        return y, s, sa

    @jax.custom_vjp
    def wk7_kernel(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
    ):
        y, s, sa = _wkv7_kernel(w, q, k, v, a, b, h0)
        finnal_state = s[:, :, -1]
        return (y, jnp.transpose(finnal_state, [0, 1, 3, 2]))

    # 前向定义
    def _fwd(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
    ):
        y, s, sa = _wkv7_kernel(w, q, k, v, a, b, h0)
        finnal_state = s[:, :, -1]
        return (y, jnp.transpose(finnal_state, [0, 1, 3, 2])), (w, q, k, v, a, b, s, sa)

    def _wkv7_bwd_kernel(w, q, k, v, a, b, dy, s, sa, dht):
        dh0_type = jax.ShapeDtypeStruct(dht.shape, dht.dtype)
        dw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
        dq_type = jax.ShapeDtypeStruct(q.shape, q.dtype)
        dk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
        da_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
        db_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

        dh0, dw, dq, dk, dv, da, db = jax.ffi.ffi_call(
            "wkv7_bwd",
            (dh0_type, dw_type, dq_type, dk_type, dv_type, da_type, db_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, dy, s, sa, dht)

        return dw, dq, dk, dv, da, db, dh0

    # 反向定义
    def _bwd(res, grads):
        w, q, k, v, a, b, s, sa = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        # 调用反向 kernel
        return _wkv7_bwd_kernel(w, q, k, v, a, b, dy, s, sa, dht)

    wk7_kernel.defvjp(_fwd, _bwd)

    def generalized_delta_rule(
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
        广义 delta 规则，接口与 Torch 实现完全一致
        参数:
            r,w,k,v,a,b: 输入张量，形状 (B, T, H, K) 或 (B, H, T, K) 当 head_first=True
            initial_state: 可选 (B, H, K, K) 初始状态，None 则零初始化
            output_final_state: 是否同时返回最后状态
            head_first: 是否将 head 维提前
            chunk_len: 必须整除 T，默认 16
        返回:
            out: (B, T, H, K)  与输入 dtype 一致
            last_state: (B, H, K, K) 当 output_final_state=True
        """
        # 统一转 (B, T, H, K)
        dtype = r.dtype
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)

        B, T, H, K = r.shape
        if T % CHUNK_LEN:
            raise ValueError(
                f"Sequence length T={T} must be divisible by chunk_len={CHUNK_LEN}"
            )

        # 处理初始状态
        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        # 调用 kernel

        out, last_state = jax.checkpoint(
            wk7_kernel, policy=cp.save_anything_except_these_names(())
        )(w, r, k, v, a, b, h0)
        out = jnp.asarray(out, dtype)  # 保证输出 dtype 与输入一致

        if output_final_state:
            return out, last_state
        return out

    def _wkv7_inference_kernel(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
    ):
        """
        推理专用 kernel，不保存 sa 和中间 s
        返回: y (B, T, H, K), final_state (B, H, K, K)
        """
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        # **关键：仅返回最终状态，非 chunk 历史**
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_inference", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, h0)  # z 参数自动忽略

        return y, s

    # -------------------- 公共推理 API --------------------
    def generalized_delta_rule_inference(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        output_final_state: bool = True,
        initial_state: Optional[jnp.ndarray] = None,
        head_first: bool = False,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        纯推理版本的广义 delta 规则

        参数:
            r,w,k,v,a,b: 输入张量，形状 (B, T, H, K) 或 (B, H, T, K)
            initial_state: (B, H, K, K) 初始状态，None 则零初始化
            head_first: 是否将 head 维提前
        返回:
            out: (B, T, H, K) 输出，dtype 与输入一致
            final_state: (B, H, K, K) 仅最终状态
        """
        dtype = r.dtype
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)

        B, T, H, K = r.shape

        # 处理初始状态
        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        # **无需 checkpoint，推理不保存中间值**
        out, final_state = _wkv7_inference_kernel(w, r, k, v, a, b, h0)
        out = jnp.asarray(out, dtype)
        return out, final_state if output_final_state else out

    # 返回两个函数，用户按需选择
    return [generalized_delta_rule, generalized_delta_rule_inference]

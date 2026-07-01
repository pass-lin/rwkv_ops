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
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

# 引入自定义分区器 (适配 JAX 新版 Shardy 引擎)
from jax.experimental.custom_partitioning import custom_partitioning

CHUNK_LEN = 16  # 这是一个常数
# ---------- 延迟编译（改到当前目录） ----------
_CURRENT_DIR = pathlib.Path(
    __file__
).parent.absolute()  # rwkv_ops/rwkv7_kernel/jax_cuda_kernel

# 用于绕过 glibc 2.41+ 与 CUDA 13.1 的 rsqrt noexcept 冲突
_NVCC_WRAPPER = _CURRENT_DIR.parents[1] / "cuda_tools" / "nvcc_wrap"

# =========================================================================
# 【核心修复】：为 Shardy 引擎定义的静态 Einsum 切分映射字符串
# 字母含义: b=Batch, t=Time, h=Head, k=HeadDim1, v=HeadDim2, c=Chunk
# =========================================================================
FWD_RULE = "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v -> b t h k, b h c k v, b t h k"
BWD_RULE = "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h c k v, b t h k, b h k v -> b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v"
FWD_MASK_RULE = "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v, b t -> b t h k, b h c k v, b t h k"
BWD_MASK_RULE = "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b t, b t h k, b h c k v, b t h k, b h k v -> b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v"
INF_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v -> b t h k, b h k v"
)
INF_MASK_RULE = "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b h k v, b t -> b t h k, b h k v"


# 保留给老版本 JAX 兼容使用的回调函数
def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs, qs)


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs, qs, qs, qs, qs, qs)


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs)


# =========================================================================
# 【关键修复】：根据 JAX 官方文档提供的标准 Partition 样板函数
# =========================================================================
def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        # 获取最终分配给该算子的 sharding 信息
        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


# =========================================================================


def get_jax_generalized_delta_rule(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_{HEAD_SIZE}"
    _SO_PATH = _CURRENT_DIR / f"build_{HEAD_SIZE}/wkv7.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_jax] First use – compiling CUDA kernel…")
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
            "-D_C_=64",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
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
            raise RuntimeError("Compilation failed – wkv7.so not found.")

        print("[rwkv7_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

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

    jax.ffi.register_ffi_target(
        "wkv7_fwd_with_mask", jax.ffi.pycapsule(_lib.Wkv7FwdWithMask), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_bwd_with_mask", jax.ffi.pycapsule(_lib.Wkv7BwdWithMask), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_inference_with_mask",
        jax.ffi.pycapsule(_lib.Wkv7InferenceWithMask),
        platform="CUDA",
    )

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # =========================================================================
    # 前向 + 反向 kernel (无 Mask)
    # =========================================================================
    def _wkv7_kernel_impl(w, q, k, v, a, b, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        return jax.ffi.ffi_call(
            "wkv7_fwd", (out_type, s_type, sa_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, h0)

    @custom_partitioning
    def _wkv7_kernel(w, q, k, v, a, b, h0):
        return _wkv7_kernel_impl(w, q, k, v, a, b, h0)

    _wkv7_kernel.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_RULE,
        partition=_create_partition(_wkv7_kernel_impl),
    )

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

    def _wkv7_bwd_kernel_impl(w, q, k, v, a, b, dy, s, sa, dht):
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

    @custom_partitioning
    def _wkv7_bwd_kernel(w, q, k, v, a, b, dy, s, sa, dht):
        return _wkv7_bwd_kernel_impl(w, q, k, v, a, b, dy, s, sa, dht)

    _wkv7_bwd_kernel.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_RULE,
        partition=_create_partition(_wkv7_bwd_kernel_impl),
    )

    def _bwd(res, grads):
        w, q, k, v, a, b, s, sa = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        return _wkv7_bwd_kernel(w, q, k, v, a, b, dy, s, sa, dht)

    wk7_kernel.defvjp(_fwd, _bwd)

    # =========================================================================
    # 前向 + 反向 kernel (带 Mask)
    # =========================================================================
    def _wkv7_kernel_with_mask_impl(w, q, k, v, a, b, h0, mask):
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        return jax.ffi.ffi_call(
            "wkv7_fwd_with_mask",
            (out_type, s_type, sa_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, mask, h0)

    @custom_partitioning
    def _wkv7_kernel_with_mask(w, q, k, v, a, b, h0, mask):
        return _wkv7_kernel_with_mask_impl(w, q, k, v, a, b, h0, mask)

    _wkv7_kernel_with_mask.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_MASK_RULE,
        partition=_create_partition(_wkv7_kernel_with_mask_impl),
    )

    @jax.custom_vjp
    def wk7_kernel_with_mask(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        y, s, sa = _wkv7_kernel_with_mask(w, q, k, v, a, b, h0, mask)
        finnal_state = s[:, :, -1]
        return (y, jnp.transpose(finnal_state, [0, 1, 3, 2]))

    def _fwd_with_mask(
        w: jnp.ndarray,
        q: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        h0: jnp.ndarray,
        mask: jnp.ndarray,
    ):
        y, s, sa = _wkv7_kernel_with_mask(w, q, k, v, a, b, h0, mask)
        finnal_state = s[:, :, -1]
        return (y, jnp.transpose(finnal_state, [0, 1, 3, 2])), (
            w,
            q,
            k,
            v,
            a,
            b,
            s,
            sa,
            mask,
        )

    def _wkv7_bwd_kernel_with_mask_impl(w, q, k, v, a, b, mask, dy, s, sa, dht):
        dh0_type = jax.ShapeDtypeStruct(dht.shape, dht.dtype)
        dw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
        dq_type = jax.ShapeDtypeStruct(q.shape, q.dtype)
        dk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
        da_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
        db_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

        dh0, dw, dq, dk, dv, da, db = jax.ffi.ffi_call(
            "wkv7_bwd_with_mask",
            (dh0_type, dw_type, dq_type, dk_type, dv_type, da_type, db_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, mask, dy, s, sa, dht)
        return dw, dq, dk, dv, da, db, dh0

    @custom_partitioning
    def _wkv7_bwd_kernel_with_mask(w, q, k, v, a, b, mask, dy, s, sa, dht):
        return _wkv7_bwd_kernel_with_mask_impl(w, q, k, v, a, b, mask, dy, s, sa, dht)

    _wkv7_bwd_kernel_with_mask.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_MASK_RULE,
        partition=_create_partition(_wkv7_bwd_kernel_with_mask_impl),
    )

    def _bwd_with_mask(res, grads):
        w, q, k, v, a, b, s, sa, mask = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        dw, dq, dk, dv, da, db, dh0 = _wkv7_bwd_kernel_with_mask(
            w, q, k, v, a, b, mask, dy, s, sa, dht
        )
        return dw, dq, dk, dv, da, db, dh0, None

    wk7_kernel_with_mask.defvjp(_fwd_with_mask, _bwd_with_mask)

    # ---------- 公共 API ----------
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
        mask: Optional[jnp.ndarray] = None,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
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

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        if mask is None:
            out, last_state = wk7_kernel(w, r, k, v, a, b, h0)
        else:
            if mask.shape != (B, T):
                raise ValueError(
                    f"mask shape must be (B, T) = ({B}, {T}), got {mask.shape}"
                )
            mask = jnp.asarray(mask, jnp.bfloat16)
            out, last_state = wk7_kernel_with_mask(w, r, k, v, a, b, h0, mask)

        out = jnp.asarray(out, dtype)

        if output_final_state:
            return out, last_state
        return out

    # =========================================================================
    # 推理 Kernel (无 Mask)
    # =========================================================================
    def _wkv7_inference_kernel_impl(w, q, k, v, a, b, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_inference", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, h0)
        return y, s

    @custom_partitioning
    def _wkv7_inference_kernel(w, q, k, v, a, b, h0):
        return _wkv7_inference_kernel_impl(w, q, k, v, a, b, h0)

    _wkv7_inference_kernel.def_partition(
        infer_sharding_from_operands=_inf_infer_sharding,
        sharding_rule=INF_RULE,
        partition=_create_partition(_wkv7_inference_kernel_impl),
    )

    # =========================================================================
    # 推理 Kernel (带 Mask)
    # =========================================================================
    def _wkv7_inference_kernel_with_mask_impl(w, q, k, v, a, b, h0, mask):
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_inference_with_mask", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, mask, h0)
        return y, s

    @custom_partitioning
    def _wkv7_inference_kernel_with_mask(w, q, k, v, a, b, h0, mask):
        return _wkv7_inference_kernel_with_mask_impl(w, q, k, v, a, b, h0, mask)

    _wkv7_inference_kernel_with_mask.def_partition(
        infer_sharding_from_operands=_inf_infer_sharding,
        sharding_rule=INF_MASK_RULE,
        partition=_create_partition(_wkv7_inference_kernel_with_mask_impl),
    )

    # ---------- 公共推理 API ----------
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
        mask: Optional[jnp.ndarray] = None,
    ):
        dtype = r.dtype
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)

        B, T, H, K = r.shape

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        if mask is None:
            out, final_state = _wkv7_inference_kernel(w, r, k, v, a, b, h0)
        else:
            if mask.shape != (B, T):
                raise ValueError(
                    f"mask shape must be (B, T) = ({B}, {T}), got {mask.shape}"
                )
            mask = jnp.asarray(mask, jnp.bfloat16)
            out, final_state = _wkv7_inference_kernel_with_mask(
                w, r, k, v, a, b, h0, mask
            )

        out = jnp.asarray(out, dtype)
        return (out, final_state) if output_final_state else out

    return [generalized_delta_rule, generalized_delta_rule_inference]

"""
JAX 版 RWKV7-SN wkv kernel
mask 为 [B, T//16] 的显式 chunk-level 标志，0 表示跳过 State Norm，1 表示执行。
tau 为外部预处理后的 per-head per-chunk 阈值，返回 dtau 供上层 softplus 参数梯度。
"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

from jax.experimental.custom_partitioning import custom_partitioning

CHUNK_LEN = 16
_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()
_NVCC_WRAPPER = _CURRENT_DIR.parents[1] / "cuda_tools" / "nvcc_wrap"

FWD_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b c, b h k v -> "
    "b t h k, b h c k v, b t h k"
)
BWD_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b c, b t h k, "
    "b h c k v, b t h k, b h k v -> b t h k, b t h k, b t h k, b t h k, "
    "b t h k, b t h k, b c h, b c, b h k v"
)
INF_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b c, b h k v -> "
    "b t h k, b h k v"
)


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs, qs)


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs, qs, qs, qs, qs, qs, qs, qs)


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (qs, qs)


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


def get_jax_generalized_delta_rule_sn(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_{HEAD_SIZE}"
    _SO_PATH = _BUILD_DIR / "wkv7_sn.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_sn_jax] First use – compiling CUDA kernel…")
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
            raise RuntimeError("Compilation failed – wkv7_sn.so not found.")

        print("[rwkv7_sn_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    _lib = ctypes.CDLL(_ensure_compiled())

    jax.ffi.register_ffi_target(
        "wkv7_sn_fwd", jax.ffi.pycapsule(_lib.Wkv7SnFwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_sn_bwd", jax.ffi.pycapsule(_lib.Wkv7SnBwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_sn_inference",
        jax.ffi.pycapsule(_lib.Wkv7SnInference),
        platform="CUDA",
    )

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # -------------------- 训练前向 --------------------
    def _wkv7_sn_kernel_impl(w, q, k, v, a, b, tau, mask, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        return jax.ffi.ffi_call(
            "wkv7_sn_fwd", (out_type, s_type, sa_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, tau, mask, h0)

    @custom_partitioning
    def _wkv7_sn_kernel(w, q, k, v, a, b, tau, mask, h0):
        return _wkv7_sn_kernel_impl(w, q, k, v, a, b, tau, mask, h0)

    _wkv7_sn_kernel.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_RULE,
        partition=_create_partition(_wkv7_sn_kernel_impl),
    )

    def _apply_sn_to_final_state(state, tau, mask):
        # state: [B, H, K, K]; tau: [B, T//16, H]; mask: [B, T//16]
        last_tau = tau[:, -1][:, :, None, None]
        last_mask = mask[:, -1][:, None, None, None]
        tau_safe = jnp.maximum(last_tau, 1e-6)
        sn_state = last_tau * jnp.tanh(state / tau_safe)
        return jnp.where(last_mask > 0, sn_state, state)

    def _compute_outputs(y, s, tau, mask):
        final_state = s[:, :, -1]
        final_state = jnp.transpose(final_state, [0, 1, 3, 2])
        final_state = _apply_sn_to_final_state(final_state, tau, mask)
        return y, final_state

    @jax.custom_vjp
    def wk7_sn_kernel(w, q, k, v, a, b, tau, mask, h0):
        y, s, sa = _wkv7_sn_kernel(w, q, k, v, a, b, tau, mask, h0)
        return _compute_outputs(y, s, tau, mask)

    def _fwd(w, q, k, v, a, b, tau, mask, h0):
        y, s, sa = _wkv7_sn_kernel(w, q, k, v, a, b, tau, mask, h0)
        y_out, final_state = _compute_outputs(y, s, tau, mask)
        return (y_out, final_state), (w, q, k, v, a, b, tau, mask, s, sa)

    def _wkv7_sn_bwd_kernel_impl(w, q, k, v, a, b, tau, mask, dy, s, sa, dht):
        dh0_type = jax.ShapeDtypeStruct(dht.shape, dht.dtype)
        dtau_type = jax.ShapeDtypeStruct(tau.shape, tau.dtype)
        dw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
        dq_type = jax.ShapeDtypeStruct(q.shape, q.dtype)
        dk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
        da_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
        db_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

        dh0, dtau, dw, dq, dk, dv, da, db = jax.ffi.ffi_call(
            "wkv7_sn_bwd",
            (dh0_type, dtau_type, dw_type, dq_type, dk_type, dv_type, da_type, db_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, mask, dy, s, sa, dht)
        return dw, dq, dk, dv, da, db, dtau, dh0

    @custom_partitioning
    def _wkv7_sn_bwd_kernel(w, q, k, v, a, b, tau, mask, dy, s, sa, dht):
        return _wkv7_sn_bwd_kernel_impl(w, q, k, v, a, b, tau, mask, dy, s, sa, dht)

    _wkv7_sn_bwd_kernel.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_RULE,
        partition=_create_partition(_wkv7_sn_bwd_kernel_impl),
    )

    def _bwd(res, grads):
        w, q, k, v, a, b, tau, mask, s, sa = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        dw, dq, dk, dv, da, db, dtau, dh0 = _wkv7_sn_bwd_kernel(
            w, q, k, v, a, b, tau, mask, dy, s, sa, dht
        )
        return dw, dq, dk, dv, da, db, dtau, None, dh0

    wk7_sn_kernel.defvjp(_fwd, _bwd)

    # -------------------- 推理前向 --------------------
    def _wkv7_sn_inference_kernel_impl(w, q, k, v, a, b, tau, mask, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_sn_inference", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, tau, mask, h0)
        return y, s

    @custom_partitioning
    def _wkv7_sn_inference_kernel(w, q, k, v, a, b, tau, mask, h0):
        return _wkv7_sn_inference_kernel_impl(w, q, k, v, a, b, tau, mask, h0)

    _wkv7_sn_inference_kernel.def_partition(
        infer_sharding_from_operands=_inf_infer_sharding,
        sharding_rule=INF_RULE,
        partition=_create_partition(_wkv7_sn_inference_kernel_impl),
    )

    # -------------------- 公共 API --------------------
    def generalized_delta_rule_sn(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        tau: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        dtype = r.dtype
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)
        tau = jnp.asarray(tau, jnp.float32)
        B, T, H, K = r.shape
        if T % CHUNK_LEN:
            raise ValueError(
                f"Sequence length T={T} must be divisible by chunk_len={CHUNK_LEN}"
            )
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tau.shape} does not match expected (B={B}, T//16={T // CHUNK_LEN}, H={H})"
            )
        if mask is None:
            mask = jnp.ones((B, T // CHUNK_LEN), dtype=jnp.float32)
        else:
            mask = jnp.asarray(mask, jnp.float32)
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        out, last_state = wk7_sn_kernel(w, r, k, v, a, b, tau, mask, h0)
        out = jnp.asarray(out, dtype)

        if output_final_state:
            return out, last_state
        return out

    def generalized_delta_rule_sn_inference(
        r: jnp.ndarray,
        w: jnp.ndarray,
        k: jnp.ndarray,
        v: jnp.ndarray,
        a: jnp.ndarray,
        b: jnp.ndarray,
        tau: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        initial_state: Optional[jnp.ndarray] = None,
        output_final_state: bool = True,
        head_first: bool = False,
    ) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
        """
        State Norm 推理 / prefill 入口（无梯度）。

        与训练版本数值等价，但显存占用更低，因为不会分配反向所需的 `s`、`sa`
        checkpoint。注意当前 CUDA 推理 kernel 仍按 chunk 读取 tau/mask，所以 T 必须
        被 16 整除；任意长度请使用单步 RNN 接口。
        """
        dtype = r.dtype
        r = _transpose_head(r, head_first)
        w = _transpose_head(w, head_first)
        k = _transpose_head(k, head_first)
        v = _transpose_head(v, head_first)
        a = _transpose_head(a, head_first)
        b = _transpose_head(b, head_first)
        tau = jnp.asarray(tau, jnp.float32)

        B, T, H, K = r.shape
        if T % CHUNK_LEN:
            raise ValueError(
                f"RWKV-SN inference/prefill requires T divisible by {CHUNK_LEN}, "
                f"but got T={T}."
            )
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tau.shape} does not match expected (B={B}, T//16={T // CHUNK_LEN}, H={H})"
            )
        if mask is None:
            mask = jnp.ones((B, T // CHUNK_LEN), dtype=jnp.float32)
        else:
            mask = jnp.asarray(mask, jnp.float32)
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        out, final_state = _wkv7_sn_inference_kernel(w, r, k, v, a, b, tau, mask, h0)
        out = jnp.asarray(out, dtype)
        return (out, final_state) if output_final_state else out

    return [generalized_delta_rule_sn, generalized_delta_rule_sn_inference]

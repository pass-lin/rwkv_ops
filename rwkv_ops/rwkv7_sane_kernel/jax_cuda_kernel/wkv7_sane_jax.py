"""JAX 版 RWKV7-SANE CUDA kernel 封装。"""

from __future__ import annotations
import pathlib
import subprocess
import ctypes
import warnings
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing import Optional, Tuple, Union

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

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
    "b t h k, b t h k, b c h, b h k v"
)
INF_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b c, b h k v -> "
    "b t h k, b h k v"
)

FWD_NO_MASK_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b h k v -> "
    "b t h k, b h c k v, b t h k"
)
BWD_NO_MASK_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b t h k, "
    "b h c k v, b t h k, b h k v -> b t h k, b t h k, b t h k, b t h k, "
    "b t h k, b t h k, b c h, b h k v"
)
INF_NO_MASK_RULE = (
    "b t h k, b t h k, b t h k, b t h k, b t h k, b t h k, b c h, b h k v -> "
    "b t h k, b h k v"
)


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec


def _sharding_like_q(qs):
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_for_state(qs):
    """为 s / sa 之外的 State checkpoint (B, H, C, K, K) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(
        qs.mesh, PartitionSpec(spec[0], spec[2], None, spec[3], spec[3])
    )


def _sharding_for_final_state(qs):
    """为最终 State (B, H, K, K) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[2], spec[3], spec[3]))


def _sharding_for_tau(qs):
    """为 tau / dtau (B, T//16, H) 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], None, spec[2]))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (
        _sharding_like_q(qs),
        _sharding_for_state(qs),
        _sharding_like_q(qs),
    )


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    q_like = _sharding_like_q(qs)
    tau_like = _sharding_for_tau(qs)
    h0_like = _sharding_for_final_state(qs)
    return (
        q_like,
        q_like,
        q_like,
        q_like,
        q_like,
        q_like,
        tau_like,
        h0_like,
    )


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[1]
    return (_sharding_like_q(qs), _sharding_for_final_state(qs))


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


def get_jax_generalized_delta_rule_sane(HEAD_SIZE=64):
    _BUILD_DIR = _CURRENT_DIR / f"build_{HEAD_SIZE}"
    _SO_PATH = _BUILD_DIR / "wkv7_sane.so"

    def _ensure_compiled() -> pathlib.Path:
        if _SO_PATH.exists():
            return _SO_PATH

        print("[rwkv7_sane_jax] First use – compiling CUDA kernel…")
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
            raise RuntimeError("Compilation failed – wkv7_sane.so not found.")

        print("[rwkv7_sane_jax] Compilation finished – output at", _SO_PATH)
        return _SO_PATH

    _lib = ctypes.CDLL(_ensure_compiled())

    jax.ffi.register_ffi_target(
        "wkv7_sane_fwd", jax.ffi.pycapsule(_lib.Wkv7SaneFwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_sane_bwd", jax.ffi.pycapsule(_lib.Wkv7SaneBwd), platform="CUDA"
    )
    jax.ffi.register_ffi_target(
        "wkv7_sane_inference",
        jax.ffi.pycapsule(_lib.Wkv7SaneInference),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "wkv7_sane_fwd_no_mask",
        jax.ffi.pycapsule(_lib.Wkv7SaneFwdNoMask),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "wkv7_sane_bwd_no_mask",
        jax.ffi.pycapsule(_lib.Wkv7SaneBwdNoMask),
        platform="CUDA",
    )
    jax.ffi.register_ffi_target(
        "wkv7_sane_inference_no_mask",
        jax.ffi.pycapsule(_lib.Wkv7SaneInferenceNoMask),
        platform="CUDA",
    )

    def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.bfloat16)
        if head_first:
            return jnp.transpose(x, (0, 2, 1, 3))
        return x

    # 训练前向（带 mask）
    def _wkv7_sane_kernel_impl(w, q, k, v, a, b, tau, mask, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        return jax.ffi.ffi_call(
            "wkv7_sane_fwd", (out_type, s_type, sa_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, tau, mask, h0)

    @custom_partitioning
    def _wkv7_sane_kernel(w, q, k, v, a, b, tau, mask, h0):
        return _wkv7_sane_kernel_impl(w, q, k, v, a, b, tau, mask, h0)

    _wkv7_sane_kernel.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_RULE,
        partition=_create_partition(_wkv7_sane_kernel_impl),
    )

    # 训练前向（无 mask）
    def _wkv7_sane_kernel_no_mask_impl(w, q, k, v, a, b, tau, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        chunk_num = int(T // CHUNK_LEN)
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, chunk_num, K, K), jnp.float32)
        sa_type = jax.ShapeDtypeStruct((B, T, H, K), jnp.float32)

        return jax.ffi.ffi_call(
            "wkv7_sane_fwd_no_mask",
            (out_type, s_type, sa_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, h0)

    @custom_partitioning
    def _wkv7_sane_kernel_no_mask(w, q, k, v, a, b, tau, h0):
        return _wkv7_sane_kernel_no_mask_impl(w, q, k, v, a, b, tau, h0)

    _wkv7_sane_kernel_no_mask.def_partition(
        infer_sharding_from_operands=_fwd_infer_sharding,
        sharding_rule=FWD_NO_MASK_RULE,
        partition=_create_partition(_wkv7_sane_kernel_no_mask_impl),
    )

    def _apply_sane_to_final_state(state, tau, mask):
        # state: [B, H, K, K]; tau: [B, T//16, H]; mask: [B, T//16]
        last_tau = tau[:, -1][:, :, None, None]
        last_mask = mask[:, -1][:, None, None, None]
        tau_safe = jnp.maximum(last_tau, 1e-6)
        sane_state = last_tau * jnp.tanh(state / tau_safe)
        return jnp.where(last_mask > 0, sane_state, state)

    def _apply_sane_to_final_state_no_mask(state, tau):
        # state: [B, H, K, K]; tau: [B, T//16, H]
        last_tau = tau[:, -1][:, :, None, None]
        tau_safe = jnp.maximum(last_tau, 1e-6)
        return last_tau * jnp.tanh(state / tau_safe)

    def _compute_outputs(y, s, tau, mask):
        final_state = s[:, :, -1]
        final_state = jnp.transpose(final_state, [0, 1, 3, 2])
        final_state = _apply_sane_to_final_state(final_state, tau, mask)
        return y, final_state

    def _compute_outputs_no_mask(y, s, tau):
        final_state = s[:, :, -1]
        final_state = jnp.transpose(final_state, [0, 1, 3, 2])
        final_state = _apply_sane_to_final_state_no_mask(final_state, tau)
        return y, final_state

    @jax.custom_vjp
    def wk7_sane_kernel(w, q, k, v, a, b, tau, mask, h0):
        y, s, sa = _wkv7_sane_kernel(w, q, k, v, a, b, tau, mask, h0)
        return _compute_outputs(y, s, tau, mask)

    def _fwd(w, q, k, v, a, b, tau, mask, h0):
        y, s, sa = _wkv7_sane_kernel(w, q, k, v, a, b, tau, mask, h0)
        y_out, final_state = _compute_outputs(y, s, tau, mask)
        return (y_out, final_state), (w, q, k, v, a, b, tau, mask, s, sa)

    def _wkv7_sane_bwd_kernel_impl(w, q, k, v, a, b, tau, mask, dy, s, sa, dht):
        dh0_type = jax.ShapeDtypeStruct(dht.shape, dht.dtype)
        dtau_type = jax.ShapeDtypeStruct(tau.shape, tau.dtype)
        dw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
        dq_type = jax.ShapeDtypeStruct(q.shape, q.dtype)
        dk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
        da_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
        db_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

        dh0, dtau, dw, dq, dk, dv, da, db = jax.ffi.ffi_call(
            "wkv7_sane_bwd",
            (dh0_type, dtau_type, dw_type, dq_type, dk_type, dv_type, da_type, db_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, mask, dy, s, sa, dht)
        return dw, dq, dk, dv, da, db, dtau, dh0

    @custom_partitioning
    def _wkv7_sane_bwd_kernel(w, q, k, v, a, b, tau, mask, dy, s, sa, dht):
        return _wkv7_sane_bwd_kernel_impl(w, q, k, v, a, b, tau, mask, dy, s, sa, dht)

    _wkv7_sane_bwd_kernel.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_RULE,
        partition=_create_partition(_wkv7_sane_bwd_kernel_impl),
    )

    def _bwd(res, grads):
        w, q, k, v, a, b, tau, mask, s, sa = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        dw, dq, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_kernel(
            w, q, k, v, a, b, tau, mask, dy, s, sa, dht
        )
        return dw, dq, dk, dv, da, db, dtau, None, dh0

    wk7_sane_kernel.defvjp(_fwd, _bwd)

    # 训练前向/反向（无 mask）
    @jax.custom_vjp
    def wk7_sane_kernel_no_mask(w, q, k, v, a, b, tau, h0):
        y, s, sa = _wkv7_sane_kernel_no_mask(w, q, k, v, a, b, tau, h0)
        return _compute_outputs_no_mask(y, s, tau)

    def _fwd_no_mask(w, q, k, v, a, b, tau, h0):
        y, s, sa = _wkv7_sane_kernel_no_mask(w, q, k, v, a, b, tau, h0)
        y_out, final_state = _compute_outputs_no_mask(y, s, tau)
        return (y_out, final_state), (w, q, k, v, a, b, tau, s, sa)

    def _wkv7_sane_bwd_kernel_no_mask_impl(w, q, k, v, a, b, tau, dy, s, sa, dht):
        dh0_type = jax.ShapeDtypeStruct(dht.shape, dht.dtype)
        dtau_type = jax.ShapeDtypeStruct(tau.shape, tau.dtype)
        dw_type = jax.ShapeDtypeStruct(w.shape, w.dtype)
        dq_type = jax.ShapeDtypeStruct(q.shape, q.dtype)
        dk_type = jax.ShapeDtypeStruct(k.shape, k.dtype)
        dv_type = jax.ShapeDtypeStruct(v.shape, v.dtype)
        da_type = jax.ShapeDtypeStruct(a.shape, a.dtype)
        db_type = jax.ShapeDtypeStruct(b.shape, b.dtype)

        dh0, dtau, dw, dq, dk, dv, da, db = jax.ffi.ffi_call(
            "wkv7_sane_bwd_no_mask",
            (dh0_type, dtau_type, dw_type, dq_type, dk_type, dv_type, da_type, db_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, dy, s, sa, dht)
        return dw, dq, dk, dv, da, db, dtau, dh0

    @custom_partitioning
    def _wkv7_sane_bwd_kernel_no_mask(w, q, k, v, a, b, tau, dy, s, sa, dht):
        return _wkv7_sane_bwd_kernel_no_mask_impl(w, q, k, v, a, b, tau, dy, s, sa, dht)

    _wkv7_sane_bwd_kernel_no_mask.def_partition(
        infer_sharding_from_operands=_bwd_infer_sharding,
        sharding_rule=BWD_NO_MASK_RULE,
        partition=_create_partition(_wkv7_sane_bwd_kernel_no_mask_impl),
    )

    def _bwd_no_mask(res, grads):
        w, q, k, v, a, b, tau, s, sa = res
        dy, dht = grads
        dy = jnp.asarray(dy, jnp.bfloat16)
        dw, dq, dk, dv, da, db, dtau, dh0 = _wkv7_sane_bwd_kernel_no_mask(
            w, q, k, v, a, b, tau, dy, s, sa, dht
        )
        return dw, dq, dk, dv, da, db, dtau, dh0

    wk7_sane_kernel_no_mask.defvjp(_fwd_no_mask, _bwd_no_mask)

    # 推理前向（带 mask）
    def _wkv7_sane_inference_kernel_impl(w, q, k, v, a, b, tau, mask, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_sane_inference", (out_type, s_type), vmap_method="broadcast_all"
        )(w, q, k, v, a, b, tau, mask, h0)
        return y, s

    @custom_partitioning
    def _wkv7_sane_inference_kernel(w, q, k, v, a, b, tau, mask, h0):
        return _wkv7_sane_inference_kernel_impl(w, q, k, v, a, b, tau, mask, h0)

    _wkv7_sane_inference_kernel.def_partition(
        infer_sharding_from_operands=_inf_infer_sharding,
        sharding_rule=INF_RULE,
        partition=_create_partition(_wkv7_sane_inference_kernel_impl),
    )

    # 推理前向（无 mask）
    def _wkv7_sane_inference_kernel_no_mask_impl(w, q, k, v, a, b, tau, h0):
        B, T, H, K = q.shape
        dtype = q.dtype
        out_type = jax.ShapeDtypeStruct((B, T, H, K), dtype)
        s_type = jax.ShapeDtypeStruct((B, H, K, K), jnp.float32)

        y, s = jax.ffi.ffi_call(
            "wkv7_sane_inference_no_mask",
            (out_type, s_type),
            vmap_method="broadcast_all",
        )(w, q, k, v, a, b, tau, h0)
        return y, s

    @custom_partitioning
    def _wkv7_sane_inference_kernel_no_mask(w, q, k, v, a, b, tau, h0):
        return _wkv7_sane_inference_kernel_no_mask_impl(w, q, k, v, a, b, tau, h0)

    _wkv7_sane_inference_kernel_no_mask.def_partition(
        infer_sharding_from_operands=_inf_infer_sharding,
        sharding_rule=INF_NO_MASK_RULE,
        partition=_create_partition(_wkv7_sane_inference_kernel_no_mask_impl),
    )

    # 公共 API
    def generalized_delta_rule_sane(
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
        """带 State Anomaly Neutralization 的 RWKV-7 广义 delta 规则（训练版）。

        当 mask=None 且 output_final_state=True 时，会发出 UserWarning 并将
        final_state 设为 None，避免 padding chunk 污染 state。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。T 必须被 16 整除。
            tau: [B, T//16, H], float32。阈值，必须严格 > 1。
            mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE。
            initial_state: [B, H, K, K], float32, 可选。None 则零初始化。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 或 mask=None 时不返回。

        Raises:
            ValueError: T 不被 16 整除，或 tau/mask 形状不匹配。
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
                f"Sequence length T={T} must be divisible by chunk_len={CHUNK_LEN}"
            )
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tau.shape} does not match expected (B={B}, T//16={T // CHUNK_LEN}, H={H})"
            )

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        # 当且仅当需要 final_state 且显式提供 mask 时才使用带 mask 算子。
        use_mask = output_final_state and mask is not None

        if use_mask:
            mask = jnp.asarray(mask, jnp.float32)
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )
            out, last_state = wk7_sane_kernel(w, r, k, v, a, b, tau, mask, h0)
            out = jnp.asarray(out, dtype)
            return out, last_state

        # 无 mask 路径：chunk 边界无条件执行 State Anomaly Neutralization。
        out, last_state = wk7_sane_kernel_no_mask(w, r, k, v, a, b, tau, h0)
        out = jnp.asarray(out, dtype)

        if not output_final_state:
            return out

        # mask is None 且 output_final_state=True：警告并返回 None state。
        warnings.warn(
            "[rwkv7_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[rwkv7_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    def generalized_delta_rule_sane_inference(
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
        """带 State Anomaly Neutralization 的 RWKV-7 推理入口（无梯度）。

        与训练版本数值等价，但不保存反向 checkpoint，显存占用更低。
        tau/mask 按 chunk 读取，T 不必被 16 整除。

        Args:
            r, w, k, v, a, b: [B, T, H, K], bfloat16。
            tau: [B, T//16, H], float32。
            mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE。
            initial_state: [B, H, K, K], float32, 可选。
            output_final_state: bool, 是否返回最终 state。
            head_first: bool, 输入是否 head 维优先 ([B, H, T, K])。

        Returns:
            out: [B, T, H, K]，与输入同 dtype。
            final_state: [B, H, K, K], float32。
                output_final_state=False 或 mask=None 时不返回。

        Raises:
            ValueError: tau/mask 形状不匹配。
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
        if tau.shape != (B, T // CHUNK_LEN, H):
            raise ValueError(
                f"tau shape {tau.shape} does not match expected (B={B}, T//16={T // CHUNK_LEN}, H={H})"
            )

        if initial_state is None:
            h0 = jnp.zeros((B, H, K, K), jnp.float32)
        else:
            h0 = jnp.asarray(initial_state, jnp.float32)

        use_mask = output_final_state and mask is not None

        if use_mask:
            mask = jnp.asarray(mask, jnp.float32)
            if mask.shape != (B, T // CHUNK_LEN):
                raise ValueError(
                    f"mask shape {mask.shape} must match (B, T//16) = ({B}, {T // CHUNK_LEN})"
                )
            out, final_state = _wkv7_sane_inference_kernel(
                w, r, k, v, a, b, tau, mask, h0
            )
            out = jnp.asarray(out, dtype)
            return out, final_state

        out, final_state = _wkv7_sane_inference_kernel_no_mask(
            w, r, k, v, a, b, tau, h0
        )
        out = jnp.asarray(out, dtype)

        if not output_final_state:
            return out

        warnings.warn(
            "[rwkv7_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[rwkv7_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return [generalized_delta_rule_sane, generalized_delta_rule_sane_inference]

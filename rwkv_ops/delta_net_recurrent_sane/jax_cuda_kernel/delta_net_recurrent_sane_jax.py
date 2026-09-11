"""JAX 版 DeltaNet recurrent SANE CUDA kernel 封装。"""

from __future__ import annotations

import ctypes
import functools
import pathlib
import subprocess
import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from ...pallas_utils import create_partition

_CURRENT_DIR = pathlib.Path(__file__).parent.absolute()

# 按 (K, V, chunk_size) 缓存已编译的 CUDA 库，避免重复注册 FFI target。
_COMPILED_LIBS: dict[tuple[int, int, int], ctypes.CDLL] = {}
_REGISTERED_FFI_TARGETS: set[str] = set()

# nvcc 包装器用于绕过 glibc 2.41+ 与 CUDA 13.1 的 rsqrt noexcept 冲突。
_NVCC_WRAPPER = _CURRENT_DIR.parents[1] / "cuda_tools" / "nvcc_wrap"

# SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, k=HeadK, v=HeadV, c=Chunk
# tau 带 head 维可 TP，mask 无 head 维需 replicate。
FWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> "
    "b n t v, b n t v, b n c k v, b n t, b n t, b n k v"
)
BWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n t v, b n k v, "
    "b n t v, b n t, b n t, b n c k v, b n c, b c -> "
    "b n t k, b n t k, b n t v, b n t, b n t, b n k v, b n c"
)
INF_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> b n t v, b n k v"
)
SINGLE_RULE = "b n k, b n k, b n v, b n, b n, b n, b, b n k v -> b n v, b n k v"

# FFI target 名后缀与 .so 内导出符号的对应关系。
_FFI_SYMBOLS = {
    "fwd": ("DeltaNetRecurrentSaneFwdBf16", "DeltaNetRecurrentSaneFwdF32"),
    "bwd": ("DeltaNetRecurrentSaneBwdBf16", "DeltaNetRecurrentSaneBwdF32"),
    "inference": (
        "DeltaNetRecurrentSaneInferenceBf16",
        "DeltaNetRecurrentSaneInferenceF32",
    ),
    "single_step": (
        "DeltaNetRecurrentSaneSingleStepBf16",
        "DeltaNetRecurrentSaneSingleStepF32",
    ),
}


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None:
        return None
    return spec


def _sharding_like_q(qs):
    """为与 q 同形的 [B, N, T, K] 张量构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_like_v(vs):
    """为与 v 同形的 [B, N, T, V] 张量构造 sharding。"""
    spec = getattr(vs, "spec", None)
    if spec is None or len(spec) != 4:
        return vs
    return NamedSharding(vs.mesh, PartitionSpec(spec[0], spec[1], spec[2], None))


def _sharding_for_state(qs):
    """为 State checkpoint [B, N, C, K, V] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None, spec[3], None))


def _sharding_for_final_state(qs):
    """为最终 State / dh0 [B, N, K, V] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[3], None))


def _sharding_for_gb(qs):
    """为 beta [B, N, T] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[2]))


def _sharding_for_tau(qs):
    """为 tau / dtau [B, N, C] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_v(qs),
        _sharding_like_v(qs),
        _sharding_for_state(qs),
        _sharding_for_gb(qs),
        _sharding_for_gb(qs),
        _sharding_for_final_state(qs),
    )


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_q(qs),
        _sharding_like_q(qs),
        _sharding_like_v(qs),
        _sharding_for_gb(qs),
        _sharding_for_gb(qs),
        _sharding_for_final_state(qs),
        _sharding_for_tau(qs),
    )


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (_sharding_like_v(qs), _sharding_for_final_state(qs))


def _single_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    spec = _q_spec(qs)
    if spec is None or len(spec) != 3:
        return arg_shardings[0], arg_shardings[-1]
    o_sharding = NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None))
    state_sharding = NamedSharding(
        qs.mesh, PartitionSpec(spec[0], spec[1], spec[2], None)
    )
    return (o_sharding, state_sharding)


def _target_name(kind: str, dtype, K: int, V: int, chunk_size: int) -> str:
    """生成带 (dtype, K, V, chunk_size) 键的 FFI target 名。"""
    dtype_tag = "bf16" if dtype == jnp.bfloat16 else "f32"
    return f"delta_net_recurrent_sane_{kind}_{dtype_tag}_{K}_{V}_{chunk_size}"


def _ensure_compiled(K: int, V: int, chunk_size: int) -> pathlib.Path:
    """首次使用时按 (K, V, chunk_size) 编译 CUDA 扩展并返回 so 路径。"""
    build_dir = _CURRENT_DIR / f"build_{K}_{V}_{chunk_size}"
    so_path = build_dir / "delta_net_recurrent_sane.so"
    if so_path.exists():
        return so_path

    print(
        f"[delta_net_recurrent_sane_jax] First use – compiling CUDA kernel (K={K}, V={V}, chunk_size={chunk_size})…"
    )
    build_dir.mkdir(exist_ok=True)

    xla_include_dir = jax.ffi.include_dir()
    if not xla_include_dir:
        raise RuntimeError("jax.ffi.include_dir() 返回空，请检查 JAX >= 0.4.31")

    # 不加 fast math，保持与 triton / torch CUDA 实现的数值口径一致。
    cuda_flags = [
        "-O3",
        f"-D_K_={K}",
        f"-D_V_={V}",
        f"-D_CHUNK_LEN_={chunk_size}",
    ]

    cmake_args = [
        "cmake",
        "-S",
        str(_CURRENT_DIR),
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

    if not so_path.exists():
        raise RuntimeError(
            "Compilation failed – delta_net_recurrent_sane.so not found."
        )

    print("[delta_net_recurrent_sane_jax] Compilation finished – output at", so_path)
    return so_path


def _get_lib(K: int, V: int, chunk_size: int) -> None:
    """加载（必要时编译）CUDA 库并注册全部 FFI target，幂等。"""
    lib_key = (K, V, chunk_size)
    if lib_key not in _COMPILED_LIBS:
        _COMPILED_LIBS[lib_key] = ctypes.CDLL(str(_ensure_compiled(K, V, chunk_size)))
    lib = _COMPILED_LIBS[lib_key]

    for kind, (sym_bf16, sym_f32) in _FFI_SYMBOLS.items():
        for dtype, sym in ((jnp.bfloat16, sym_bf16), (jnp.float32, sym_f32)):
            name = _target_name(kind, dtype, K, V, chunk_size)
            if name not in _REGISTERED_FFI_TARGETS:
                jax.ffi.register_ffi_target(
                    name, jax.ffi.pycapsule(getattr(lib, sym)), platform="CUDA"
                )
                _REGISTERED_FFI_TARGETS.add(name)


def _transpose_head(x: jnp.ndarray, head_first: bool) -> jnp.ndarray:
    """在 [B, T, H, *] 与 [B, H, T, *] 之间切换。

    支持 q/k/v 的 4D 输入以及 beta 的 3D 输入。
    """
    x = jnp.asarray(x)
    if head_first:
        return x
    if x.ndim == 4:
        return jnp.transpose(x, (0, 2, 1, 3))
    if x.ndim == 3:
        return jnp.transpose(x, (0, 2, 1))
    raise ValueError(f"_transpose_head only supports 3D or 4D inputs, got {x.ndim}D")


def _transpose_tau(tau: jnp.ndarray) -> jnp.ndarray:
    """tau 公共接口始终为 [B, T//chunk_size, N]；需要转成 head-first [B, N, T//chunk_size]。"""
    tau = jnp.asarray(tau, dtype=jnp.float32)
    return jnp.transpose(tau, (0, 2, 1))


def _prepare_h0(initial_state, B, N, K, V):
    """准备 float32 初始 state，支持 [1, N, K, V] 广播。"""
    if initial_state is None:
        return jnp.zeros((B, N, K, V), dtype=jnp.float32)
    h0 = jnp.asarray(initial_state, dtype=jnp.float32)
    if h0.shape[0] == 1 and B > 1:
        h0 = jnp.broadcast_to(h0, (B, N, K, V))
    return h0


def _check_dtype(dtype) -> None:
    """CUDA kernel 仅提供 bfloat16 / float32 两套实例化。"""
    if dtype not in (jnp.bfloat16, jnp.float32):
        raise ValueError(
            f"delta_net_recurrent_sane CUDA kernel only supports bfloat16/float32, got {dtype}"
        )


def _warn_mask_none() -> None:
    warnings.warn(
        "[delta_net_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
        "由于未提供 padding mask，返回的 final_state 可能被污染，"
        "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
        "[delta_net_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
        "The returned final_state is set to None because padding chunks "
        "may contaminate the state. Provide an explicit mask to obtain final_state.",
        UserWarning,
        stacklevel=3,
    )


# 训练前向


def _fwd_out_shape(q, v, chunk_size: int):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return (
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // chunk_size, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    )


def _delta_net_recurrent_sane_fwd_ffi_call(
    q, k, v, beta, tau, mask, h0, chunk_size: int
):
    _, _, _, K = q.shape
    V = v.shape[-1]
    return jax.ffi.ffi_call(
        _target_name("fwd", q.dtype, K, V, chunk_size),
        _fwd_out_shape(q, v, chunk_size),
        vmap_method="broadcast_all",
    )(q, k, v, beta, tau, mask, h0)


@functools.partial(custom_partitioning, static_argnums=(8,))
def _delta_net_recurrent_sane_fwd_spmd(q, k, v, beta, tau, mask, h0, chunk_size: int):
    return _delta_net_recurrent_sane_fwd_ffi_call(
        q, k, v, beta, tau, mask, h0, chunk_size
    )


_delta_net_recurrent_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_delta_net_recurrent_sane_fwd_ffi_call),
)


# 训练反向


def _bwd_out_shape(q, v, chunk_size: int):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return (
        jax.ShapeDtypeStruct((B, N, T, K), q.dtype),
        jax.ShapeDtypeStruct((B, N, T, K), q.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // chunk_size), jnp.float32),
    )


def _delta_net_recurrent_sane_bwd_ffi_call(
    q, k, v, beta, dy, dht, kv_mem, inv_q, inv_k, chkp, tau, mask, chunk_size: int
):
    _, _, _, K = q.shape
    V = v.shape[-1]
    return jax.ffi.ffi_call(
        _target_name("bwd", q.dtype, K, V, chunk_size),
        _bwd_out_shape(q, v, chunk_size),
        vmap_method="broadcast_all",
    )(q, k, v, beta, dy, dht, kv_mem, inv_q, inv_k, chkp, tau, mask)


@functools.partial(custom_partitioning, static_argnums=(13,))
def _delta_net_recurrent_sane_bwd_spmd(
    q, k, v, beta, dy, dht, kv_mem, inv_q, inv_k, chkp, tau, mask, chunk_size: int
):
    return _delta_net_recurrent_sane_bwd_ffi_call(
        q, k, v, beta, dy, dht, kv_mem, inv_q, inv_k, chkp, tau, mask, chunk_size
    )


_delta_net_recurrent_sane_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_delta_net_recurrent_sane_bwd_ffi_call),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(7,))
def _delta_net_recurrent_sane_train(q, k, v, beta, tau, mask, h0, chunk_size: int):
    out, _, _, _, _, ht = _delta_net_recurrent_sane_fwd_spmd(
        q, k, v, beta, tau, mask, h0, chunk_size
    )
    return out, ht


def _dn_train_fwd(q, k, v, beta, tau, mask, h0, chunk_size: int):
    out, kv_mem, state_chkp, inv_q, inv_k, ht = _delta_net_recurrent_sane_fwd_spmd(
        q, k, v, beta, tau, mask, h0, chunk_size
    )
    return (out, ht), (q, k, v, beta, tau, mask, kv_mem, inv_q, inv_k, state_chkp)


def _dn_train_bwd(chunk_size: int, res, grads):
    q, k, v, beta, tau, mask, kv_mem, inv_q, inv_k, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, q.dtype)
    if dht is None:
        B, N, _, K = q.shape
        V = v.shape[-1]
        dht = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dq, dk, dv, dbeta, dh0, dtau = _delta_net_recurrent_sane_bwd_spmd(
        q,
        k,
        v,
        beta,
        dy,
        dht,
        kv_mem,
        inv_q,
        inv_k,
        state_chkp,
        tau,
        mask,
        chunk_size,
    )
    return dq, dk, dv, dbeta, dtau, None, dh0


_delta_net_recurrent_sane_train.defvjp(_dn_train_fwd, _dn_train_bwd)


# 推理前向


def _inf_out_shape(q, v):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return (
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    )


def _delta_net_recurrent_sane_inf_ffi_call(
    q, k, v, beta, tau, mask, h0, chunk_size: int
):
    _, _, _, K = q.shape
    V = v.shape[-1]
    return jax.ffi.ffi_call(
        _target_name("inference", q.dtype, K, V, chunk_size),
        _inf_out_shape(q, v),
        vmap_method="broadcast_all",
    )(q, k, v, beta, tau, mask, h0)


@functools.partial(custom_partitioning, static_argnums=(8,))
def _delta_net_recurrent_sane_inf_spmd(q, k, v, beta, tau, mask, h0, chunk_size: int):
    return _delta_net_recurrent_sane_inf_ffi_call(
        q, k, v, beta, tau, mask, h0, chunk_size
    )


_delta_net_recurrent_sane_inf_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=INF_RULE,
    partition=create_partition(_delta_net_recurrent_sane_inf_ffi_call),
)


# 单步 RNN


def _single_out_shape(q, v):
    B, N, K = q.shape
    V = v.shape[-1]
    return (
        jax.ShapeDtypeStruct((B, N, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    )


def _delta_net_recurrent_sane_single_step_ffi_call(
    q, k, v, beta, tau, do_sane, h0, chunk_size: int
):
    _, _, K = q.shape
    V = v.shape[-1]
    return jax.ffi.ffi_call(
        _target_name("single_step", q.dtype, K, V, chunk_size),
        _single_out_shape(q, v),
        vmap_method="broadcast_all",
    )(q, k, v, beta, tau, do_sane, h0)


@functools.partial(custom_partitioning, static_argnums=(8,))
def _delta_net_recurrent_sane_single_step_spmd(
    q, k, v, beta, tau, do_sane, h0, chunk_size: int
):
    return _delta_net_recurrent_sane_single_step_ffi_call(
        q, k, v, beta, tau, do_sane, h0, chunk_size
    )


_delta_net_recurrent_sane_single_step_spmd.def_partition(
    infer_sharding_from_operands=_single_infer_sharding,
    sharding_rule=SINGLE_RULE,
    partition=create_partition(_delta_net_recurrent_sane_single_step_ffi_call),
)


# 对外 API


def delta_net_recurrent_sane(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 训练算子（JAX CUDA）。

    在 chunk 边界（每 chunk_size 个 token）按 mask 对 state 执行
    `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        q, k: [B, T, H, K]，bfloat16 或 float32，查询与键。T 必须被 chunk_size 整除。
        v: [B, T, H, V]，与 q/k 同 dtype，值。
        g: [B, T, H]，float32，log-space decay。
        beta: [B, T, H]，float32，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，float32。SANE 阈值，必须 > 0。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。为编译期常量，每种
            (K, V, chunk_size) 组合首次使用时各自编译一次。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32。
            output_final_state=False 时为 None；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: T 不被 chunk_size 整除，tau/mask 形状不匹配，
            或输入 dtype 不是 bfloat16/float32。
    """
    dtype = v.dtype
    _check_dtype(dtype)
    q = jnp.asarray(q, dtype)
    k = jnp.asarray(k, dtype)
    beta = jnp.asarray(beta, jnp.float32)
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    beta = _transpose_head(beta, head_first)
    tau = _transpose_tau(tau)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % chunk_size != 0:
        raise ValueError(
            f"CUDA SANE kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
        )

    C = T // chunk_size
    if tau.shape != (B, N, C):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//chunk_size={C})"
        )

    use_mask = output_final_state and mask is not None
    if use_mask:
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        if mask_arr.shape != (B, C):
            raise ValueError(
                f"mask shape {mask_arr.shape} must match (B, T//chunk_size) = ({B}, {C})"
            )
    else:
        mask_arr = jnp.ones((B, C), dtype=jnp.float32)

    _get_lib(K, V, chunk_size)
    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _delta_net_recurrent_sane_train(
        q, k, v, beta, tau, mask_arr, h0, chunk_size
    )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out, None

    if mask is None:
        _warn_mask_none()
        return out, None

    return out, final_state


def delta_net_recurrent_sane_inference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 推理算子（JAX CUDA，无梯度）。

    与训练版数学一致但不保存反向中间量，支持任意序列长度 T：
    只在满 chunk 的边界执行 SANE，末尾不足 chunk 的余数步不执行。

    Args:
        q, k: [B, T, H, K]，bfloat16 或 float32，查询与键。T 支持任意长度。
        v: [B, T, H, V]，与 q/k 同 dtype，值。
        g: [B, T, H]，float32，log-space decay。
        beta: [B, T, H]，float32，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，float32。SANE 阈值。
        mask: [B, T//chunk_size]，float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；
            output_final_state=False 时为 None；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: tau/mask 形状不匹配，或输入 dtype 不是 bfloat16/float32。
    """
    dtype = v.dtype
    _check_dtype(dtype)
    q = jnp.asarray(q, dtype)
    k = jnp.asarray(k, dtype)
    beta = jnp.asarray(beta, jnp.float32)
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    beta = _transpose_head(beta, head_first)
    tau = _transpose_tau(tau)

    B, N, T, K = q.shape
    V = v.shape[-1]
    C = max(T // chunk_size, 1)

    if tau.shape[-1] < C:
        tau = jnp.pad(
            tau, ((0, 0), (0, 0), (0, C - tau.shape[-1])), constant_values=1.0
        )
    if tau.shape != (B, N, C):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//chunk_size={C})"
        )

    use_mask = output_final_state and mask is not None
    if use_mask:
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        if mask_arr.shape[-1] < C:
            mask_arr = jnp.pad(
                mask_arr,
                ((0, 0), (0, C - mask_arr.shape[-1])),
                constant_values=1.0,
            )
        if mask_arr.shape != (B, C):
            raise ValueError(
                f"mask shape {mask_arr.shape} must match (B, T//chunk_size) = ({B}, {C})"
            )
    else:
        mask_arr = jnp.ones((B, C), dtype=jnp.float32)

    _get_lib(K, V, chunk_size)
    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _delta_net_recurrent_sane_inf_spmd(
        q, k, v, beta, tau, mask_arr, h0, chunk_size
    )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out, None

    if mask is None:
        _warn_mask_none()
        return out, None

    return out, final_state


def delta_net_recurrent_sane_single_step(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    do_sane: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = True,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 单步 RNN（JAX CUDA）。

    输出基于 SANE 之前的 state，写出的下一步 state 按 do_sane blend 后落定。
    chunk_size 参数仅用于保持 API 一致性，单步实现不依赖 chunk 长度。

    Args:
        q, k: [B, H, K]，bfloat16 或 float32，查询与键。
        v: [B, H, V]，与 q/k 同 dtype，值。
        g: [B, H]，float32，log-space decay。
        beta: [B, H]，float32，写入强度，必须已在外部过 sigmoid。
        tau: [B, H]，float32。SANE 阈值。
        do_sane: [B]，float32。>0 在该步执行 SANE。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回下一步 state。
        head_first: bool，仅支持 True。
        chunk_size: int，忽略，仅参与编译缓存键。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V]，float32；output_final_state=False 时为 None。

    Raises:
        NotImplementedError: head_first=False。
        ValueError: tau/do_sane 形状不匹配，或输入 dtype 不是 bfloat16/float32。
    """
    if not head_first:
        raise NotImplementedError(
            "delta_net_recurrent_sane_single_step currently only supports head_first=True."
        )

    dtype = v.dtype
    _check_dtype(dtype)
    q = jnp.asarray(q, dtype)
    k = jnp.asarray(k, dtype)
    beta = jnp.asarray(beta, jnp.float32)
    tau = jnp.asarray(tau, jnp.float32)
    do_sane = jnp.asarray(do_sane, jnp.float32)

    B, N, K = q.shape
    V = v.shape[-1]
    if tau.shape != (B, N):
        raise ValueError(f"tau shape {tau.shape} does not match expected ({B}, {N})")
    if do_sane.shape != (B,):
        raise ValueError(
            f"do_sane shape {do_sane.shape} does not match expected ({B},)"
        )

    _get_lib(K, V, chunk_size)
    h0 = _prepare_h0(initial_state, B, N, K, V)
    out, next_state = _delta_net_recurrent_sane_single_step_spmd(
        q, k, v, beta, tau, do_sane, h0, chunk_size
    )
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, next_state
    return out, None

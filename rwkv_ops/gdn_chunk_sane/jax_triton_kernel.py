"""JAX 版 Gated DeltaNet chunkwise SANE Triton kernel 封装。"""

import warnings

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_triton as jt

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

# SANE 专用 kernel 从本包导入。
from .triton.chunk_h import _gdn_chunk_fwd_h_sane_kernel
from .triton.chunk_bwd_dhu import _gdn_chunk_bwd_dhu_sane_kernel

# 其余未改动 kernel 从 gdn_chunk.triton 子模块导入。
from ..gdn_chunk.triton.cumsum import _chunk_local_cumsum_kernel
from ..gdn_chunk.triton.intra import _gdn_chunk_fwd_intra_kernel
from ..gdn_chunk.triton.wy import _gdn_chunk_recompute_w_u_fwd_kernel
from ..gdn_chunk.triton.chunk_o import _gdn_chunk_fwd_o_kernel
from ..gdn_chunk.triton.chunk_bwd_dv import _gdn_chunk_bwd_dv_local_kernel
from ..gdn_chunk.triton.chunk_bwd_dqkwg import _gdn_chunk_bwd_dqkwg_kernel
from ..gdn_chunk.triton.wy_bwd import _gdn_chunk_prepare_wy_repr_bwd_kernel
from ..gdn_chunk.triton.l2norm import (
    _gdn_chunk_l2norm_bwd_kernel,
    _gdn_chunk_l2norm_fwd_kernel,
)


# ===== autotune cache helpers =====


def _clear_gdn_chunk_sane_autotune_cache():
    """清空 gdn_chunk_sane 所有 Triton kernel 的 autotune cache。"""
    for kernel in (
        _gdn_chunk_fwd_h_sane_kernel,
        _gdn_chunk_bwd_dhu_sane_kernel,
        _gdn_chunk_fwd_o_kernel,
        _gdn_chunk_fwd_intra_kernel,
        _gdn_chunk_recompute_w_u_fwd_kernel,
        _gdn_chunk_l2norm_fwd_kernel,
        _gdn_chunk_l2norm_bwd_kernel,
        _chunk_local_cumsum_kernel,
        _gdn_chunk_bwd_dqkwg_kernel,
        _gdn_chunk_bwd_dv_local_kernel,
        _gdn_chunk_prepare_wy_repr_bwd_kernel,
    ):
        if hasattr(kernel, "cache"):
            kernel.cache.clear()


# ===== layout helpers =====


def _transpose_head(x):
    """外部 [B, T, H, *] -> 内部 [B, H, T, *]。"""
    ndim = x.ndim
    if ndim == 3:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 4:
        return jnp.transpose(x, (0, 2, 1, 3))
    raise ValueError(f"Unsupported ndim={ndim}")


def _transpose_back(x):
    """内部 [B, H, T, *] -> 外部 [B, T, H, *]。"""
    ndim = x.ndim
    if ndim == 3:
        return jnp.transpose(x, (0, 2, 1))
    if ndim == 4:
        return jnp.transpose(x, (0, 2, 1, 3))
    raise ValueError(f"Unsupported ndim={ndim}")


# ===== SPMD helpers =====


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec


def _like_q(qs):
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _for_g1d(qs):
    """为 [B, H, T] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[2]))


def _for_h(qs):
    """为 [B, H, N, K, V] 构造 sharding；K 随 q 切，V replicate。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None, spec[3], None))


def _for_A(qs):
    """为 [B, H, N, C, C] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None, None, None))


def _for_final(qs):
    """为 [B, H, K, V] 构造 sharding；K 随 q 切，V replicate。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[3], None))


def _for_tau(qs):
    """为 tau / dtau [B, H, T//chunk_size] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None))


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


# ===== chunk-local cumsum =====


def _chunk_local_cumsum_call(g, chunk_size, reverse=False):
    """JAX-Triton 封装：在 [B, H, T] 上计算 chunk 内局部 cumsum。"""
    B, H, T = g.shape
    if T % chunk_size != 0:
        raise ValueError(
            f"Triton SANE kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
        )
    out_shape = jax.ShapeDtypeStruct(g.shape, jnp.float32)
    grid = (B * H, T // chunk_size)
    return jt.triton_call(
        g,
        B,
        H,
        T,
        kernel=_chunk_local_cumsum_kernel,
        out_shape=out_shape,
        grid=grid,
        C=chunk_size,
        REVERSE=reverse,
    )


_chunk_local_cumsum_spmd = custom_partitioning(
    _chunk_local_cumsum_call, static_argnums=(1, 2)
)
_chunk_local_cumsum_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_g1d(arg_shardings[0]),
    ),
    sharding_rule="b h t -> b h t",
    partition=_create_partition(_chunk_local_cumsum_call),
)


# ===== L2 norm =====


def _l2norm_fwd_call(x):
    """x: [B, H, T, K] -> (out, inv_norm)。"""
    B, H, T, K = x.shape
    x_2d = x.reshape(B * H * T, K)

    BK = 2 ** ((K - 1).bit_length())
    grid = (jt.cdiv(B * H * T, 64),)
    out_shapes = [
        jax.ShapeDtypeStruct(x_2d.shape, x.dtype),
        jax.ShapeDtypeStruct((B * H * T,), jnp.float32),
    ]
    out_2d, inv_norm = jt.triton_call(
        x_2d,
        B * H * T,
        K,
        kernel=_gdn_chunk_l2norm_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        BK=BK,
        BT=64,
    )
    out = out_2d.reshape(B, H, T, K)
    inv_norm = inv_norm.reshape(B, H, T)
    return out, inv_norm


def _l2norm_bwd_call(x, inv_norm, dout):
    B, H, T, K = x.shape
    x_2d = x.reshape(B * H * T, K)
    dout_2d = dout.reshape(B * H * T, K)
    inv_norm_1d = inv_norm.reshape(B * H * T)

    BK = 2 ** ((K - 1).bit_length())
    grid = (jt.cdiv(B * H * T, 64),)
    out_shapes = [jax.ShapeDtypeStruct(x_2d.shape, x.dtype)]
    (dx_2d,) = jt.triton_call(
        x_2d,
        inv_norm_1d,
        dout_2d,
        B * H * T,
        K,
        kernel=_gdn_chunk_l2norm_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        BK=BK,
        BT=64,
    )
    return dx_2d.reshape(B, H, T, K)


_l2norm_fwd_spmd = custom_partitioning(_l2norm_fwd_call)
_l2norm_fwd_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _for_g1d(arg_shardings[0]),
    ),
    sharding_rule="b h t k -> b h t k, b h t",
    partition=_create_partition(_l2norm_fwd_call),
)

_l2norm_bwd_spmd = custom_partitioning(_l2norm_bwd_call)
_l2norm_bwd_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
    ),
    sharding_rule="b h t k, b h t, b h t k -> b h t k",
    partition=_create_partition(_l2norm_bwd_call),
)


# ===== Forward kernels =====


def _fwd_intra_call(k, g, beta, chunk_size):
    B, H, T, K = k.shape
    C = chunk_size
    N = T // C
    out_shapes = [jax.ShapeDtypeStruct((B, H, N, C, C), k.dtype)]

    grid = (B * H * N,)
    BK = 2 ** ((K - 1).bit_length())
    (A,) = jt.triton_call(
        k,
        g,
        beta,
        B,
        H,
        T,
        K,
        kernel=_gdn_chunk_fwd_intra_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
    )
    return A


_fwd_intra_spmd = custom_partitioning(_fwd_intra_call, static_argnums=(3,))
_fwd_intra_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_A(arg_shardings[0]),
    ),
    sharding_rule="b h t k, b h t, b h t -> b h n c c",
    partition=_create_partition(_fwd_intra_call),
)


def _recompute_w_u_call(k, v, beta, A, g, chunk_size):
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
    ]

    grid = (B * H * (T // C),)
    BK = 2 ** ((K - 1).bit_length())
    BV = 2 ** ((V - 1).bit_length())
    w, u = jt.triton_call(
        k,
        v,
        beta,
        A,
        g,
        B,
        H,
        T,
        K,
        V,
        kernel=_gdn_chunk_recompute_w_u_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return w, u


_recompute_w_u_spmd = custom_partitioning(_recompute_w_u_call, static_argnums=(5,))
_recompute_w_u_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
    ),
    sharding_rule="b h t k, b h t v, b h t, b h n c c, b h t -> b h t k, b h t v",
    partition=_create_partition(_recompute_w_u_call),
)


def _fwd_h_sane_no_mask_call(k, w, u, g, tau, h0, chunk_size):
    """无 mask 的 SANE chunk_h 前向调用（chunk 边界无条件 SANE）。"""
    B, H, T, K = k.shape
    V = u.shape[-1]
    C = chunk_size
    N = T // C
    # kernel 签名始终要求 mask_ptr，无 mask 时传 dummy 零张量。
    dummy_mask = jnp.zeros((B, N), dtype=jnp.float32)
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct(u.shape, u.dtype),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
    ]

    def grid(meta):
        return (B * H * jt.cdiv(V, meta.get("BV", 64)),)

    BK = 2 ** ((K - 1).bit_length())
    BV = 64
    h, v_new, ht = jt.triton_call(
        k,
        w,
        u,
        g,
        tau,
        dummy_mask,
        h0,
        B,
        H,
        T,
        K,
        V,
        kernel=_gdn_chunk_fwd_h_sane_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        USE_MASK=False,
    )
    return h, v_new, ht


_fwd_h_sane_no_mask_spmd = custom_partitioning(
    _fwd_h_sane_no_mask_call, static_argnums=(6,)
)
_fwd_h_sane_no_mask_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _like_q(arg_shardings[2]),
        _for_final(arg_shardings[5]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t v, b h t, b h c, b h k v -> "
        "b h n k v, b h t v, b h k v"
    ),
    partition=_create_partition(_fwd_h_sane_no_mask_call),
)


def _fwd_h_sane_with_mask_call(k, w, u, g, tau, mask, h0, chunk_size):
    """带 mask 的 SANE chunk_h 前向调用。"""
    B, H, T, K = k.shape
    V = u.shape[-1]
    C = chunk_size
    N = T // C
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct(u.shape, u.dtype),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
    ]

    def grid(meta):
        return (B * H * jt.cdiv(V, meta.get("BV", 64)),)

    BK = 2 ** ((K - 1).bit_length())
    BV = 64
    h, v_new, ht = jt.triton_call(
        k,
        w,
        u,
        g,
        tau,
        mask,
        h0,
        B,
        H,
        T,
        K,
        V,
        kernel=_gdn_chunk_fwd_h_sane_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        USE_MASK=True,
    )
    return h, v_new, ht


_fwd_h_sane_with_mask_spmd = custom_partitioning(
    _fwd_h_sane_with_mask_call, static_argnums=(7,)
)
_fwd_h_sane_with_mask_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _like_q(arg_shardings[2]),
        _for_final(arg_shardings[6]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t v, b h t, b h c, b c, b h k v -> "
        "b h n k v, b h t v, b h k v"
    ),
    partition=_create_partition(_fwd_h_sane_with_mask_call),
)


def _fwd_o_call(q, k, v_new, h, g, chunk_size):
    B, H, T, K = q.shape
    V = v_new.shape[-1]
    C = chunk_size
    out_shapes = [jax.ShapeDtypeStruct(v_new.shape, v_new.dtype)]
    scale = float(K**-0.5)
    BK = 2 ** ((K - 1).bit_length())
    BV = 64

    def grid(meta):
        return (B * H * (T // C) * jt.cdiv(V, meta.get("BV", BV)),)

    (o,) = jt.triton_call(
        q,
        k,
        v_new,
        h,
        g,
        scale,
        B,
        H,
        T,
        K,
        V,
        kernel=_gdn_chunk_fwd_o_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return o


_fwd_o_spmd = custom_partitioning(_fwd_o_call, static_argnums=(5,))
_fwd_o_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[1]),
    ),
    sharding_rule="b h t k, b h t k, b h t v, b h n k v, b h t -> b h t v",
    partition=_create_partition(_fwd_o_call),
)


# ===== Backward kernels =====


def _bwd_dv_local_call(q, k, g, do, chunk_size):
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    out_shapes = [jax.ShapeDtypeStruct(do.shape, do.dtype)]
    scale = float(K**-0.5)
    BK = 2 ** ((K - 1).bit_length())
    BV = 64

    def grid(meta):
        return (B * H * (T // C) * jt.cdiv(V, meta.get("BV", BV)),)

    (dv,) = jt.triton_call(
        q,
        k,
        g,
        do,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_gdn_chunk_bwd_dv_local_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return dv


_bwd_dv_local_spmd = custom_partitioning(_bwd_dv_local_call, static_argnums=(4,))
_bwd_dv_local_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[3]),
    ),
    sharding_rule="b h t k, b h t k, b h t, b h t v -> b h t v",
    partition=_create_partition(_bwd_dv_local_call),
)


def _bwd_dhu_sane_no_mask_call(
    q, k, w, g, h, v_new, tau, do, dv_local, dht, chunk_size
):
    """无 mask 的 SANE chunk_h 反向调用。"""
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    N = T // C
    dummy_mask = jnp.zeros((B, N), dtype=jnp.float32)
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct((B, H, N), jnp.float32),
    ]

    def grid(meta):
        return (B * H * jt.cdiv(V, meta.get("BV", 64)),)

    BK = 2 ** ((K - 1).bit_length())
    BV = 64
    scale = float(K**-0.5)
    if dht is None:
        dht = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    dh, dh0, dv, dtau = jt.triton_call(
        q,
        k,
        w,
        g,
        h,
        v_new,
        tau,
        dummy_mask,
        do,
        dv_local,
        dht,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_gdn_chunk_bwd_dhu_sane_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
        zeroed_outputs=(3,),
        USE_MASK=False,
    )
    return dh, dh0, dv, dtau


_bwd_dhu_sane_no_mask_spmd = custom_partitioning(
    _bwd_dhu_sane_no_mask_call, static_argnums=(10,)
)
_bwd_dhu_sane_no_mask_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _for_final(arg_shardings[0]),
        _like_q(arg_shardings[8]),
        _for_tau(arg_shardings[0]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t k, b h t, b h n k v, b h t v, b h c, "
        "b h t v, b h t v, b h k v -> b h n k v, b h k v, b h t v, b h c"
    ),
    partition=_create_partition(_bwd_dhu_sane_no_mask_call),
)


def _bwd_dhu_sane_with_mask_call(
    q, k, w, g, h, v_new, tau, mask, do, dv_local, dht, chunk_size
):
    """带 mask 的 SANE chunk_h 反向调用。"""
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, T // C, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
        jax.ShapeDtypeStruct(do.shape, do.dtype),
        jax.ShapeDtypeStruct((B, H, T // C), jnp.float32),
    ]

    def grid(meta):
        return (B * H * jt.cdiv(V, meta.get("BV", 64)),)

    BK = 2 ** ((K - 1).bit_length())
    BV = 64
    scale = float(K**-0.5)
    if dht is None:
        dht = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    dh, dh0, dv, dtau = jt.triton_call(
        q,
        k,
        w,
        g,
        h,
        v_new,
        tau,
        mask,
        do,
        dv_local,
        dht,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_gdn_chunk_bwd_dhu_sane_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
        zeroed_outputs=(3,),
        USE_MASK=True,
    )
    return dh, dh0, dv, dtau


_bwd_dhu_sane_with_mask_spmd = custom_partitioning(
    _bwd_dhu_sane_with_mask_call, static_argnums=(11,)
)
_bwd_dhu_sane_with_mask_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _for_final(arg_shardings[0]),
        _like_q(arg_shardings[9]),
        _for_tau(arg_shardings[0]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t k, b h t, b h n k v, b h t v, b h c, b c, "
        "b h t v, b h t v, b h k v -> b h n k v, b h k v, b h t v, b h c"
    ),
    partition=_create_partition(_bwd_dhu_sane_with_mask_call),
)


def _bwd_dqkwg_call(q, k, v_new, w, g, h, dh, do, dv, chunk_size):
    B, H, T, K = q.shape
    V = v_new.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(w.shape, w.dtype),
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),
    ]
    scale = float(K**-0.5)

    def grid(meta):
        return (B * H * (T // C) * jt.cdiv(K, meta.get("BK", 64)),)

    dq, dk, dw, dg = jt.triton_call(
        q,
        k,
        v_new,
        w,
        g,
        h,
        dh,
        do,
        dv,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_gdn_chunk_bwd_dqkwg_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
    )
    return dq, dk, dw, dg


_bwd_dqkwg_spmd = custom_partitioning(_bwd_dqkwg_call, static_argnums=(9,))
_bwd_dqkwg_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
        _like_q(arg_shardings[2]),
        _for_g1d(arg_shardings[0]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t v, b h t k, b h t, b h n k v, b h n k v, b h t v, b h t v "
        "-> b h t k, b h t k, b h t k, b h t"
    ),
    partition=_create_partition(_bwd_dqkwg_call),
)


def _wy_bwd_call(k, v, beta, g, A, dw, dv, chunk_size):
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),
    ]

    grid = (B * H * (T // C),)
    dk, dv_out, db, dg = jt.triton_call(
        k,
        v,
        beta,
        g,
        A,
        dw,
        dv,
        B,
        H,
        T,
        K,
        V,
        kernel=_gdn_chunk_prepare_wy_repr_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
    )
    return dk, dv_out, db, dg


_wy_bwd_spmd = custom_partitioning(_wy_bwd_call, static_argnums=(7,))
_wy_bwd_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
        _like_q(arg_shardings[5]),
        _for_g1d(arg_shardings[0]),
    ),
    sharding_rule=(
        "b h t k, b h t v, b h t, b h t, b h n c c, b h t k, b h t v "
        "-> b h t k, b h t v, b h t, b h t"
    ),
    partition=_create_partition(_wy_bwd_call),
)


# ===== custom_vjp (per chunk_size, use_mask) =====


_GDN_CHUNK_SANE_TRITON_OP_CACHE = {}


def _get_gdn_chunk_sane_triton_op(chunk_size, use_mask):
    """返回对特定 (chunk_size, use_mask) 闭包的 custom_vjp 算子。"""
    key = (chunk_size, use_mask)
    if key in _GDN_CHUNK_SANE_TRITON_OP_CACHE:
        return _GDN_CHUNK_SANE_TRITON_OP_CACHE[key]

    if use_mask:

        def _fwd_impl(q, k, v, g, beta, tau, mask, h0):
            q2, inv_norm_q = _l2norm_fwd_spmd(q)
            k2, inv_norm_k = _l2norm_fwd_spmd(k)

            g_cum = _chunk_local_cumsum_spmd(g, chunk_size, False)
            A = _fwd_intra_spmd(k2, g_cum, beta, chunk_size)
            w, u = _recompute_w_u_spmd(k2, v, beta, A, g_cum, chunk_size)
            h, v_new, ht = _fwd_h_sane_with_mask_spmd(
                k2, w, u, g_cum, tau, mask, h0, chunk_size
            )
            o = _fwd_o_spmd(q2, k2, v_new, h, g_cum, chunk_size)

            res = (
                q,
                k,
                q2,
                k2,
                v,
                g_cum,
                beta,
                A,
                w,
                u,
                h,
                v_new,
                ht,
                inv_norm_q,
                inv_norm_k,
                tau,
                mask,
            )
            return o, ht, res

        @jax.custom_vjp
        def _op(q, k, v, g, beta, tau, mask, h0):
            o, ht, _ = _fwd_impl(q, k, v, g, beta, tau, mask, h0)
            return o, ht

        def _fwd(q, k, v, g, beta, tau, mask, h0):
            o, ht, res = _fwd_impl(q, k, v, g, beta, tau, mask, h0)
            return (o, ht), res

        def _bwd(res, grads):
            _clear_gdn_chunk_sane_autotune_cache()
            (
                q_orig,
                k_orig,
                q2,
                k2,
                v,
                g_cum,
                beta,
                A,
                w,
                u,
                h,
                v_new,
                ht,
                inv_norm_q,
                inv_norm_k,
                tau,
                mask,
            ) = res
            do, dht = grads

            dv_local = _bwd_dv_local_spmd(q2, k2, g_cum, do, chunk_size)
            dh, dh0, dv, dtau = _bwd_dhu_sane_with_mask_spmd(
                q2, k2, w, g_cum, h, v_new, tau, mask, do, dv_local, dht, chunk_size
            )
            dq, dk, dw, dg = _bwd_dqkwg_spmd(
                q2, k2, v_new, w, g_cum, h, dh, do, dv, chunk_size
            )
            dk2, dv2, db, dg2 = _wy_bwd_spmd(k2, v, beta, g_cum, A, dw, dv, chunk_size)
            dk = dk + dk2
            dv = dv + dv2
            dg = dg + dg2

            dg = _chunk_local_cumsum_spmd(dg, chunk_size, True)

            dq = _l2norm_bwd_spmd(q_orig, inv_norm_q, dq)
            dk = _l2norm_bwd_spmd(k_orig, inv_norm_k, dk)

            return dq, dk, dv, dg, db, dtau, None, dh0

        _op.defvjp(_fwd, _bwd)
    else:

        def _fwd_impl(q, k, v, g, beta, tau, h0):
            q2, inv_norm_q = _l2norm_fwd_spmd(q)
            k2, inv_norm_k = _l2norm_fwd_spmd(k)

            g_cum = _chunk_local_cumsum_spmd(g, chunk_size, False)
            A = _fwd_intra_spmd(k2, g_cum, beta, chunk_size)
            w, u = _recompute_w_u_spmd(k2, v, beta, A, g_cum, chunk_size)
            h, v_new, ht = _fwd_h_sane_no_mask_spmd(
                k2, w, u, g_cum, tau, h0, chunk_size
            )
            o = _fwd_o_spmd(q2, k2, v_new, h, g_cum, chunk_size)

            res = (
                q,
                k,
                q2,
                k2,
                v,
                g_cum,
                beta,
                A,
                w,
                u,
                h,
                v_new,
                ht,
                inv_norm_q,
                inv_norm_k,
                tau,
            )
            return o, ht, res

        @jax.custom_vjp
        def _op(q, k, v, g, beta, tau, h0):
            o, ht, _ = _fwd_impl(q, k, v, g, beta, tau, h0)
            return o, ht

        def _fwd(q, k, v, g, beta, tau, h0):
            o, ht, res = _fwd_impl(q, k, v, g, beta, tau, h0)
            return (o, ht), res

        def _bwd(res, grads):
            _clear_gdn_chunk_sane_autotune_cache()
            (
                q_orig,
                k_orig,
                q2,
                k2,
                v,
                g_cum,
                beta,
                A,
                w,
                u,
                h,
                v_new,
                ht,
                inv_norm_q,
                inv_norm_k,
                tau,
            ) = res
            do, dht = grads

            dv_local = _bwd_dv_local_spmd(q2, k2, g_cum, do, chunk_size)
            dh, dh0, dv, dtau = _bwd_dhu_sane_no_mask_spmd(
                q2, k2, w, g_cum, h, v_new, tau, do, dv_local, dht, chunk_size
            )
            dq, dk, dw, dg = _bwd_dqkwg_spmd(
                q2, k2, v_new, w, g_cum, h, dh, do, dv, chunk_size
            )
            dk2, dv2, db, dg2 = _wy_bwd_spmd(k2, v, beta, g_cum, A, dw, dv, chunk_size)
            dk = dk + dk2
            dv = dv + dv2
            dg = dg + dg2

            dg = _chunk_local_cumsum_spmd(dg, chunk_size, True)

            dq = _l2norm_bwd_spmd(q_orig, inv_norm_q, dq)
            dk = _l2norm_bwd_spmd(k_orig, inv_norm_k, dk)

            return dq, dk, dv, dg, db, dtau, dh0

        _op.defvjp(_fwd, _bwd)

    _GDN_CHUNK_SANE_TRITON_OP_CACHE[key] = _op
    return _op


# ===== Public API =====


def gated_delta_net_chunk_sane(
    q,
    k,
    v,
    g,
    beta,
    tau,
    mask=None,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """Gated DeltaNet chunkwise JAX-Triton SANE 实现（训练前向）。

    接口在 `gated_delta_net_chunk` 基础上增加 SANE 参数 `tau` 与 `mask`。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，decay gate（对数空间）。
        beta: [B, T, H]，写入强度门控，已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，SANE 阈值，float32，必须 > 1。
        mask: [B, T//chunk_size]，per-chunk mask，float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；
            output_final_state=False 时不返回；mask=None 时为 None。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 tau/mask 形状不匹配。
    """
    _clear_gdn_chunk_sane_autotune_cache()

    dtype = q.dtype
    q = _transpose_head(jnp.asarray(q, dtype))
    k = _transpose_head(jnp.asarray(k, dtype))
    v = _transpose_head(jnp.asarray(v, dtype))
    g = _transpose_head(jnp.asarray(g, jnp.float32))
    beta = _transpose_head(jnp.asarray(beta, jnp.float32))
    tau = _transpose_head(jnp.asarray(tau, jnp.float32))

    B, H, T, K = q.shape
    if T % chunk_size != 0:
        raise ValueError(
            f"Triton SANE kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
        )

    N = T // chunk_size
    if tau.shape != (B, H, N):
        raise ValueError(f"tau shape {tau.shape} 与期望 (B={B}, H={H}, T//C={N}) 不符")

    if initial_state is None:
        h0 = jnp.zeros((B, H, K, v.shape[-1]), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)
        if h0.shape[0] == 1 and B > 1:
            h0 = jnp.broadcast_to(h0, (B, H, K, v.shape[-1]))

    use_mask = output_final_state and mask is not None

    if use_mask:
        mask = jnp.asarray(mask, dtype=jnp.float32)
        if mask.shape != (B, N):
            raise ValueError(f"mask shape {mask.shape} 与期望 (B={B}, T//C={N}) 不符")
        out, final_state = _get_gdn_chunk_sane_triton_op(chunk_size, True)(
            q, k, v, g, beta, tau, mask, h0
        )
        out = _transpose_back(out)
        out = jnp.asarray(out, dtype)
        final_state = jnp.asarray(final_state, jnp.float32)
        return (out, final_state) if output_final_state else out

    # 无 mask 路径：chunk 边界无条件执行 SANE。
    out, final_state = _get_gdn_chunk_sane_triton_op(chunk_size, False)(
        q, k, v, g, beta, tau, h0
    )
    out = _transpose_back(out)
    out = jnp.asarray(out, dtype)
    final_state = jnp.asarray(final_state, jnp.float32)

    if not output_final_state:
        return out, None

    warnings.warn(
        "[gdn_chunk_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
        "由于未提供 padding mask，返回的 final_state 可能被污染，"
        "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
        "[gdn_chunk_sane] mask is None: using unconditional State Anomaly Neutralization. "
        "The returned final_state is set to None because padding chunks "
        "may contaminate the state. Provide an explicit mask to obtain final_state.",
        UserWarning,
        stacklevel=2,
    )
    return out, None

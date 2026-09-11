"""JAX 版 DeltaNet chunkwise Triton kernel 封装。"""

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_triton as jt

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from .triton.intra import _delta_net_chunk_fwd_intra_kernel
from .triton.wy import _delta_net_chunk_recompute_w_u_fwd_kernel
from .triton.chunk_h import _delta_net_chunk_fwd_h_kernel
from .triton.chunk_o import _delta_net_chunk_fwd_o_kernel
from .triton.chunk_bwd_dv import _delta_net_chunk_bwd_dv_local_kernel
from .triton.chunk_bwd_dhu import _delta_net_chunk_bwd_dhu_kernel
from .triton.chunk_bwd_dqk import _delta_net_chunk_bwd_dqk_kernel
from .triton.wy_bwd import _delta_net_chunk_prepare_wy_repr_bwd_kernel
from .triton.l2norm import (
    _delta_net_chunk_l2norm_bwd_kernel,
    _delta_net_chunk_l2norm_fwd_kernel,
)


def _clear_delta_net_chunk_autotune_cache():
    """清空 delta_net_chunk 所有 Triton kernel 的 autotune cache。

    与 torch 侧 workaround 一致：不同调用路径（有/无 initial_state、
    output_final_state=True/False）共享 kernel 对象，autotune cache 可能
    把为一条路径选出的 config 复用到另一条路径，导致 bf16 下输出 NaN。
    """
    for kernel in (
        _delta_net_chunk_fwd_h_kernel,
        _delta_net_chunk_fwd_o_kernel,
        _delta_net_chunk_fwd_intra_kernel,
        _delta_net_chunk_recompute_w_u_fwd_kernel,
        _delta_net_chunk_l2norm_fwd_kernel,
        _delta_net_chunk_l2norm_bwd_kernel,
        _delta_net_chunk_bwd_dhu_kernel,
        _delta_net_chunk_bwd_dqk_kernel,
        _delta_net_chunk_bwd_dv_local_kernel,
        _delta_net_chunk_prepare_wy_repr_bwd_kernel,
    ):
        if hasattr(kernel, "cache"):
            kernel.cache.clear()


# layout helpers


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


# SPMD helpers


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


def _for_beta1d(qs):
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


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


# L2 norm


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
        kernel=_delta_net_chunk_l2norm_fwd_kernel,
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
        kernel=_delta_net_chunk_l2norm_bwd_kernel,
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
        _for_beta1d(arg_shardings[0]),
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


# Forward kernels


def _fwd_intra_call(k, beta, chunk_size):
    B, H, T, K = k.shape
    C = chunk_size
    N = T // C
    out_shapes = [jax.ShapeDtypeStruct((B, H, N, C, C), k.dtype)]

    grid = (B * H * N,)
    BK = 2 ** ((K - 1).bit_length())
    (A,) = jt.triton_call(
        k,
        beta,
        B,
        H,
        T,
        K,
        kernel=_delta_net_chunk_fwd_intra_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
    )
    return A


_fwd_intra_spmd = custom_partitioning(_fwd_intra_call, static_argnums=(2,))
_fwd_intra_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_A(arg_shardings[0]),
    ),
    sharding_rule="b h t k, b h t -> b h n c c",
    partition=_create_partition(_fwd_intra_call),
)


def _recompute_w_u_call(k, v, beta, A, chunk_size):
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
        B,
        H,
        T,
        K,
        V,
        kernel=_delta_net_chunk_recompute_w_u_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return w, u


_recompute_w_u_spmd = custom_partitioning(_recompute_w_u_call, static_argnums=(4,))
_recompute_w_u_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
    ),
    sharding_rule="b h t k, b h t v, b h t, b h n c c -> b h t k, b h t v",
    partition=_create_partition(_recompute_w_u_call),
)


def _fwd_h_call(k, w, u, h0, chunk_size):
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
        h0,
        B,
        H,
        T,
        K,
        V,
        kernel=_delta_net_chunk_fwd_h_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
    )
    return h, v_new, ht


_fwd_h_spmd = custom_partitioning(_fwd_h_call, static_argnums=(4,))
_fwd_h_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _like_q(arg_shardings[2]),
        _for_final(arg_shardings[3]),
    ),
    sharding_rule="b h t k, b h t k, b h t v, b h k v -> b h n k v, b h t v, b h k v",
    partition=_create_partition(_fwd_h_call),
)


def _fwd_o_call(q, k, v_new, h, chunk_size):
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
        scale,
        B,
        H,
        T,
        K,
        V,
        kernel=_delta_net_chunk_fwd_o_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return o


_fwd_o_spmd = custom_partitioning(_fwd_o_call, static_argnums=(4,))
_fwd_o_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[1]),
    ),
    sharding_rule="b h t k, b h t k, b h t v, b h n k v -> b h t v",
    partition=_create_partition(_fwd_o_call),
)


# Backward kernels


def _bwd_dv_local_call(q, k, do, chunk_size):
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
        do,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_delta_net_chunk_bwd_dv_local_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return dv


_bwd_dv_local_spmd = custom_partitioning(_bwd_dv_local_call, static_argnums=(3,))
_bwd_dv_local_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[2]),
    ),
    sharding_rule="b h t k, b h t k, b h t v -> b h t v",
    partition=_create_partition(_bwd_dv_local_call),
)


def _bwd_dhu_call(q, k, w, do, dv_local, dht, chunk_size):
    B, H, T, K = q.shape
    V = do.shape[-1]
    C = chunk_size
    N = T // C
    out_shapes = [
        jax.ShapeDtypeStruct((B, H, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, H, K, V), jnp.float32),
        jax.ShapeDtypeStruct(do.shape, do.dtype),
    ]

    def grid(meta):
        return (B * H * jt.cdiv(V, meta.get("BV", 64)),)

    BK = 2 ** ((K - 1).bit_length())
    BV = 64
    scale = float(K**-0.5)
    if dht is None:
        dht = jnp.zeros((B, H, K, V), dtype=jnp.float32)

    dh, dh0, dv = jt.triton_call(
        q,
        k,
        w,
        do,
        dv_local,
        dht,
        B,
        H,
        T,
        K,
        V,
        scale,
        kernel=_delta_net_chunk_bwd_dhu_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
        BK=BK,
        BV=BV,
    )
    return dh, dh0, dv


_bwd_dhu_spmd = custom_partitioning(_bwd_dhu_call, static_argnums=(6,))
_bwd_dhu_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _for_h(arg_shardings[0]),
        _for_final(arg_shardings[0]),
        _like_q(arg_shardings[3]),
    ),
    sharding_rule="b h t k, b h t k, b h t k, b h t v, b h t v, b h k v -> b h n k v, b h k v, b h t v",
    partition=_create_partition(_bwd_dhu_call),
)


def _bwd_dqk_call(q, k, v_new, w, h, dh, do, dv, chunk_size):
    B, H, T, K = q.shape
    V = v_new.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct(q.shape, q.dtype),
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(w.shape, w.dtype),
    ]
    scale = float(K**-0.5)

    def grid(meta):
        return (B * H * (T // C) * jt.cdiv(K, meta.get("BK", 64)),)

    dq, dk, dw = jt.triton_call(
        q,
        k,
        v_new,
        w,
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
        kernel=_delta_net_chunk_bwd_dqk_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
    )
    return dq, dk, dw


_bwd_dqk_spmd = custom_partitioning(_bwd_dqk_call, static_argnums=(8,))
_bwd_dqk_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
        _like_q(arg_shardings[2]),
    ),
    sharding_rule=(
        "b h t k, b h t k, b h t v, b h t k, b h n k v, b h n k v, b h t v, b h t v "
        "-> b h t k, b h t k, b h t k"
    ),
    partition=_create_partition(_bwd_dqk_call),
)


def _wy_bwd_call(k, v, beta, A, dw, dv, chunk_size):
    B, H, T, K = k.shape
    V = v.shape[-1]
    C = chunk_size
    out_shapes = [
        jax.ShapeDtypeStruct(k.shape, k.dtype),
        jax.ShapeDtypeStruct(v.shape, v.dtype),
        jax.ShapeDtypeStruct((B, H, T), jnp.float32),
    ]

    grid = (B * H * (T // C),)
    dk, dv_out, db = jt.triton_call(
        k,
        v,
        beta,
        A,
        dw,
        dv,
        B,
        H,
        T,
        K,
        V,
        kernel=_delta_net_chunk_prepare_wy_repr_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        C=C,
    )
    return dk, dv_out, db


_wy_bwd_spmd = custom_partitioning(_wy_bwd_call, static_argnums=(6,))
_wy_bwd_spmd.def_partition(
    infer_sharding_from_operands=lambda arg_shapes, arg_shardings: (
        _like_q(arg_shardings[0]),
        _like_q(arg_shardings[1]),
        _for_beta1d(arg_shardings[0]),
    ),
    sharding_rule=(
        "b h t k, b h t v, b h t, b h n c c, b h t k, b h t v -> b h t k, b h t v, b h t"
    ),
    partition=_create_partition(_wy_bwd_call),
)


# custom_vjp (per chunk_size, concrete constexpr)


_DELTA_NET_CHUNK_TRITON_OP_CACHE = {}


def _get_delta_net_chunk_triton_op(chunk_size):
    """返回对特定 chunk_size 闭包的 custom_vjp 算子。"""
    if chunk_size in _DELTA_NET_CHUNK_TRITON_OP_CACHE:
        return _DELTA_NET_CHUNK_TRITON_OP_CACHE[chunk_size]

    def _fwd_impl(q, k, v, beta, h0):
        q2, inv_norm_q = _l2norm_fwd_spmd(q)
        k2, inv_norm_k = _l2norm_fwd_spmd(k)

        A = _fwd_intra_spmd(k2, beta, chunk_size)
        w, u = _recompute_w_u_spmd(k2, v, beta, A, chunk_size)
        h, v_new, ht = _fwd_h_spmd(k2, w, u, h0, chunk_size)
        o = _fwd_o_spmd(q2, k2, v_new, h, chunk_size)

        # l2norm_bwd 需要原始输入（非归一化结果），因此同时保存原始 q/k。
        res = (
            q,
            k,
            q2,
            k2,
            v,
            beta,
            A,
            w,
            u,
            h,
            v_new,
            ht,
            inv_norm_q,
            inv_norm_k,
        )
        return o, ht, res

    @jax.custom_vjp
    def _op(q, k, v, beta, h0):
        o, ht, _ = _fwd_impl(q, k, v, beta, h0)
        return o, ht

    def _fwd(q, k, v, beta, h0):
        o, ht, res = _fwd_impl(q, k, v, beta, h0)
        return (o, ht), res

    def _bwd(res, grads):
        _clear_delta_net_chunk_autotune_cache()
        (
            q_orig,
            k_orig,
            q2,
            k2,
            v,
            beta,
            A,
            w,
            u,
            h,
            v_new,
            ht,
            inv_norm_q,
            inv_norm_k,
        ) = res
        do, dht = grads

        dv_local = _bwd_dv_local_spmd(q2, k2, do, chunk_size)
        dh, dh0, dv = _bwd_dhu_spmd(q2, k2, w, do, dv_local, dht, chunk_size)
        dq, dk, dw = _bwd_dqk_spmd(q2, k2, v_new, w, h, dh, do, dv, chunk_size)
        dk2, dv2, db = _wy_bwd_spmd(k2, v, beta, A, dw, dv, chunk_size)
        dk = dk + dk2
        dv = dv + dv2

        dq = _l2norm_bwd_spmd(q_orig, inv_norm_q, dq)
        dk = _l2norm_bwd_spmd(k_orig, inv_norm_k, dk)

        return dq, dk, dv, db, dh0

    _op.defvjp(_fwd, _bwd)
    _DELTA_NET_CHUNK_TRITON_OP_CACHE[chunk_size] = _op
    return _op


def delta_net_chunk(
    q,
    k,
    v,
    beta,
    initial_state=None,
    output_final_state=False,
    chunk_size=16,
):
    """DeltaNet chunkwise JAX-Triton 实现（训练前向）。

    接口与 `rwkv_ops.delta_net_chunk.native_keras_op.delta_net_chunk` 一致。

    Args:
        q: [B, T, H, K]，查询。
        k: [B, T, H, K]，键。
        v: [B, T, H, V]，值。
        beta: [B, T, H]，写入强度门控，已在外部过 sigmoid。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终状态。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 chunk_size < 16。
    """
    _clear_delta_net_chunk_autotune_cache()

    dtype = q.dtype
    q = _transpose_head(jnp.asarray(q, dtype))
    k = _transpose_head(jnp.asarray(k, dtype))
    v = _transpose_head(jnp.asarray(v, dtype))
    beta = _transpose_head(jnp.asarray(beta, jnp.float32))

    B, H, T, K = q.shape
    if T % chunk_size != 0:
        raise ValueError(f"T={T} 必须被 chunk_size={chunk_size} 整除")
    if chunk_size < 16:
        raise ValueError(f"Triton kernel requires chunk_size >= 16, got {chunk_size}")

    if initial_state is None:
        h0 = jnp.zeros((B, H, K, v.shape[-1]), dtype=jnp.float32)
    else:
        h0 = jnp.asarray(initial_state, dtype=jnp.float32)
        if h0.shape[0] == 1 and B > 1:
            h0 = jnp.broadcast_to(h0, (B, H, K, v.shape[-1]))

    _op = _get_delta_net_chunk_triton_op(chunk_size)
    out, final_state = _op(q, k, v, beta, h0)
    out = _transpose_back(out)
    out = jnp.asarray(out, dtype)
    final_state = jnp.asarray(final_state, jnp.float32)

    if not output_final_state:
        final_state = None
    return out, final_state

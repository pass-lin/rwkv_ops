"""JAX 版 DeltaNet recurrent Triton kernel 封装。"""

from __future__ import annotations

import functools
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_triton as jt
import triton

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from .triton_kernel import (
    delta_net_recurrent_bwd_kernel,
    delta_net_recurrent_fwd_kernel,
    delta_net_recurrent_inference_fwd_kernel,
    delta_net_recurrent_single_step_fwd_kernel,
)

# SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, k=HeadK, v=HeadV, c=Chunk
FWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n k v -> "
    "b n t v, b n t v, b n c k v, b n t, b n t, b n k v"
)
BWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t v, b n k v, b n t v, b n t, b n t, "
    "b n k v, b n c k v -> b n t k, b n t k, b n t v, b n t, b n k v"
)
INF_RULE = "b n t k, b n t k, b n t v, b n t, b n k v -> b n t v, b n k v"


def _q_spec(qs):
    spec = getattr(qs, "spec", None)
    if spec is None or len(spec) != 4:
        return None
    return spec


def _sharding_like_q(qs):
    """为与 q 同形的 [B, N, T, K] 张量构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(*spec))


def _sharding_like_v(vs):
    """为与 v 同形的 [B, N, T, V] 张量构造 sharding。"""
    spec = getattr(vs, "spec", None)
    if spec is None or len(spec) != 4:
        return vs
    return NamedSharding(vs.mesh, PartitionSpec(*spec))


def _sharding_for_state(qs):
    """为 State checkpoint [B, N, C, K, V] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None, spec[3], None))


def _sharding_for_final_state(qs):
    """为最终 State / dh0 [B, N, K, V] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[3], None))


def _sharding_for_beta(qs):
    """为 beta [B, N, T] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[2]))


def _fwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_v(qs),
        _sharding_like_v(qs),
        _sharding_for_state(qs),
        _sharding_for_beta(qs),
        _sharding_for_beta(qs),
        _sharding_for_final_state(qs),
    )


def _bwd_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (
        _sharding_like_q(qs),
        _sharding_like_q(qs),
        _sharding_like_v(qs),
        _sharding_for_beta(qs),
        _sharding_for_final_state(qs),
    )


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (_sharding_like_v(qs), _sharding_for_final_state(qs))


def _create_partition(impl_fn):
    def partition(mesh, arg_shapes, result_shape):
        def lower_fn(*args):
            return impl_fn(*args)

        result_shardings = jtu.tree_map(lambda x: x.sharding, result_shape)
        arg_shardings = jtu.tree_map(lambda x: x.sharding, arg_shapes)
        return mesh, lower_fn, result_shardings, arg_shardings

    return partition


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


def _prepare_h0(initial_state, B, N, K, V):
    """准备 float32 初始 state，支持 [1, N, K, V] 广播。"""
    if initial_state is None:
        return jnp.zeros((B, N, K, V), dtype=jnp.float32)
    h0 = jnp.asarray(initial_state, dtype=jnp.float32)
    if h0.shape[0] == 1 and B > 1:
        h0 = jnp.broadcast_to(h0, (B, N, K, V))
    return h0


def _compute_bv(V):
    """计算反向 kernel 的 V block 大小。"""
    return min(128, triton.next_power_of_2(int(V)))


def _compute_grid(B, H, V):
    """recurrent kernel 的启动 grid。"""
    BV = _compute_bv(V)
    NV = (V + BV - 1) // BV
    return (B * H * NV,)


# 训练前向


def _dn_recurrent_fwd_triton_call(q, k, v, beta, h0, chunk_size: int):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    chunk_num = T // chunk_size
    BV = _compute_bv(V)

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, V), dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, chunk_num, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)
    scale = K**-0.5

    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k, ht = jt.triton_call(
        q,
        k,
        v,
        beta,
        h0,
        scale,
        B,
        N,
        T,
        kernel=delta_net_recurrent_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        CHUNK_LEN=chunk_size,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
    )
    return out, kv_mem, state_chkp, inv_norm_q, inv_norm_k, ht


@functools.partial(custom_partitioning, static_argnums=(5,))
def _dn_recurrent_fwd_spmd(q, k, v, beta, h0, chunk_size: int):
    return _dn_recurrent_fwd_triton_call(q, k, v, beta, h0, chunk_size)


_dn_recurrent_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_dn_recurrent_fwd_triton_call),
)


# 训练反向


def _dn_recurrent_bwd_triton_call(
    q,
    k,
    v,
    beta,
    dy,
    dht,
    kv_mem,
    inv_norm_q,
    inv_norm_k,
    h0,
    state_chkp,
    chunk_size: int,
):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    BV = _compute_bv(V)

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, K), dtype),
        jax.ShapeDtypeStruct((B, N, T, K), dtype),
        jax.ShapeDtypeStruct((B, N, T, V), dtype),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)
    scale = K**-0.5

    return jt.triton_call(
        q,
        k,
        v,
        beta,
        dy,
        dht,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
        scale,
        B,
        N,
        T,
        kernel=delta_net_recurrent_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        zeroed_outputs=(0, 1, 2, 3, 4),
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        CHUNK_LEN=chunk_size,
        USE_FINAL_STATE_GRADIENT=True,
    )


@functools.partial(custom_partitioning, static_argnums=(11,))
def _dn_recurrent_bwd_spmd(
    q,
    k,
    v,
    beta,
    dy,
    dht,
    kv_mem,
    inv_norm_q,
    inv_norm_k,
    h0,
    state_chkp,
    chunk_size: int,
):
    return _dn_recurrent_bwd_triton_call(
        q,
        k,
        v,
        beta,
        dy,
        dht,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
        chunk_size,
    )


_dn_recurrent_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_dn_recurrent_bwd_triton_call),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(5,))
def _dn_recurrent_train(q, k, v, beta, h0, chunk_size: int):
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k, final_state = (
        _dn_recurrent_fwd_spmd(q, k, v, beta, h0, chunk_size)
    )
    return out, final_state


def _dn_train_fwd(q, k, v, beta, h0, chunk_size: int):
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k, final_state = (
        _dn_recurrent_fwd_spmd(q, k, v, beta, h0, chunk_size)
    )
    return (out, final_state), (
        q,
        k,
        v,
        beta,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
    )


def _dn_train_bwd(chunk_size: int, res, grads):
    q, k, v, beta, kv_mem, inv_norm_q, inv_norm_k, h0, state_chkp = res
    dy, dht = grads
    dy = jnp.asarray(dy, q.dtype)
    if dht is None:
        B, N, T, K = q.shape
        V = v.shape[-1]
        dht = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dq, dk, dv, dbeta, dh0 = _dn_recurrent_bwd_spmd(
        q,
        k,
        v,
        beta,
        dy,
        dht,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
        chunk_size,
    )
    return dq, dk, dv, dbeta, dh0


_dn_recurrent_train.defvjp(_dn_train_fwd, _dn_train_bwd)


# 推理前向


def _dn_recurrent_inf_triton_call(q, k, v, beta, h0):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    BV = _compute_bv(V)

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, V), dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)
    scale = K**-0.5

    return jt.triton_call(
        q,
        k,
        v,
        beta,
        h0,
        scale,
        B,
        N,
        T,
        kernel=delta_net_recurrent_inference_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
    )


@custom_partitioning
def _dn_recurrent_inf_spmd(q, k, v, beta, h0):
    return _dn_recurrent_inf_triton_call(q, k, v, beta, h0)


_dn_recurrent_inf_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=INF_RULE,
    partition=_create_partition(_dn_recurrent_inf_triton_call),
)


# 单步 RNN 前向


def _dn_recurrent_single_step_triton_call(q, k, v, beta, h0):
    B, N, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    BV = _compute_bv(V)

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, V), dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)
    scale = K**-0.5

    return jt.triton_call(
        q,
        k,
        v,
        beta,
        h0,
        scale,
        B,
        N,
        kernel=delta_net_recurrent_single_step_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
    )


# 对外 API


def delta_net_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """DeltaNet recurrent 训练算子（JAX Triton 实现）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        ValueError: T 不被 chunk_size 整除。
    """
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    beta = _transpose_head(beta, head_first)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % chunk_size != 0:
        raise ValueError(
            f"Triton kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
        )

    h0 = _prepare_h0(initial_state, B, N, K, V)
    out, final_state = _dn_recurrent_train(q, k, v, beta, h0, chunk_size)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, final_state
    return out


def delta_net_recurrent_inference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """DeltaNet recurrent 推理算子（JAX Triton 实现，无梯度）。

    Args:
        chunk_size: int，chunk 长度，默认 16。recurrent 推理内核不依赖 chunk_size，
            保留参数仅为了与训练算子签名一致。
    """
    _ = chunk_size
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    beta = _transpose_head(beta, head_first)

    B, N, T, K = q.shape
    V = v.shape[-1]
    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _dn_recurrent_inf_spmd(q, k, v, beta, h0)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, final_state
    return out


def delta_net_recurrent_single_step(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = True,
    chunk_size: int = 16,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """DeltaNet recurrent 单步 RNN 算子（JAX Triton 实现）。

    Args:
        chunk_size: int，chunk 长度，默认 16。单步 RNN 忽略该参数，仅签名一致。
    """
    _ = chunk_size
    if not head_first:
        raise NotImplementedError(
            "delta_net_recurrent_single_step currently only supports head_first=True."
        )

    dtype = v.dtype
    B, N, K = q.shape
    V = v.shape[-1]
    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, next_state = _dn_recurrent_single_step_triton_call(q, k, v, beta, h0)
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, next_state
    return out

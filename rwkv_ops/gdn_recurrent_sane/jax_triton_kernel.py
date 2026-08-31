"""JAX 版 Gated DeltaNet recurrent SANE Triton kernel 封装。"""

from __future__ import annotations

import functools
import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import jax_triton as jt
import triton

from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from .triton_kernel import (
    gated_delta_net_recurrent_sane_bwd_kernel,
    gated_delta_net_recurrent_sane_fwd_kernel,
    gated_delta_net_recurrent_sane_inference_fwd_kernel,
    gated_delta_net_recurrent_sane_single_step_fwd_kernel,
)

CHUNK_LEN = 16

# SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, k=HeadK, v=HeadV, c=Chunk
FWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> "
    "b n t v, b n t v, b n c k v, b n t, b n t, b n k v"
)
BWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n t v, b n k v, "
    "b n t v, b n t, b n k v, b n c k v, b n c, b c -> "
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b n k v"
)
INF_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> b n t v, b n k v"
)
SINGLE_RULE = "b n k, b n k, b n v, b n, b n, b n, b, b n k v -> b n v, b n k v"


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


def _sharding_for_gb(qs):
    """为 g/beta / inv_norm [B, N, T] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], spec[2]))


def _sharding_for_tau(qs):
    """为 tau / dtau [B, N, C] 构造 sharding。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], spec[1], None))


def _sharding_for_mask(qs):
    """为 mask [B, C] 构造 sharding（无 head 维，TP 下 replicate）。"""
    spec = _q_spec(qs)
    if spec is None:
        return qs
    return NamedSharding(qs.mesh, PartitionSpec(spec[0], None))


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
        _sharding_for_tau(qs),
        _sharding_for_final_state(qs),
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

    支持 q/k/v 的 4D 输入以及 g/beta 的 3D 输入。
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
    """tau 公共接口始终为 [B, T//16, N]；需要转成 head-first [B, N, T//16]。"""
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


def _apply_sane_to_final_state(
    state: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """对最终 state 应用 State Anomaly Neutralization。

    Args:
        state: [B, N, K, V], float32。
        tau: [B, N, C], float32。
        mask: [B, C], float32（可选）。

    Returns:
        [B, N, K, V], float32。
    """
    last_tau = tau[:, :, -1][:, :, None, None]
    tau_safe = jnp.maximum(last_tau, 1e-6)
    sane_state = last_tau * jnp.tanh(state / tau_safe)
    if mask is None:
        return sane_state
    last_mask = mask[:, -1][:, None, None, None]
    return jnp.where(last_mask > 0, sane_state, state)


def _compute_bv(V):
    """计算反向 kernel 的 V block 大小。"""
    return min(128, triton.next_power_of_2(int(V)))


def _compute_grid(B, H, V):
    """recurrent kernel 的启动 grid。"""
    BV = _compute_bv(V)
    NV = (V + BV - 1) // BV
    return (B * H * NV,)


# 训练前向


def _gdn_recurrent_sane_fwd_triton_call(q, k, v, g, beta, tau, mask, h0, use_mask):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = v.dtype
    chunk_num = T // CHUNK_LEN
    BV = _compute_bv(V)
    scale = K**-0.5

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, V), dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, chunk_num, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)

    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k, _ = jt.triton_call(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        h0,
        scale,
        B,
        N,
        T,
        kernel=gated_delta_net_recurrent_sane_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        CHUNK_LEN=CHUNK_LEN,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=False,
        USE_MASK=use_mask,
    )
    return out, kv_mem, state_chkp, inv_norm_q, inv_norm_k


def _gdn_recurrent_sane_fwd_spmd_impl(q, k, v, g, beta, tau, mask, h0, use_mask):
    return _gdn_recurrent_sane_fwd_triton_call(
        q, k, v, g, beta, tau, mask, h0, use_mask
    )


_gdn_recurrent_sane_fwd_spmd = custom_partitioning(
    _gdn_recurrent_sane_fwd_spmd_impl, static_argnums=(8,)
)
_gdn_recurrent_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=_create_partition(_gdn_recurrent_sane_fwd_spmd_impl),
)


# 训练反向


def _gdn_recurrent_sane_bwd_triton_call(
    q,
    k,
    v,
    g,
    beta,
    dy,
    dht,
    kv_mem,
    inv_norm_q,
    inv_norm_k,
    h0,
    state_chkp,
    tau,
    mask,
    use_mask,
):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = q.dtype
    BV = _compute_bv(V)
    scale = K**-0.5
    use_final_state_gradient = dht is not None

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, K), dtype),
        jax.ShapeDtypeStruct((B, N, T, K), dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // CHUNK_LEN), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)

    return jt.triton_call(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
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
        kernel=gated_delta_net_recurrent_sane_bwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        zeroed_outputs=(0, 1, 2, 3, 4, 5, 6),
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        CHUNK_LEN=CHUNK_LEN,
        USE_FINAL_STATE_GRADIENT=use_final_state_gradient,
        USE_MASK=use_mask,
    )


def _gdn_recurrent_sane_bwd_spmd_impl(
    q,
    k,
    v,
    g,
    beta,
    dy,
    dht,
    kv_mem,
    inv_norm_q,
    inv_norm_k,
    h0,
    state_chkp,
    tau,
    mask,
    use_mask,
):
    return _gdn_recurrent_sane_bwd_triton_call(
        q,
        k,
        v,
        g,
        beta,
        dy,
        dht,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
        tau,
        mask,
        use_mask,
    )


_gdn_recurrent_sane_bwd_spmd = custom_partitioning(
    _gdn_recurrent_sane_bwd_spmd_impl, static_argnums=(14,)
)
_gdn_recurrent_sane_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=_create_partition(_gdn_recurrent_sane_bwd_spmd_impl),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _gdn_recurrent_sane_train(q, k, v, g, beta, tau, mask, h0, use_mask):
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_sane_fwd_spmd(
        q, k, v, g, beta, tau, mask, h0, use_mask
    )
    final_state = _apply_sane_to_final_state(
        state_chkp[:, :, -1, :, :], tau, mask=mask if use_mask else None
    )
    return out, final_state


def _gdn_train_fwd(q, k, v, g, beta, tau, mask, h0, use_mask):
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_sane_fwd_spmd(
        q, k, v, g, beta, tau, mask, h0, use_mask
    )
    final_state = _apply_sane_to_final_state(
        state_chkp[:, :, -1, :, :], tau, mask=mask if use_mask else None
    )
    return (out, final_state), (
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        state_chkp,
        h0,
    )


def _gdn_train_bwd(use_mask, res, grads):
    (
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        state_chkp,
        h0,
    ) = res
    dy, dht = grads
    dy = jnp.asarray(dy, q.dtype)
    if dht is None:
        B, N, T, K = q.shape
        V = v.shape[-1]
        dht = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    dq, dk, dv, dg, dbeta, dtau, dh0 = _gdn_recurrent_sane_bwd_spmd(
        q,
        k,
        v,
        g,
        beta,
        dy,
        dht,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        h0,
        state_chkp,
        tau,
        mask,
        use_mask,
    )
    return dq, dk, dv, dg, dbeta, dtau, None, dh0


_gdn_recurrent_sane_train.defvjp(_gdn_train_fwd, _gdn_train_bwd)


# 推理前向


def _gdn_recurrent_sane_inf_triton_call(q, k, v, g, beta, tau, mask, h0, use_mask):
    B, N, T, K = q.shape
    V = v.shape[-1]
    dtype = v.dtype
    BV = _compute_bv(V)
    scale = K**-0.5

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, T, V), dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)

    return jt.triton_call(
        q,
        k,
        v,
        g,
        beta,
        tau,
        mask,
        h0,
        scale,
        B,
        N,
        T,
        kernel=gated_delta_net_recurrent_sane_inference_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
        USE_MASK=use_mask,
        CHUNK_LEN=CHUNK_LEN,
    )


def _gdn_recurrent_sane_inf_spmd_impl(q, k, v, g, beta, tau, mask, h0, use_mask):
    return _gdn_recurrent_sane_inf_triton_call(
        q, k, v, g, beta, tau, mask, h0, use_mask
    )


_gdn_recurrent_sane_inf_spmd = custom_partitioning(
    _gdn_recurrent_sane_inf_spmd_impl, static_argnums=(8,)
)
_gdn_recurrent_sane_inf_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=INF_RULE,
    partition=_create_partition(_gdn_recurrent_sane_inf_spmd_impl),
)


# 单步 RNN 前向


def _gdn_recurrent_sane_single_step_triton_call(q, k, v, g, beta, tau, do_sane, h0):
    B, N, K = q.shape
    V = v.shape[-1]
    dtype = v.dtype
    BV = _compute_bv(V)
    scale = K**-0.5

    out_shapes = [
        jax.ShapeDtypeStruct((B, N, V), dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]

    grid = _compute_grid(B, N, V)

    return jt.triton_call(
        q,
        k,
        v,
        g,
        beta,
        tau,
        do_sane,
        h0,
        scale,
        B,
        N,
        kernel=gated_delta_net_recurrent_sane_single_step_fwd_kernel,
        out_shape=out_shapes,
        grid=grid,
        K=K,
        V=V,
        BK=triton.next_power_of_2(K),
        BV=BV,
        USE_INITIAL_STATE=True,
        STORE_FINAL_STATE=True,
    )


def _gdn_recurrent_sane_single_step_spmd_impl(q, k, v, g, beta, tau, do_sane, h0):
    return _gdn_recurrent_sane_single_step_triton_call(
        q, k, v, g, beta, tau, do_sane, h0
    )


_gdn_recurrent_sane_single_step_spmd = custom_partitioning(
    _gdn_recurrent_sane_single_step_spmd_impl
)
_gdn_recurrent_sane_single_step_spmd.def_partition(
    infer_sharding_from_operands=_single_infer_sharding,
    sharding_rule=SINGLE_RULE,
    partition=_create_partition(_gdn_recurrent_sane_single_step_spmd_impl),
)


# 对外 API


def gated_delta_net_recurrent_sane(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = False,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[Tuple[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 训练算子（JAX Triton）。

    在 chunk 边界（每 16 个 token）按 mask 对 state 执行
    `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//16, H], float32。阈值，必须 > 0。
        mask: [B, T//16], float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, V] 或 [1, H, K, V], float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。当前 Triton kernel 仅支持 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V], float32。
            output_final_state=False 时不返回（为 None）；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: T 不被 16 整除，或 tau/mask 形状不匹配。
    """
    if chunk_size != CHUNK_LEN:
        raise NotImplementedError(
            f"JAX Triton gdn_recurrent_sane currently only supports chunk_size={CHUNK_LEN}, "
            f"got {chunk_size}"
        )

    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
    beta = _transpose_head(beta, head_first)
    tau = _transpose_tau(tau)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % CHUNK_LEN != 0:
        raise ValueError(
            f"Triton SANE kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    C = T // CHUNK_LEN
    if tau.shape != (B, N, C):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//16={C})"
        )

    use_mask = output_final_state and mask is not None
    if use_mask:
        mask_arr = jnp.asarray(mask, dtype=jnp.float32)
        if mask_arr.shape != (B, C):
            raise ValueError(
                f"mask shape {mask_arr.shape} must match (B, T//16) = ({B}, {C})"
            )
    else:
        mask_arr = jnp.ones((B, C), dtype=jnp.float32)

    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _gdn_recurrent_sane_train(
        q, k, v, g, beta, tau, mask_arr, h0, use_mask
    )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out, None

    if mask is None:
        warnings.warn(
            "[gdn_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[gdn_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return out, final_state


def gated_delta_net_recurrent_sane_inference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    mask: Optional[jnp.ndarray] = None,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
    chunk_size: int = 16,
) -> Union[Tuple[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 推理算子（JAX Triton）。

    与训练版数学一致但不保存反向中间量；T 不必被 16 整除。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//16, H], float32。
        mask: [B, T//16], float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V], float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。
        chunk_size: int，chunk 长度，默认 16。当前 Triton kernel 仅支持 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V], float32；
            output_final_state=False 时不返回（为 None）；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: tau/mask 形状不匹配。
    """
    if chunk_size != CHUNK_LEN:
        raise NotImplementedError(
            f"JAX Triton gdn_recurrent_sane_inference currently only supports chunk_size={CHUNK_LEN}, "
            f"got {chunk_size}"
        )

    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
    beta = _transpose_head(beta, head_first)
    tau = _transpose_tau(tau)

    B, N, T, K = q.shape
    V = v.shape[-1]
    C = max(T // CHUNK_LEN, 1)

    if tau.shape[-1] < C:
        tau = jnp.pad(
            tau, ((0, 0), (0, 0), (0, C - tau.shape[-1])), constant_values=1.0
        )
    if tau.shape != (B, N, C):
        raise ValueError(
            f"tau shape {tau.shape} does not match expected (B={B}, N={N}, T//16={C})"
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
                f"mask shape {mask_arr.shape} must match (B, T//16) = ({B}, {C})"
            )
    else:
        mask_arr = jnp.ones((B, C), dtype=jnp.float32)

    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _gdn_recurrent_sane_inf_spmd(
        q, k, v, g, beta, tau, mask_arr, h0, use_mask
    )

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if not output_final_state:
        return out, None

    if mask is None:
        warnings.warn(
            "[gdn_recurrent_sane] mask is None: 使用无条件 State Anomaly Neutralization 算子。"
            "由于未提供 padding mask，返回的 final_state 可能被污染，"
            "因此已将其设为 None。如需 final_state 请提供显式 mask。\n"
            "[gdn_recurrent_sane] mask is None: using unconditional State Anomaly Neutralization. "
            "The returned final_state is set to None because padding chunks "
            "may contaminate the state. Provide an explicit mask to obtain final_state.",
            UserWarning,
            stacklevel=2,
        )
        return out, None

    return out, final_state


def gated_delta_net_recurrent_sane_single_step(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    tau: jnp.ndarray,
    do_sane: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = True,
    chunk_size: int = 16,
) -> Union[Tuple[jnp.ndarray, Optional[jnp.ndarray]], jnp.ndarray]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 单步 RNN（JAX Triton）。

    Args:
        q, k: [B, H, K]，查询与键。
        v: [B, H, V]，值。
        g: [B, H]，log-space decay。
        beta: [B, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, H], float32。
        do_sane: [B], float32。>0 执行 SANE。
        initial_state: [B, H, K, V] 或 [1, H, K, V], float32，可选。
        output_final_state: bool，是否返回下一步 state。
        head_first: bool，输入输出是否 head 维优先。单步默认 True（[B, H, *]）。
        chunk_size: int，chunk 长度，默认 16。单步实现忽略该参数（仅签名一致）。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V], float32；output_final_state=False 时为 None。

    Raises:
        NotImplementedError: head_first=False 尚未支持。
        ValueError: tau/do_sane 形状不匹配。
    """
    if not head_first:
        raise NotImplementedError(
            "gated_delta_net_recurrent_sane_single_step currently only supports head_first=True."
        )

    dtype = v.dtype
    q = jnp.asarray(q)
    k = jnp.asarray(k)
    v = jnp.asarray(v)
    g = jnp.asarray(g)
    beta = jnp.asarray(beta)
    tau = jnp.asarray(tau, dtype=jnp.float32)
    do_sane = jnp.asarray(do_sane, dtype=jnp.float32)

    B, N, K = q.shape
    V = v.shape[-1]
    if tau.shape != (B, N):
        raise ValueError(f"tau shape {tau.shape} does not match expected ({B}, {N})")
    if do_sane.shape != (B,):
        raise ValueError(
            f"do_sane shape {do_sane.shape} does not match expected ({B},)"
        )

    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, next_state = _gdn_recurrent_sane_single_step_spmd(
        q, k, v, g, beta, tau, do_sane, h0
    )

    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, next_state
    return out, None

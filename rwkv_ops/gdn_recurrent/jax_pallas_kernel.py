"""JAX 版 Gated DeltaNet recurrent Pallas kernel 封装。"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from ..pallas_utils import create_partition, ensure_config, launch

CHUNK_LEN = 16

# SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, k=HeadK, v=HeadV, c=Chunk
FWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n k v -> "
    "b n t v, b n t v, b n c k v, b n t, b n t"
)
BWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n t v, b n k v, "
    "b n t v, b n t, b n c k v, b n k v -> "
    "b n t k, b n t k, b n t v, b n t, b n t, b n k v"
)
INF_RULE = "b n t k, b n t k, b n t v, b n t, b n t, b n k v -> b n t v, b n k v"
SINGLE_RULE = "b n k, b n k, b n v, b n, b n, b n k v -> b n v, b n k v"


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
    return NamedSharding(vs.mesh, PartitionSpec(spec[0], spec[1], spec[2], None))


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
    """为 g/beta [B, N, T] 构造 sharding。"""
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
        _sharding_for_gb(qs),
        _sharding_for_gb(qs),
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
    )


def _inf_infer_sharding(arg_shapes, arg_shardings):
    qs = arg_shardings[0]
    return (_sharding_like_v(qs), _sharding_for_final_state(qs))


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


def _prepare_h0(initial_state, B, N, K, V):
    """准备 float32 初始 state，支持 [1, N, K, V] 广播。"""
    if initial_state is None:
        return jnp.zeros((B, N, K, V), dtype=jnp.float32)
    h0 = jnp.asarray(initial_state, dtype=jnp.float32)
    if h0.shape[0] == 1 and B > 1:
        h0 = jnp.broadcast_to(h0, (B, N, K, V))
    return h0


def _gdn_recurrent_fwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    h0_ref,
    o_ref,
    kv_mem_ref,
    chkp_ref,
    inv_norm_q_ref,
    inv_norm_k_ref,
):
    """Gated DeltaNet recurrent 训练前向 Pallas kernel。

    grid 为 (B, N)，每个 program 处理一个 (batch, head)。
    chunk 内静态展开 16 步，保存每步 kv_mem 与 chunk 级 state 快照供反向使用。
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = q_ref.shape[2] // CHUNK_LEN
    scale = jnp.float32(q_ref.shape[3]) ** -0.5
    eps = jnp.float32(1e-6)

    state = h0_ref[b, h].astype(jnp.float32)

    def chunk_body(c, state):
        for j in range(CHUNK_LEN):
            t = c * CHUNK_LEN + j

            q_t = q_ref[b, h, t, :].astype(jnp.float32)
            k_t = k_ref[b, h, t, :].astype(jnp.float32)
            v_t = v_ref[b, h, t, :].astype(jnp.float32)
            g_t = g_ref[b, h, t].astype(jnp.float32)
            beta_t = beta_ref[b, h, t].astype(jnp.float32)

            inv_norm_q_t = jax.lax.rsqrt(jnp.sum(q_t * q_t) + eps)
            inv_norm_k_t = jax.lax.rsqrt(jnp.sum(k_t * k_t) + eps)
            inv_norm_q_ref[b, h, t] = inv_norm_q_t
            inv_norm_k_ref[b, h, t] = inv_norm_k_t

            q_hat = q_t * inv_norm_q_t
            k_hat = k_t * inv_norm_k_t
            q_tilde = q_hat * scale

            state = state * jnp.exp(g_t)
            kv_mem = jnp.sum(state * k_hat[:, None], axis=0)
            kv_mem_ref[b, h, t, :] = kv_mem

            delta = beta_t * (v_t - kv_mem)
            state = state + k_hat[:, None] * delta[None, :]

            out_t = jnp.sum(state * q_tilde[:, None], axis=0)
            o_ref[b, h, t, :] = out_t.astype(o_ref.dtype)

        chkp_ref[b, h, c] = state
        return state

    jax.lax.fori_loop(0, num_chunks, chunk_body, state)


def _gdn_recurrent_bwd_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    dy_ref,
    dht_ref,
    kv_mem_ref,
    inv_norm_q_ref,
    inv_norm_k_ref,
    chkp_ref,
    h0_ref,
    dq_ref,
    dk_ref,
    dv_ref,
    dg_ref,
    dbeta_ref,
    dh0_ref,
):
    """Gated DeltaNet recurrent 训练反向 Pallas kernel。

    grid 为 (B, N)，每个 program 处理一个 (batch, head)。
    从 dht 开始反向递推，利用 chunk 级 state 快照避免数值爆炸。
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    num_chunks = q_ref.shape[2] // CHUNK_LEN
    scale = jnp.float32(q_ref.shape[3]) ** -0.5

    dS = dht_ref[b, h].astype(jnp.float32)

    def chunk_body(c_rev, dS):
        c = num_chunks - 1 - c_rev
        state = chkp_ref[b, h, c].astype(jnp.float32)

        for j in range(CHUNK_LEN - 1, -1, -1):
            t = c * CHUNK_LEN + j

            q_t = q_ref[b, h, t, :].astype(jnp.float32)
            k_t = k_ref[b, h, t, :].astype(jnp.float32)
            v_t = v_ref[b, h, t, :].astype(jnp.float32)
            g_t = g_ref[b, h, t].astype(jnp.float32)
            beta_t = beta_ref[b, h, t].astype(jnp.float32)
            dy_t = dy_ref[b, h, t, :].astype(jnp.float32)
            kv_mem_t = kv_mem_ref[b, h, t, :].astype(jnp.float32)

            inv_norm_q_t = inv_norm_q_ref[b, h, t]
            inv_norm_k_t = inv_norm_k_ref[b, h, t]

            q_hat = q_t * inv_norm_q_t
            k_hat = k_t * inv_norm_k_t
            q_tilde = q_hat * scale

            delta = beta_t * (v_t - kv_mem_t)

            # state 表示 S_{t+1}；输出 out_t 依赖 S_{t+1} 与 q_tilde。
            d_q_tilde = jnp.sum(state * dy_t[None, :], axis=1)

            # dS 是 L 对 S_{t+1} 的梯度；加上 out_t 带来的贡献。
            dstate_new = dS + q_tilde[:, None] * dy_t[None, :]

            # 传播到 delta_t 与 kv_mem_t。
            d_delta = jnp.sum(dstate_new * k_hat[:, None], axis=0)
            d_v = beta_t * d_delta
            d_beta = jnp.sum((v_t - kv_mem_t) * d_delta)
            d_kv_mem = -beta_t * d_delta

            # S_{t+1} = S_t * exp(g_t) + k_hat * delta^T。
            state_decay = state - k_hat[:, None] * delta[None, :]

            # L 对 k_hat 的梯度。
            d_k_hat = jnp.sum(dstate_new * delta[None, :], axis=1) + jnp.sum(
                state_decay * d_kv_mem[None, :], axis=1
            )

            # L 对 S_t * exp(g_t) 的梯度。
            dstate_decay = dstate_new + k_hat[:, None] * d_kv_mem[None, :]

            # L 对 g_t 的梯度。
            d_g = jnp.sum(state_decay * dstate_decay)

            # 传给下一步的 L 对 S_t 的梯度。
            exp_g = jnp.exp(g_t)
            dS = exp_g * dstate_decay

            # 恢复 S_t 供下一次迭代使用。
            state = state_decay / exp_g

            # L2 归一化反向。
            d_q_hat = scale * d_q_tilde
            q_hat_dot = jnp.sum(q_hat * d_q_hat)
            dq_t = inv_norm_q_t * (d_q_hat - q_hat * q_hat_dot)

            k_hat_dot = jnp.sum(k_hat * d_k_hat)
            dk_t = inv_norm_k_t * (d_k_hat - k_hat * k_hat_dot)

            dq_ref[b, h, t, :] = dq_t.astype(dq_ref.dtype)
            dk_ref[b, h, t, :] = dk_t.astype(dk_ref.dtype)
            dv_ref[b, h, t, :] = d_v.astype(dv_ref.dtype)
            dg_ref[b, h, t] = d_g.astype(dg_ref.dtype)
            dbeta_ref[b, h, t] = d_beta.astype(dbeta_ref.dtype)

        return dS

    dS = jax.lax.fori_loop(0, num_chunks, chunk_body, dS)
    dh0_ref[b, h] = dS.astype(dh0_ref.dtype)


def _gdn_recurrent_inf_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    h0_ref,
    o_ref,
    ht_ref,
):
    """Gated DeltaNet recurrent 推理前向 Pallas kernel。

    与训练前向数学一致，但不保存反向所需的中间量。
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    T = q_ref.shape[2]
    scale = jnp.float32(q_ref.shape[3]) ** -0.5
    eps = jnp.float32(1e-6)

    state = h0_ref[b, h].astype(jnp.float32)

    def step_body(t, state):
        q_t = q_ref[b, h, t, :].astype(jnp.float32)
        k_t = k_ref[b, h, t, :].astype(jnp.float32)
        v_t = v_ref[b, h, t, :].astype(jnp.float32)
        g_t = g_ref[b, h, t].astype(jnp.float32)
        beta_t = beta_ref[b, h, t].astype(jnp.float32)

        inv_norm_q_t = jax.lax.rsqrt(jnp.sum(q_t * q_t) + eps)
        inv_norm_k_t = jax.lax.rsqrt(jnp.sum(k_t * k_t) + eps)
        q_hat = q_t * inv_norm_q_t * scale
        k_hat = k_t * inv_norm_k_t

        state = state * jnp.exp(g_t)
        kv_mem = jnp.sum(state * k_hat[:, None], axis=0)
        delta = beta_t * (v_t - kv_mem)
        state = state + k_hat[:, None] * delta[None, :]

        out_t = jnp.sum(state * q_hat[:, None], axis=0)
        o_ref[b, h, t, :] = out_t.astype(o_ref.dtype)
        return state

    state = jax.lax.fori_loop(0, T, step_body, state)
    ht_ref[b, h] = state.astype(ht_ref.dtype)


def _gdn_recurrent_single_step_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    h0_ref,
    o_ref,
    ht_ref,
):
    """Gated DeltaNet recurrent 单步 RNN Pallas kernel。

    输入无时间维：q/k 为 [B, N, K]，v/o 为 [B, N, V]，g/beta 为 [B, N]。
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    scale = jnp.float32(q_ref.shape[2]) ** -0.5
    eps = jnp.float32(1e-6)

    state = h0_ref[b, h].astype(jnp.float32)

    q_t = q_ref[b, h, :].astype(jnp.float32)
    k_t = k_ref[b, h, :].astype(jnp.float32)
    v_t = v_ref[b, h, :].astype(jnp.float32)
    g_t = g_ref[b, h].astype(jnp.float32)
    beta_t = beta_ref[b, h].astype(jnp.float32)

    inv_norm_q_t = jax.lax.rsqrt(jnp.sum(q_t * q_t) + eps)
    inv_norm_k_t = jax.lax.rsqrt(jnp.sum(k_t * k_t) + eps)
    q_hat = q_t * inv_norm_q_t * scale
    k_hat = k_t * inv_norm_k_t

    state = state * jnp.exp(g_t)
    kv_mem = jnp.sum(state * k_hat[:, None], axis=0)
    delta = beta_t * (v_t - kv_mem)
    state = state + k_hat[:, None] * delta[None, :]

    out_t = jnp.sum(state * q_hat[:, None], axis=0)
    o_ref[b, h, :] = out_t.astype(o_ref.dtype)
    ht_ref[b, h] = state.astype(ht_ref.dtype)


# 训练前向 launcher


def _fwd_out_shape(r, v):
    B, N, T, K = r.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // CHUNK_LEN, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
    ]


def _gdn_recurrent_fwd_pallas_call(q, k, v, g, beta, h0):
    B, N, T, K = q.shape
    return launch(
        "gdn_recurrent_fwd",
        _gdn_recurrent_fwd_kernel,
        _fwd_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


def _gdn_recurrent_fwd_warmup(q, k, v, g, beta, h0):
    B, N, T, K = q.shape
    ensure_config(
        "gdn_recurrent_fwd",
        _gdn_recurrent_fwd_kernel,
        _fwd_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


@custom_partitioning
def _gdn_recurrent_fwd_spmd(q, k, v, g, beta, h0):
    return _gdn_recurrent_fwd_pallas_call(q, k, v, g, beta, h0)


_gdn_recurrent_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_gdn_recurrent_fwd_pallas_call),
)


# 训练反向 launcher


def _bwd_out_shape(r, v):
    B, N, T, K = r.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, K), r.dtype),
        jax.ShapeDtypeStruct((B, N, T, K), r.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]


def _gdn_recurrent_bwd_pallas_call(
    q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0
):
    B, N, T, K = q.shape
    return launch(
        "gdn_recurrent_bwd",
        _gdn_recurrent_bwd_kernel,
        _bwd_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0),
    )


def _gdn_recurrent_bwd_warmup(
    q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0
):
    B, N, T, K = q.shape
    ensure_config(
        "gdn_recurrent_bwd",
        _gdn_recurrent_bwd_kernel,
        _bwd_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0),
    )


@custom_partitioning
def _gdn_recurrent_bwd_spmd(
    q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0
):
    return _gdn_recurrent_bwd_pallas_call(
        q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, chkp, h0
    )


_gdn_recurrent_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_gdn_recurrent_bwd_pallas_call),
)


@jax.custom_vjp
def _gdn_recurrent_train(q, k, v, g, beta, h0):
    _gdn_recurrent_fwd_warmup(q, k, v, g, beta, h0)
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_fwd_spmd(
        q, k, v, g, beta, h0
    )
    final_state = state_chkp[:, :, -1, :, :]
    return out, final_state


def _gdn_train_fwd(q, k, v, g, beta, h0):
    _gdn_recurrent_fwd_warmup(q, k, v, g, beta, h0)
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_fwd_spmd(
        q, k, v, g, beta, h0
    )
    final_state = state_chkp[:, :, -1, :, :]
    return (out, final_state), (
        q,
        k,
        v,
        g,
        beta,
        kv_mem,
        inv_norm_q,
        inv_norm_k,
        state_chkp,
        h0,
    )


def _gdn_train_bwd(res, grads):
    q, k, v, g, beta, kv_mem, inv_norm_q, inv_norm_k, state_chkp, h0 = res
    dy, dht = grads
    dy = jnp.asarray(dy, q.dtype)
    if dht is None:
        B, N, T, K = q.shape
        V = v.shape[-1]
        dht = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    _gdn_recurrent_bwd_warmup(
        q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, state_chkp, h0
    )
    dq, dk, dv, dg, dbeta, dh0 = _gdn_recurrent_bwd_spmd(
        q, k, v, g, beta, dy, dht, kv_mem, inv_norm_q, inv_norm_k, state_chkp, h0
    )
    return dq, dk, dv, dg, dbeta, dh0


_gdn_recurrent_train.defvjp(_gdn_train_fwd, _gdn_train_bwd)


# 推理前向 launcher


def _inf_out_shape(r, v):
    B, N, T, K = r.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]


def _gdn_recurrent_inf_pallas_call(q, k, v, g, beta, h0):
    B, N, T, K = q.shape
    return launch(
        "gdn_recurrent_inf",
        _gdn_recurrent_inf_kernel,
        _inf_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


def _gdn_recurrent_inf_warmup(q, k, v, g, beta, h0):
    B, N, T, K = q.shape
    ensure_config(
        "gdn_recurrent_inf",
        _gdn_recurrent_inf_kernel,
        _inf_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


@custom_partitioning
def _gdn_recurrent_inf_spmd(q, k, v, g, beta, h0):
    return _gdn_recurrent_inf_pallas_call(q, k, v, g, beta, h0)


_gdn_recurrent_inf_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=INF_RULE,
    partition=create_partition(_gdn_recurrent_inf_pallas_call),
)


# 单步 RNN launcher


def _single_out_shape(q, v):
    B, N, K = q.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]


def _gdn_recurrent_single_step_pallas_call(q, k, v, g, beta, h0):
    B, N, K = q.shape
    return launch(
        "gdn_recurrent_single_step",
        _gdn_recurrent_single_step_kernel,
        _single_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


def _gdn_recurrent_single_step_warmup(q, k, v, g, beta, h0):
    B, N, K = q.shape
    ensure_config(
        "gdn_recurrent_single_step",
        _gdn_recurrent_single_step_kernel,
        _single_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, h0),
    )


@custom_partitioning
def _gdn_recurrent_single_step_spmd(q, k, v, g, beta, h0):
    return _gdn_recurrent_single_step_pallas_call(q, k, v, g, beta, h0)


_gdn_recurrent_single_step_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=SINGLE_RULE,
    partition=create_partition(_gdn_recurrent_single_step_pallas_call),
)


# 对外 API


def gated_delta_net_recurrent(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Gated DeltaNet recurrent 训练算子（JAX Pallas 实现）。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先（[B, H, T, *]）。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；仅当 output_final_state=True 时返回。

    Raises:
        ValueError: T 不被 CHUNK_LEN 整除。
    """
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
    beta = _transpose_head(beta, head_first)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % CHUNK_LEN != 0:
        raise ValueError(
            f"Pallas kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    h0 = _prepare_h0(initial_state, B, N, K, V)
    out, final_state = _gdn_recurrent_train(q, k, v, g, beta, h0)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, final_state
    return out


def gated_delta_net_recurrent_inference(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = False,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Gated DeltaNet recurrent 推理算子（JAX Pallas 实现，无梯度）。"""
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
    beta = _transpose_head(beta, head_first)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % CHUNK_LEN != 0:
        raise ValueError(
            f"Pallas kernel requires sequence length T={T} to be divisible by {CHUNK_LEN}"
        )

    h0 = _prepare_h0(initial_state, B, N, K, V)
    _gdn_recurrent_inf_warmup(q, k, v, g, beta, h0)
    out, final_state = _gdn_recurrent_inf_spmd(q, k, v, g, beta, h0)

    out = jnp.transpose(out, (0, 2, 1, 3))
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, final_state
    return out


def gated_delta_net_recurrent_single_step(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    g: jnp.ndarray,
    beta: jnp.ndarray,
    initial_state: Optional[jnp.ndarray] = None,
    output_final_state: bool = True,
    head_first: bool = True,
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Gated DeltaNet recurrent 单步 RNN 算子（JAX Pallas 实现）。"""
    if not head_first:
        raise NotImplementedError(
            "gated_delta_net_recurrent_single_step currently only supports head_first=True."
        )

    dtype = v.dtype
    B, N, K = q.shape
    V = v.shape[-1]
    h0 = _prepare_h0(initial_state, B, N, K, V)

    _gdn_recurrent_single_step_warmup(q, k, v, g, beta, h0)
    out, next_state = _gdn_recurrent_single_step_spmd(q, k, v, g, beta, h0)
    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, next_state
    return out

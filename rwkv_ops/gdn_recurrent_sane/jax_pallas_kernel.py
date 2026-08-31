"""JAX 版 Gated DeltaNet recurrent SANE Pallas kernel 封装。"""

from __future__ import annotations

import functools
import warnings
from typing import Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import NamedSharding, PartitionSpec

from ..pallas_utils import create_partition, ensure_config, launch


# SPMD 切分规则（Einsum 风格）
# b=Batch, n=Head, t=Time, k=HeadK, v=HeadV, c=Chunk
# 训练前向：tau 带 head 维可 TP，mask 无 head 维需 replicate。
FWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> "
    "b n t v, b n t v, b n c k v, b n t, b n t"
)
BWD_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n t v, b n k v, "
    "b n t v, b n t, b n c k v, b n k v, b n c, b c -> "
    "b n t k, b n t k, b n t v, b n t, b n t, b n k v, b n c"
)
INF_RULE = (
    "b n t k, b n t k, b n t v, b n t, b n t, b n c, b c, b n k v -> b n t v, b n k v"
)
SINGLE_RULE = "b n k, b n k, b n v, b n, b n, b n, b, b n k v -> b n v, b n k v"


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
    """为 g/beta [B, N, T] 构造 sharding。"""
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


def _sharding_for_mask(qs):
    """为 mask [B, C] 构造 sharding（无 head 维，TP 下 replicate）。"""
    spec = _q_spec(qs)
    if spec is None or len(spec) != 4:
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
    """tau 公共接口始终为 [B, T//chunk_size, N]；需要转成 head-first [B, N, T//chunk_size]。"""
    tau = jnp.asarray(tau, dtype=jnp.float32)
    # [B, T//chunk_size, N] -> [B, N, T//chunk_size]
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


def _make_gdn_recurrent_sane_fwd_kernel(chunk_size: int):
    """构造 Gated DeltaNet recurrent SANE 训练前向 Pallas kernel。

    返回的 kernel 在 grid (B, N) 上运行，每个 program 处理一个 (batch, head)。
    chunk 内静态展开 chunk_size 步，保存 SANE 之前的 state checkpoint；
    chunk 边界按 mask 选择是否执行 SANE。
    """

    def _gdn_recurrent_sane_fwd_kernel(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        tau_ref,
        mask_ref,
        h0_ref,
        o_ref,
        kv_mem_ref,
        chkp_ref,
        inv_norm_q_ref,
        inv_norm_k_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = q_ref.shape[2] // chunk_size
        scale = jax.lax.rsqrt(jnp.float32(q_ref.shape[3]))
        eps = jnp.float32(1e-6)

        state = h0_ref[b, h].astype(jnp.float32)

        def chunk_body(c, state):
            for j in range(chunk_size):
                t = c * chunk_size + j

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

            # checkpoint 保存 SANE 之前的 state 供反向使用。
            chkp_ref[b, h, c] = state

            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            u = state / tau_safe
            tnh = jnp.tanh(u)
            sane_state = tau_safe * tnh
            m = mask_ref[b, c].astype(jnp.float32)
            state = state * (1.0 - m) + sane_state * m
            return state

        jax.lax.fori_loop(0, num_chunks, chunk_body, state)

    return _gdn_recurrent_sane_fwd_kernel


def _make_gdn_recurrent_sane_bwd_kernel(chunk_size: int):
    """构造 Gated DeltaNet recurrent SANE 训练反向 Pallas kernel。

    返回的 kernel 在 grid (B, N) 上运行，每个 program 处理一个 (batch, head)。
    在每个 chunk 边界先应用 SANE 反向，再回传 chunk 内 chunk_size 步的 GDN 梯度。
    """

    def _gdn_recurrent_sane_bwd_kernel(
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
        tau_ref,
        mask_ref,
        dq_ref,
        dk_ref,
        dv_ref,
        dg_ref,
        dbeta_ref,
        dh0_ref,
        dtau_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        num_chunks = q_ref.shape[2] // chunk_size
        scale = jax.lax.rsqrt(jnp.float32(q_ref.shape[3]))

        dS = dht_ref[b, h].astype(jnp.float32)

        def chunk_body(c_rev, dS):
            c = num_chunks - 1 - c_rev
            # chkp 保存的是 SANE 之前的 state。
            state = chkp_ref[b, h, c].astype(jnp.float32)

            # 先对下游梯度 dS 应用 SANE 导数（按 mask blend）。
            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            m = mask_ref[b, c].astype(jnp.float32)
            u = state / tau_safe
            tnh = jnp.tanh(u)
            sech2 = 1.0 - tnh * tnh
            blend = (1.0 - m) + m * sech2
            dtau_local = jnp.sum(dS * m * (tnh - u * sech2))
            dS = dS * blend
            dtau_ref[b, h, c] = dtau_local

            for j in range(chunk_size - 1, -1, -1):
                t = c * chunk_size + j

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

    return _gdn_recurrent_sane_bwd_kernel


def _make_gdn_recurrent_sane_inf_kernel(chunk_size: int):
    """构造 Gated DeltaNet recurrent SANE 推理前向 Pallas kernel。

    与训练前向数学一致，但不做反向保存；支持任意序列长度 T。
    """

    def _gdn_recurrent_sane_inf_kernel(
        q_ref,
        k_ref,
        v_ref,
        g_ref,
        beta_ref,
        tau_ref,
        mask_ref,
        h0_ref,
        o_ref,
        ht_ref,
    ):
        b = pl.program_id(0)
        h = pl.program_id(1)
        T = q_ref.shape[2]
        scale = jax.lax.rsqrt(jnp.float32(q_ref.shape[3]))
        eps = jnp.float32(1e-6)
        num_chunks = T // chunk_size
        rem = T - num_chunks * chunk_size

        state = h0_ref[b, h].astype(jnp.float32)

        def step(state, t):
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

        def chunk_body(c, state):
            for j in range(chunk_size):
                state = step(state, c * chunk_size + j)

            tau_v = tau_ref[b, h, c].astype(jnp.float32)
            tau_safe = jnp.maximum(tau_v, 1e-6)
            u = state / tau_safe
            tnh = jnp.tanh(u)
            sane_state = tau_safe * tnh
            m = mask_ref[b, c].astype(jnp.float32)
            return state * (1.0 - m) + sane_state * m

        state = jax.lax.fori_loop(0, num_chunks, chunk_body, state)

        for j in range(rem):
            state = step(state, num_chunks * chunk_size + j)

        ht_ref[b, h] = state.astype(ht_ref.dtype)

    return _gdn_recurrent_sane_inf_kernel


def _gdn_recurrent_sane_single_step_kernel(
    q_ref,
    k_ref,
    v_ref,
    g_ref,
    beta_ref,
    tau_ref,
    do_sane_ref,
    h0_ref,
    o_ref,
    ht_ref,
):
    """Gated DeltaNet recurrent SANE 单步 RNN Pallas kernel。

    输入无时间维：q/k 为 [B, N, K]，v/o 为 [B, N, V]，g/beta 为 [B, N]。
    """
    b = pl.program_id(0)
    h = pl.program_id(1)
    scale = jax.lax.rsqrt(jnp.float32(q_ref.shape[2]))
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

    tau_v = tau_ref[b, h].astype(jnp.float32)
    tau_safe = jnp.maximum(tau_v, 1e-6)
    u = state / tau_safe
    tnh = jnp.tanh(u)
    sane_state = tau_safe * tnh
    m = do_sane_ref[b].astype(jnp.float32)
    state = state * (1.0 - m) + sane_state * m

    ht_ref[b, h] = state.astype(ht_ref.dtype)


# 训练前向 launcher


def _fwd_out_shape(q, v, chunk_size: int):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // chunk_size, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
    ]


def _gdn_recurrent_sane_fwd_pallas_call(
    q, k, v, g, beta, tau, mask, h0, chunk_size: int
):
    B, N, T, K = q.shape
    return launch(
        f"gdn_recurrent_sane_fwd_{chunk_size}",
        _make_gdn_recurrent_sane_fwd_kernel(chunk_size),
        _fwd_out_shape(q, v, chunk_size),
        (B, N),
        (q, k, v, g, beta, tau, mask, h0),
    )


def _gdn_recurrent_sane_fwd_warmup(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    B, N, T, K = q.shape
    ensure_config(
        f"gdn_recurrent_sane_fwd_{chunk_size}",
        _make_gdn_recurrent_sane_fwd_kernel(chunk_size),
        _fwd_out_shape(q, v, chunk_size),
        (B, N),
        (q, k, v, g, beta, tau, mask, h0),
    )


@functools.partial(custom_partitioning, static_argnums=(8,))
def _gdn_recurrent_sane_fwd_spmd(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    return _gdn_recurrent_sane_fwd_pallas_call(
        q, k, v, g, beta, tau, mask, h0, chunk_size
    )


_gdn_recurrent_sane_fwd_spmd.def_partition(
    infer_sharding_from_operands=_fwd_infer_sharding,
    sharding_rule=FWD_RULE,
    partition=create_partition(_gdn_recurrent_sane_fwd_pallas_call),
)


# 训练反向 launcher


def _bwd_out_shape(q, v, chunk_size: int):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, K), q.dtype),
        jax.ShapeDtypeStruct((B, N, T, K), q.dtype),
        jax.ShapeDtypeStruct((B, N, T, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T), jnp.float32),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
        jax.ShapeDtypeStruct((B, N, T // chunk_size), jnp.float32),
    ]


def _gdn_recurrent_sane_bwd_pallas_call(
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
    chkp,
    h0,
    tau,
    mask,
    chunk_size: int,
):
    B, N, T, K = q.shape
    return launch(
        f"gdn_recurrent_sane_bwd_{chunk_size}",
        _make_gdn_recurrent_sane_bwd_kernel(chunk_size),
        _bwd_out_shape(q, v, chunk_size),
        (B, N),
        (
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
            chkp,
            h0,
            tau,
            mask,
        ),
    )


def _gdn_recurrent_sane_bwd_warmup(
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
    chkp,
    h0,
    tau,
    mask,
    chunk_size: int,
):
    B, N, T, K = q.shape
    ensure_config(
        f"gdn_recurrent_sane_bwd_{chunk_size}",
        _make_gdn_recurrent_sane_bwd_kernel(chunk_size),
        _bwd_out_shape(q, v, chunk_size),
        (B, N),
        (
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
            chkp,
            h0,
            tau,
            mask,
        ),
    )


@functools.partial(custom_partitioning, static_argnums=(14,))
def _gdn_recurrent_sane_bwd_spmd(
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
    chkp,
    h0,
    tau,
    mask,
    chunk_size: int,
):
    return _gdn_recurrent_sane_bwd_pallas_call(
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
        chkp,
        h0,
        tau,
        mask,
        chunk_size,
    )


_gdn_recurrent_sane_bwd_spmd.def_partition(
    infer_sharding_from_operands=_bwd_infer_sharding,
    sharding_rule=BWD_RULE,
    partition=create_partition(_gdn_recurrent_sane_bwd_pallas_call),
)


@functools.partial(jax.custom_vjp, nondiff_argnums=(8,))
def _gdn_recurrent_sane_train(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    _gdn_recurrent_sane_fwd_warmup(q, k, v, g, beta, tau, mask, h0, chunk_size)
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_sane_fwd_spmd(
        q, k, v, g, beta, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
    return out, final_state


def _gdn_train_fwd(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    _gdn_recurrent_sane_fwd_warmup(q, k, v, g, beta, tau, mask, h0, chunk_size)
    out, kv_mem, state_chkp, inv_norm_q, inv_norm_k = _gdn_recurrent_sane_fwd_spmd(
        q, k, v, g, beta, tau, mask, h0, chunk_size
    )
    final_state = _apply_sane_to_final_state(state_chkp[:, :, -1, :, :], tau, mask=mask)
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


def _gdn_train_bwd(chunk_size: int, res, grads):
    q, k, v, g, beta, tau, mask, kv_mem, inv_norm_q, inv_norm_k, state_chkp, h0 = res
    dy, dht = grads
    dy = jnp.asarray(dy, q.dtype)
    if dht is None:
        B, N, T, K = q.shape
        V = v.shape[-1]
        dht = jnp.zeros((B, N, K, V), dtype=jnp.float32)
    else:
        dht = jnp.asarray(dht, jnp.float32)

    _gdn_recurrent_sane_bwd_warmup(
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
        state_chkp,
        h0,
        tau,
        mask,
        chunk_size,
    )
    dq, dk, dv, dg, dbeta, dh0, dtau = _gdn_recurrent_sane_bwd_spmd(
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
        state_chkp,
        h0,
        tau,
        mask,
        chunk_size,
    )
    return dq, dk, dv, dg, dbeta, dtau, None, dh0


_gdn_recurrent_sane_train.defvjp(_gdn_train_fwd, _gdn_train_bwd)


# 推理前向 launcher


def _inf_out_shape(q, v):
    B, N, T, K = q.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, T, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]


def _gdn_recurrent_sane_inf_pallas_call(
    q, k, v, g, beta, tau, mask, h0, chunk_size: int
):
    B, N, T, K = q.shape
    return launch(
        f"gdn_recurrent_sane_inf_{chunk_size}",
        _make_gdn_recurrent_sane_inf_kernel(chunk_size),
        _inf_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, tau, mask, h0),
    )


def _gdn_recurrent_sane_inf_warmup(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    B, N, T, K = q.shape
    ensure_config(
        f"gdn_recurrent_sane_inf_{chunk_size}",
        _make_gdn_recurrent_sane_inf_kernel(chunk_size),
        _inf_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, tau, mask, h0),
    )


@functools.partial(custom_partitioning, static_argnums=(8,))
def _gdn_recurrent_sane_inf_spmd(q, k, v, g, beta, tau, mask, h0, chunk_size: int):
    return _gdn_recurrent_sane_inf_pallas_call(
        q, k, v, g, beta, tau, mask, h0, chunk_size
    )


_gdn_recurrent_sane_inf_spmd.def_partition(
    infer_sharding_from_operands=_inf_infer_sharding,
    sharding_rule=INF_RULE,
    partition=create_partition(_gdn_recurrent_sane_inf_pallas_call),
)


# 单步 RNN launcher


def _single_out_shape(q, v):
    B, N, K = q.shape
    V = v.shape[-1]
    return [
        jax.ShapeDtypeStruct((B, N, V), v.dtype),
        jax.ShapeDtypeStruct((B, N, K, V), jnp.float32),
    ]


def _gdn_recurrent_sane_single_step_pallas_call(q, k, v, g, beta, tau, do_sane, h0):
    B, N, K = q.shape
    return launch(
        "gdn_recurrent_sane_single_step",
        _gdn_recurrent_sane_single_step_kernel,
        _single_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, tau, do_sane, h0),
    )


def _gdn_recurrent_sane_single_step_warmup(q, k, v, g, beta, tau, do_sane, h0):
    B, N, K = q.shape
    ensure_config(
        "gdn_recurrent_sane_single_step",
        _gdn_recurrent_sane_single_step_kernel,
        _single_out_shape(q, v),
        (B, N),
        (q, k, v, g, beta, tau, do_sane, h0),
    )


@custom_partitioning
def _gdn_recurrent_sane_single_step_spmd(q, k, v, g, beta, tau, do_sane, h0):
    return _gdn_recurrent_sane_single_step_pallas_call(
        q, k, v, g, beta, tau, do_sane, h0
    )


_gdn_recurrent_sane_single_step_spmd.def_partition(
    infer_sharding_from_operands=_single_infer_sharding,
    sharding_rule=SINGLE_RULE,
    partition=create_partition(_gdn_recurrent_sane_single_step_pallas_call),
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
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 训练算子（JAX Pallas）。

    在 chunk 边界（每 chunk_size 个 token）按 mask 对 state 执行
    `state = tau * tanh(state / tau)`；输出始终基于 SANE 之前的 state。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，float32。阈值，必须 > 0。
        mask: [B, T//chunk_size]，float32 或 None。>0 的 chunk 边界执行 SANE；
            仅当 output_final_state=True 时生效。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先 ([B, H, T, *])。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32。
            output_final_state=False 时为 None；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: T 不被 chunk_size 整除，或 tau/mask 形状不匹配。
    """
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
    beta = _transpose_head(beta, head_first)
    tau = _transpose_tau(tau)

    B, N, T, K = q.shape
    V = v.shape[-1]
    if T % chunk_size != 0:
        raise ValueError(
            f"Pallas SANE kernel requires sequence length T={T} to be divisible by chunk_size={chunk_size}"
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

    h0 = _prepare_h0(initial_state, B, N, K, V)

    out, final_state = _gdn_recurrent_sane_train(
        q, k, v, g, beta, tau, mask_arr, h0, chunk_size
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

    final_state = _apply_sane_to_final_state(final_state, tau, mask=mask_arr)
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
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 推理算子（JAX Pallas）。

    与训练版数学一致但不保存反向中间量，支持任意序列长度 T。

    Args:
        q, k: [B, T, H, K]，查询与键。
        v: [B, T, H, V]，值。
        g: [B, T, H]，log-space decay。
        beta: [B, T, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, T//chunk_size, H]，float32。
        mask: [B, T//chunk_size]，float32 或 None。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回最终 state。
        head_first: bool，输入输出是否 head 维优先 ([B, H, T, *])。
        chunk_size: int，chunk 长度，默认 16。

    Returns:
        out: [B, T, H, V]，与 v 同 dtype。
        final_state: [B, H, K, V]，float32；
            output_final_state=False 时为 None；
            output_final_state=True 且 mask=None 时也为 None 并发出 UserWarning。

    Raises:
        ValueError: tau/mask 形状不匹配。
    """
    dtype = v.dtype
    q = _transpose_head(q, head_first)
    k = _transpose_head(k, head_first)
    v = _transpose_head(v, head_first)
    g = _transpose_head(g, head_first)
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

    h0 = _prepare_h0(initial_state, B, N, K, V)

    _gdn_recurrent_sane_inf_warmup(q, k, v, g, beta, tau, mask_arr, h0, chunk_size)
    out, final_state = _gdn_recurrent_sane_inf_spmd(
        q, k, v, g, beta, tau, mask_arr, h0, chunk_size
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
) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    """带 State Anomaly Neutralization 的 Gated DeltaNet recurrent 单步 RNN（JAX Pallas）。

    Args:
        q, k: [B, H, K]，查询与键。
        v: [B, H, V]，值。
        g: [B, H]，log-space decay。
        beta: [B, H]，写入强度，必须已在外部过 sigmoid。
        tau: [B, H]，float32。
        do_sane: [B]，float32。>0 执行 SANE。
        initial_state: [B, H, K, V] 或 [1, H, K, V]，float32，可选。
        output_final_state: bool，是否返回下一步 state。
        head_first: bool，输入输出是否 head 维优先。单步默认 True（[B, H, *]）。
        chunk_size: int，chunk 长度，默认 16。单步实现忽略该参数（仅签名一致）。

    Returns:
        out: [B, H, V]，与 v 同 dtype。
        next_state: [B, H, K, V]，float32；output_final_state=False 时为 None。

    Raises:
        NotImplementedError: head_first=False 尚未支持。
        ValueError: tau/do_sane 形状不匹配。
    """
    if not head_first:
        raise NotImplementedError(
            "gated_delta_net_recurrent_sane_single_step currently only supports head_first=True."
        )

    dtype = v.dtype
    q = jnp.asarray(q, dtype=jnp.float32)
    k = jnp.asarray(k, dtype=jnp.float32)
    v = jnp.asarray(v, dtype=jnp.float32)
    g = jnp.asarray(g, dtype=jnp.float32)
    beta = jnp.asarray(beta, dtype=jnp.float32)
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

    _gdn_recurrent_sane_single_step_warmup(q, k, v, g, beta, tau, do_sane, h0)
    out, next_state = _gdn_recurrent_sane_single_step_spmd(
        q, k, v, g, beta, tau, do_sane, h0
    )

    out = jnp.asarray(out, dtype)

    if output_final_state:
        return out, next_state
    return out, None
